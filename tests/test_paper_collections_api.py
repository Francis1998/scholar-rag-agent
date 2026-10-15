"""Saved selections must be durable, atomic, and fail closed at both API boundaries."""

import asyncio
import json
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from scripts.demo_evidence_export import offline_settings

from api.application import create_app
from api.dependencies import AppContainer
from retrieval.models import Document

UNKNOWN_COLLECTION = "col_" + "0" * 32


def denied(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("Collection metadata and previews must not generate or journal.")


@dataclass
class CollectionAPI:
    application: FastAPI
    client: TestClient
    path: Path
    first: str
    second: str

    @property
    def container(self) -> AppContainer:
        return self.application.state.container  # type: ignore[no-any-return]

    def create(self, name: str = "Reading list", ids: list[str] | None = None) -> dict[str, Any]:
        response = self.client.post(
            "/collections",
            json={"name": name, "document_ids": ids if ids is not None else [self.first]},
        )
        assert response.status_code == 201, response.text
        assert_private(response)
        result: dict[str, Any] = response.json()
        assert response.headers["location"] == f"/collections/{result['collection_id']}"
        return result


def assert_private(response: httpx.Response) -> None:
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"


@pytest.fixture
def collection_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[CollectionAPI]:
    path = tmp_path / "collections.sqlite3"
    application = create_app(offline_settings(path))
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", denied)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", denied)
    with TestClient(application) as client:
        identifiers = []
        for title, marker in (("Selected", "SELECTED_MARKER"), ("Excluded", "EXCLUDED_MARKER")):
            response = client.post(
                "/ingest/text",
                json={
                    "title": title,
                    "text": f"Synthetic data only. GraphRAG connects Retrieval {marker}.",
                    "source": "synthetic:collections",
                },
            )
            response.raise_for_status()
            identifiers.append(response.json()["document_id"])
        yield CollectionAPI(application, client, path, *identifiers)


def test_create_restart_list_detail_replace_and_conditional_delete(
    collection_api: CollectionAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = collection_api
    initial_chunks = api.container.document_store.list_chunks()
    with monkeypatch.context() as patch:
        for target, method in (
            (api.container.llm, "generate"),
            (api.container.runner, "run"),
            (api.container.runner, "preview"),
            (api.container.event_log, "append_event"),
            (api.container.event_log, "append_transition"),
        ):
            patch.setattr(target, method, denied)
        saved = api.create("  Methods  ", [f" {api.first} ", api.first, api.second])
    assert set(saved) == {"collection_id", "name", "document_ids", "revision"}
    assert saved["name"] == "Methods"
    assert saved["document_ids"] == [api.first, api.second]
    assert saved["revision"] == 1
    path = f"/collections/{saved['collection_id']}"

    api.application.state.container = AppContainer(offline_settings(api.path))
    assert api.client.get(path).json() == saved
    listing = api.client.get("/collections")
    assert_private(listing)
    assert listing.json() == {
        "collections": [
            {
                "collection_id": saved["collection_id"],
                "name": "Methods",
                "revision": 1,
                "document_count": 2,
            }
        ],
        "next_cursor": None,
    }
    update = api.client.put(
        path, json={"name": "Revised methods", "document_ids": [api.second], "expected_revision": 1}
    )
    assert update.status_code == 200, update.text
    assert_private(update)
    revised = {**saved, "name": "Revised methods", "document_ids": [api.second], "revision": 2}
    assert update.json() == revised
    for method in ("put", "delete"):
        kwargs = (
            {"json": {"name": "Stale", "document_ids": [api.first], "expected_revision": 1}}
            if method == "put"
            else {"params": {"expected_revision": 1}}
        )
        response = getattr(api.client, method)(path, **kwargs)
        assert response.status_code == 409
        assert response.json()["detail"]["code"] == "collection_revision_conflict"
        assert api.client.get(path).json() == revised
    deleted = api.client.delete(path, params={"expected_revision": 2})
    assert deleted.status_code == 204 and deleted.content == b""
    assert_private(deleted)
    assert api.client.get(path).status_code == 404
    assert api.client.get("/collections").json() == {"collections": [], "next_cursor": None}
    assert api.container.document_store.list_chunks() == initial_chunks
    assert len(api.client.get("/documents").json()["documents"]) == 2
    assert api.container.event_log.list_events() == []


def test_collection_scope_reaches_query_preview_and_historical_export(
    collection_api: CollectionAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = collection_api
    saved = api.create()
    body = {"query": "Compare GraphRAG versus Retrieval", "collection_id": saved["collection_id"]}
    with monkeypatch.context() as patch:
        patch.setattr(api.container.llm, "generate", denied)
        patch.setattr(api.container.event_log, "append_event", denied)
        patch.setattr(api.container.event_log, "append_transition", denied)
        preview = api.client.post("/retrieve", json=body)
    assert preview.status_code == 200, preview.text
    assert preview.json()["plan"]["observation"]["document_ids"] == [api.first]
    assert {s["chunk"]["document_id"] for s in preview.json()["sources"]} == {api.first}
    assert "EXCLUDED_MARKER" not in preview.text
    assert api.container.event_log.list_events() == []
    result = api.client.post("/query", json=body)
    assert result.status_code == 200, result.text
    assert_private(result)
    run = result.json()["result"]
    assert run["state"] == "DONE", run
    assert run["plan"]["observation"]["document_ids"] == [api.first]
    assert len(run["plan"]["tasks"]) == 2
    assert {c["document_id"] for c in run["answer"]["citations"]} == {api.first}
    export_path = f"/runs/{run['run_id']}/export"
    original = {
        fmt: api.client.get(export_path, params={"format": fmt}).content
        for fmt in ("json", "markdown")
    }
    snapshot = api.client.get(export_path).json()["snapshot"]
    assert snapshot["sources"] == preview.json()["sources"]
    assert snapshot["request"]["context"] == preview.json()["context"]
    assert "EXCLUDED_MARKER" not in snapshot["request"]["context"]
    before_events = api.container.event_log.list_events()
    path = f"/collections/{saved['collection_id']}"
    assert (
        api.client.put(
            path, json={"name": "Changed", "document_ids": [api.second], "expected_revision": 1}
        ).status_code
        == 200
    )
    assert api.client.delete(path, params={"expected_revision": 2}).status_code == 204
    api.application.state.container = AppContainer(offline_settings(api.path))
    for fmt, content in original.items():
        after = api.client.get(export_path, params={"format": fmt})
        assert after.status_code == 200 and after.content == content
    assert api.container.event_log.list_events() == before_events
    assert len(api.client.get("/documents").json()["documents"]) == 2


@pytest.mark.parametrize("endpoint", ["/query", "/retrieve"])
@pytest.mark.parametrize("value", [None, "", " ", "x" * 129, 42, [], "not-a-collection"])
def test_invalid_collection_scope_never_runs(
    collection_api: CollectionAPI, monkeypatch: pytest.MonkeyPatch, endpoint: str, value: object
) -> None:
    monkeypatch.setattr(collection_api.container.runner, "run", denied)
    monkeypatch.setattr(collection_api.container.runner, "preview", denied)
    response = collection_api.client.post(
        endpoint, json={"query": "GraphRAG", "collection_id": value}
    )
    assert response.status_code == 422
    assert collection_api.container.event_log.list_events() == []


@pytest.mark.parametrize("endpoint", ["/query", "/retrieve"])
def test_unknown_and_ambiguous_scope_never_widens(
    collection_api: CollectionAPI, monkeypatch: pytest.MonkeyPatch, endpoint: str
) -> None:
    api = collection_api
    monkeypatch.setattr(api.container.runner, "run", denied)
    monkeypatch.setattr(api.container.runner, "preview", denied)
    response = api.client.post(
        endpoint, json={"query": "GraphRAG", "collection_id": UNKNOWN_COLLECTION}
    )
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "collection_not_found"
    assert_private(response)
    for document_ids in ([api.first], [], None):
        response = api.client.post(
            endpoint,
            json={
                "query": "GraphRAG",
                "collection_id": UNKNOWN_COLLECTION,
                "document_ids": document_ids,
            },
        )
        assert response.status_code == 422
    assert api.container.event_log.list_events() == []


def test_missing_documents_and_name_conflicts_are_atomic(collection_api: CollectionAPI) -> None:
    api = collection_api
    saved = api.create()
    path = f"/collections/{saved['collection_id']}"
    missing = api.client.post(
        "/collections", json={"name": "Missing", "document_ids": [api.first, "missing"]}
    )
    assert missing.status_code == 422
    assert missing.json()["detail"] == {
        "code": "unknown_documents",
        "message": "All selected documents must exist in the corpus.",
        "missing_document_ids": ["missing"],
    }
    assert len(api.client.get("/collections").json()["collections"]) == 1
    replacement = api.client.put(
        path,
        json={"name": "Renamed", "document_ids": [api.second, "missing"], "expected_revision": 1},
    )
    assert replacement.status_code == 422
    assert api.client.get(path).json() == saved
    duplicate = api.client.post(
        "/collections", json={"name": " Reading list ", "document_ids": [api.second]}
    )
    assert duplicate.status_code == 409
    assert duplicate.json()["detail"]["code"] == "collection_name_conflict"
    other = api.create("Other", [api.second])
    conflict = api.client.put(
        path, json={"name": other["name"], "document_ids": [api.second], "expected_revision": 1}
    )
    assert conflict.status_code == 409
    assert api.client.get(path).json() == saved


@pytest.mark.parametrize("endpoint", ["/query", "/retrieve"])
def test_deleted_member_is_an_error_not_an_empty_or_global_scope(
    collection_api: CollectionAPI, monkeypatch: pytest.MonkeyPatch, endpoint: str
) -> None:
    api = collection_api
    saved = api.create()
    with sqlite3.connect(api.path) as connection:
        connection.execute("DELETE FROM documents WHERE document_id = ?", (api.first,))
    monkeypatch.setattr(api.container.runner, "run", denied)
    monkeypatch.setattr(api.container.runner, "preview", denied)
    response = api.client.post(
        endpoint, json={"query": "GraphRAG", "collection_id": saved["collection_id"]}
    )
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "collection_documents_missing"
    assert api.container.event_log.list_events() == []


@pytest.mark.parametrize("endpoint", ["/query", "/retrieve"])
async def test_resolved_snapshot_survives_await_and_concurrent_collections(
    collection_api: CollectionAPI, monkeypatch: pytest.MonkeyPatch, endpoint: str
) -> None:
    api = collection_api
    a, b = api.create("A"), api.create("B", [api.second])
    entered, release = asyncio.Event(), asyncio.Event()
    hyde = api.container.hybrid_retriever._hyde_expander
    expand = hyde.expand

    async def held(query: str) -> str:
        if query == "first GraphRAG":
            entered.set()
            await release.wait()
        return await expand(query)

    monkeypatch.setattr(hyde, "expand", held)
    transport = httpx.ASGITransport(app=api.application)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first = asyncio.create_task(
            client.post(
                endpoint, json={"query": "first GraphRAG", "collection_id": a["collection_id"]}
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), timeout=3)
            changed = await client.put(
                f"/collections/{a['collection_id']}",
                json={"name": "A", "document_ids": [api.second], "expected_revision": 1},
            )
            assert changed.status_code == 200
            deleted = await client.delete(
                f"/collections/{a['collection_id']}", params={"expected_revision": 2}
            )
            assert deleted.status_code == 204
            second, unscoped = await asyncio.wait_for(
                asyncio.gather(
                    client.post(
                        endpoint, json={"query": "GraphRAG", "collection_id": b["collection_id"]}
                    ),
                    client.post(endpoint, json={"query": "GraphRAG"}),
                ),
                timeout=5,
            )
            assert not first.done()
        finally:
            release.set()
        first_response = await asyncio.wait_for(first, timeout=3)
    for response, expected in zip(
        (first_response, second, unscoped),
        ([api.first], [api.second], None),
        strict=True,
    ):
        assert response.status_code == 200, response.text
        result = response.json()["result"] if endpoint == "/query" else response.json()
        assert result["plan"]["observation"]["document_ids"] == expected
        sources = (
            api.client.get(f"/runs/{result['run_id']}/export").json()["snapshot"]["sources"]
            if endpoint == "/query"
            else result["sources"]
        )
        assert {s["chunk"]["document_id"] for s in sources} == set(
            expected if expected is not None else [api.first, api.second]
        )


def test_unscoped_query_still_calls_legacy_runner_without_new_keywords(
    collection_api: CollectionAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = collection_api
    original = api.container.runner.run

    async def legacy(query: str) -> object:
        return await original(query)

    spy = AsyncMock(side_effect=legacy)
    monkeypatch.setattr(api.container.runner, "run", spy)
    response = api.client.post("/query", json={"query": "GraphRAG"})
    assert response.status_code == 200
    spy.assert_awaited_once_with("GraphRAG")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", None),
        ("name", ""),
        ("name", " \t "),
        ("name", "n" * 121),
        ("name", 42),
        ("name", "two\nlines"),
        ("name", "nul\x00name"),
        ("document_ids", None),
        ("document_ids", []),
        ("document_ids", "doc"),
        ("document_ids", [" "]),
        ("document_ids", [42]),
        ("document_ids", ["d" * 129]),
        ("document_ids", ["repeat"] * 101),
        ("unexpected", "value"),
    ],
)
def test_invalid_selections_do_not_create_or_replace(
    collection_api: CollectionAPI, field: str, value: object
) -> None:
    api = collection_api
    saved = api.create()
    body = {"name": "Valid", "document_ids": [api.first], field: value}
    response = api.client.post("/collections", json=body)
    assert response.status_code == 422, response.text
    path = f"/collections/{saved['collection_id']}"
    response = api.client.put(path, json={**body, "expected_revision": 1})
    assert response.status_code == 422, response.text
    assert api.client.get(path).json() == saved
    assert len(api.client.get("/collections").json()["collections"]) == 1


@pytest.mark.parametrize("method", ["POST", "PUT"])
@pytest.mark.parametrize("mounted", [False, True])
@pytest.mark.parametrize(
    "name",
    [
        "\ud800",
        "\udfff",
        "PRIVATE_COLLECTION_\ud800_NAME",
        ["PRIVATE_COLLECTION_\ud800_NAME"],
        {"nested": ["PRIVATE_COLLECTION_\udfff_NAME"]},
        {"\ud800": "PRIVATE_COLLECTION_NAME"},
    ],
)
def test_escaped_surrogate_names_return_422_without_writes(
    collection_api: CollectionAPI,
    caplog: pytest.LogCaptureFixture,
    method: str,
    mounted: bool,
    name: object,
) -> None:
    api = collection_api
    saved = api.create()
    path = f"/collections/{saved['collection_id']}"
    body: dict[str, object] = {"name": name, "document_ids": [api.second]}
    if method == "PUT":
        body["expected_revision"] = saved["revision"]
    application = api.application
    prefix = ""
    if mounted:
        application = FastAPI()
        application.mount("/library", api.application)
        prefix = "/library"
    before = api.path.read_bytes()
    with TestClient(application, raise_server_exceptions=False) as client:
        response = client.request(
            method,
            prefix + ("/collections" if method == "POST" else path),
            content=json.dumps(body, ensure_ascii=True).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
    assert api.path.read_bytes() == before
    assert api.client.get(path).json() == saved
    assert len(api.client.get("/collections").json()["collections"]) == 1
    assert response.status_code == 422, response.text
    assert response.json() == {
        "detail": [
            {
                "loc": ["body", "name"],
                "type": "string_unicode" if isinstance(name, str) else "string_type",
                "msg": (
                    "Input should be a valid string, unable to parse raw data as a unicode string"
                    if isinstance(name, str)
                    else "Input should be a valid string"
                ),
            }
        ]
    }
    assert_private(response)
    assert "PRIVATE_COLLECTION" not in response.text
    assert "PRIVATE_COLLECTION" not in caplog.text
    assert "unserializable_diagnostic" in caplog.text


@pytest.mark.parametrize("method", ["POST", "PUT"])
def test_mixed_validation_errors_preserve_encodable_input_and_context(
    collection_api: CollectionAPI, method: str
) -> None:
    api = collection_api
    saved = api.create()
    path = "/collections" if method == "POST" else f"/collections/{saved['collection_id']}"
    body: dict[str, object] = {
        "name": None,
        "document_ids": [],
        "unexpected": "ordinary diagnostic input",
    }
    if method == "PUT":
        body["expected_revision"] = "1"
    baseline = api.client.request(method, path, json=body)
    assert baseline.status_code == 422
    ordinary = [error for error in baseline.json()["detail"] if error["loc"] != ["body", "name"]]
    assert any("input" in error and "ctx" in error for error in ordinary)
    body["name"] = "PRIVATE_COLLECTION_\ud800_NAME"
    response = api.client.request(
        method,
        path,
        content=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert [error for error in errors if error["loc"] != ["body", "name"]] == ordinary
    assert set(errors[0]) == {"loc", "type", "msg"}
    assert "PRIVATE_COLLECTION" not in response.text
    assert api.client.get(f"/collections/{saved['collection_id']}").json() == saved


@pytest.mark.parametrize("method", ["POST", "PUT"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("surrogate_name", [False, True])
def test_nonfinite_validation_inputs_return_422_without_writes(
    collection_api: CollectionAPI,
    caplog: pytest.LogCaptureFixture,
    method: str,
    value: float,
    surrogate_name: bool,
) -> None:
    api = collection_api
    saved = api.create()
    path = f"/collections/{saved['collection_id']}"
    body: dict[str, object] = {
        "name": "PRIVATE_COLLECTION_\ud800_NAME" if surrogate_name else "Valid",
        "document_ids": [value],
    }
    if method == "PUT":
        body["expected_revision"] = saved["revision"]
    before = api.path.read_bytes()
    with TestClient(api.application, raise_server_exceptions=False) as client:
        response = client.request(
            method,
            "/collections" if method == "POST" else path,
            content=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
    assert api.path.read_bytes() == before
    assert api.client.get(path).json() == saved
    assert len(api.client.get("/collections").json()["collections"]) == 1
    assert response.status_code == 422, response.text
    expected = [
        {
            "loc": ["body", "document_ids", 0],
            "type": "string_type",
            "msg": "Input should be a valid string",
        },
        {
            "loc": ["body", "document_ids"],
            "type": "too_short",
            "msg": "Tuple should have at least 1 item after validation, not 0",
        },
    ]
    if surrogate_name:
        expected.insert(
            0,
            {
                "loc": ["body", "name"],
                "type": "string_unicode",
                "msg": (
                    "Input should be a valid string, unable to parse raw data as a unicode string"
                ),
            },
        )
    assert response.json() == {"detail": expected}
    assert_private(response)
    assert "PRIVATE_COLLECTION" not in response.text
    assert "PRIVATE_COLLECTION" not in caplog.text
    assert "unserializable_diagnostic" in caplog.text


@pytest.mark.parametrize("location", [("body", "name"), ("body", "\ud800")])
@pytest.mark.parametrize("context", ["PRIVATE_COLLECTION_\ud800_CONTEXT", float("nan")])
def test_unserializable_validation_context_is_not_echoed(
    collection_api: CollectionAPI,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    location: tuple[str, ...],
    context: object,
) -> None:
    error = {
        "loc": location,
        "type": "value_error",
        "msg": "Invalid name.",
        "input": "PRIVATE_COLLECTION_INPUT",
        "ctx": {"nested": [context]},
    }

    def reject(**kwargs: object) -> NoReturn:
        raise RequestValidationError([error])

    monkeypatch.setattr(collection_api.container.paper_collections, "create", reject)
    response = collection_api.client.post(
        "/collections", json={"name": "Valid", "document_ids": [collection_api.first]}
    )
    assert response.status_code == 422
    assert response.json() == {
        "detail": [{"loc": list(location), "type": "value_error", "msg": "Invalid name."}]
    }
    assert "PRIVATE_COLLECTION" not in response.text
    assert "PRIVATE_COLLECTION" not in caplog.text


def test_unencodable_object_keys_return_sanitized_422(collection_api: CollectionAPI) -> None:
    response = collection_api.client.post(
        "/collections",
        content=json.dumps(
            {"name": "Valid", "document_ids": [collection_api.first], "\ud800": "PRIVATE_VALUE"}
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json() == {
        "detail": [
            {
                "loc": ["body"],
                "type": "string_unicode",
                "msg": (
                    "Input should be a valid string, unable to parse raw data as a unicode string"
                ),
            }
        ]
    }
    assert "PRIVATE_VALUE" not in response.text
    response.content.decode("utf-8")


@pytest.mark.parametrize("name", [None, 42, ["\U0001f9ea"], {"nested": ["e\u0301"]}])
def test_encodable_validation_details_are_unchanged(
    collection_api: CollectionAPI, name: object
) -> None:
    response = collection_api.client.post(
        "/collections", json={"name": name, "document_ids": [collection_api.first]}
    )
    assert response.status_code == 422
    assert response.json() == {
        "detail": [
            {
                "loc": ["body", "name"],
                "type": "string_type",
                "msg": "Input should be a valid string",
                "input": name,
            }
        ]
    }
    assert "cache-control" not in response.headers


def test_valid_unicode_names_keep_identity_membership_and_revision(
    collection_api: CollectionAPI,
) -> None:
    api = collection_api
    composed = api.create(" \u00e9 ", [api.first, api.second, api.first])
    decomposed = api.create(" e\u0301 ", [api.second])
    assert composed["name"] == "\u00e9" and composed["document_ids"] == [api.first, api.second]
    assert decomposed["name"] == "e\u0301"
    assert composed["collection_id"] != decomposed["collection_id"]
    path = f"/collections/{composed['collection_id']}"
    name = "\U0001f9ea" * 120
    body = {"name": f" {name} ", "document_ids": [api.second], "expected_revision": 1}
    response = api.client.put(path, json=body)
    assert response.status_code == 200
    revised = {**composed, "name": name, "document_ids": [api.second], "revision": 2}
    assert response.json() == revised
    duplicate = api.client.post("/collections", json={"name": name, "document_ids": [api.first]})
    assert duplicate.status_code == 409
    assert duplicate.json()["detail"]["code"] == "collection_name_conflict"
    noop = api.client.put(path, json={**body, "expected_revision": 2})
    assert noop.status_code == 200
    assert noop.json() == {**revised, "revision": 3}
    stale = api.client.put(path, json=body)
    assert stale.status_code == 409
    assert stale.json()["detail"]["code"] == "collection_revision_conflict"
    api.application.state.container = AppContainer(offline_settings(api.path))
    assert api.client.get(path).json() == noop.json()
    assert api.client.get(f"/collections/{decomposed['collection_id']}").json() == decomposed


@pytest.mark.parametrize(
    "error",
    [
        UnicodeEncodeError("utf-8", "\ud800", 0, 1, "runtime failure"),
        ValueError("runtime failure"),
        TypeError("runtime failure"),
    ],
)
def test_runtime_errors_are_not_reclassified_as_validation(
    collection_api: CollectionAPI, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    def fail(**kwargs: object) -> NoReturn:
        raise error

    monkeypatch.setattr(collection_api.container.paper_collections, "create", fail)
    with pytest.raises(type(error), match="runtime failure"):
        collection_api.client.post(
            "/collections", json={"name": "Valid", "document_ids": [collection_api.first]}
        )


@pytest.mark.parametrize("revision", [None, True, "1", 0, -1, 1.5, 2**63])
def test_replace_requires_strict_revision(collection_api: CollectionAPI, revision: object) -> None:
    api = collection_api
    saved = api.create()
    path = f"/collections/{saved['collection_id']}"
    response = api.client.put(
        path,
        json={"name": "Changed", "document_ids": [api.second], "expected_revision": revision},
    )
    assert response.status_code == 422
    assert api.client.get(path).json() == saved


def test_required_fields_and_delete_precondition(collection_api: CollectionAPI) -> None:
    api = collection_api
    for body in ({}, {"name": "Missing members"}, {"document_ids": [api.first]}):
        assert api.client.post("/collections", json=body).status_code == 422
    saved = api.create()
    path = f"/collections/{saved['collection_id']}"
    assert (
        api.client.put(
            path, json={"name": "Missing revision", "document_ids": [api.first]}
        ).status_code
        == 422
    )
    assert api.client.delete(path).status_code == 422
    for revision in ("0", "-1", "true", "1.5", str(2**63)):
        assert api.client.delete(path, params={"expected_revision": revision}).status_code == 422
    assert api.client.get(path).json() == saved


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("limit", "0"),
        ("limit", "101"),
        ("limit", "1.5"),
        ("limit", "true"),
        ("cursor", ""),
        ("cursor", "unknown"),
        ("cursor", " " + UNKNOWN_COLLECTION),
        ("cursor", "col_" + "a" * 33),
    ],
)
def test_invalid_collection_pages(
    collection_api: CollectionAPI, parameter: str, value: str
) -> None:
    assert collection_api.client.get("/collections", params={parameter: value}).status_code == 422


def test_bounded_keyset_pages_are_exclusive_and_restart_stable(
    collection_api: CollectionAPI,
) -> None:
    api = collection_api
    assert api.client.get("/collections").json() == {"collections": [], "next_cursor": None}
    saved = sorted(
        [api.create(f"Selection {index:03}") for index in range(101)],
        key=lambda item: item["collection_id"],
    )
    assert len(api.client.get("/collections").json()["collections"]) == 20
    first = api.client.get("/collections", params={"limit": 100})
    assert len(first.json()["collections"]) == 100
    assert first.json()["next_cursor"] == saved[99]["collection_id"]
    api.application.state.container = AppContainer(offline_settings(api.path))
    assert api.client.get("/collections", params={"limit": 100}).content == first.content
    cursor = first.json()["next_cursor"]
    assert (
        api.client.delete(f"/collections/{cursor}", params={"expected_revision": 1}).status_code
        == 204
    )
    last = api.client.get("/collections", params={"limit": 100, "cursor": cursor}).json()
    assert [row["collection_id"] for row in last["collections"]] == [saved[-1]["collection_id"]]
    assert last["next_cursor"] is None
    assert "document_ids" not in first.json()["collections"][0]
    assert "SELECTED_MARKER" not in first.text


def test_unknown_and_malformed_metadata_ids(collection_api: CollectionAPI) -> None:
    api = collection_api
    for identifier, status in ((UNKNOWN_COLLECTION, 404), ("bad", 422), ("col_" + "A" * 32, 422)):
        path = f"/collections/{identifier}"
        responses = (
            api.client.get(path),
            api.client.put(
                path,
                json={
                    "name": "Changed",
                    "document_ids": [api.first],
                    "expected_revision": 1,
                },
            ),
            api.client.delete(path, params={"expected_revision": 1}),
        )
        assert all(response.status_code == status for response in responses)


def test_name_and_membership_boundaries_reuse_existing_scope_types(
    collection_api: CollectionAPI,
) -> None:
    api = collection_api
    identifiers = [f"paper-{index}" for index in range(98)]
    identifiers.extend(["\u7814\u7a76-\u03b2", "https://doi.org/10.1/'quoted'"])
    api.container.ingestion_pipeline.ingest_documents(
        [
            Document(
                document_id=identifier, title="Synthetic", text="GraphRAG note.", source="test"
            )
            for identifier in identifiers
        ]
    )
    saved = api.create(" \u7814" + "n" * 119 + " ", identifiers)
    assert len(saved["name"]) == 120 and saved["document_ids"] == identifiers
    payload = {"query": "GraphRAG", "collection_id": saved["collection_id"]}
    preview = api.client.post("/retrieve", json=payload)
    assert preview.status_code == 200
    assert len(preview.json()["sources"]) == 50
    assert preview.json()["configuration"]["max_source_docs"] == 50
    result = api.client.post("/query", json=payload).json()["result"]
    assert result["state"] == "DONE"
    evidence = api.client.get(f"/runs/{result['run_id']}/export").json()
    assert len(evidence["snapshot"]["sources"]) == 50
    assert evidence["plan"]["observation"]["document_ids"] == identifiers
    lower = api.create("case")
    upper = api.create("Case")
    assert lower["collection_id"] != upper["collection_id"]
    ids = [api.first]
    api.create("Detached input", ids)
    assert ids == [api.first]


async def test_concurrent_api_updates_have_exactly_one_winner(
    collection_api: CollectionAPI,
) -> None:
    api = collection_api
    saved = api.create()
    transport = httpx.ASGITransport(app=api.application)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        results = await asyncio.gather(
            *[
                client.put(
                    f"/collections/{saved['collection_id']}",
                    json={
                        "name": name,
                        "document_ids": [document],
                        "expected_revision": 1,
                    },
                )
                for name, document in (("Writer A", api.first), ("Writer B", api.second))
            ]
        )
    assert sorted(response.status_code for response in results) == [200, 409]
    winner = next(response.json() for response in results if response.status_code == 200)
    assert winner["revision"] == 2
    assert api.client.get(f"/collections/{saved['collection_id']}").json() == winner


async def test_concurrent_same_name_creates_have_exactly_one_winner(
    collection_api: CollectionAPI,
) -> None:
    api = collection_api
    transport = httpx.ASGITransport(app=api.application)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        responses = await asyncio.gather(
            client.post("/collections", json={"name": "Shared", "document_ids": [api.first]}),
            client.post("/collections", json={"name": " Shared ", "document_ids": [api.second]}),
        )
    assert sorted(response.status_code for response in responses) == [201, 409]
    assert len(api.client.get("/collections").json()["collections"]) == 1
    loser = next(response for response in responses if response.status_code == 409)
    assert loser.json()["detail"]["code"] == "collection_name_conflict"


def test_all_crud_and_preview_do_no_generation_or_event_writes(
    collection_api: CollectionAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = collection_api
    for target, method in (
        (api.container.llm, "generate"),
        (api.container.event_log, "append_event"),
        (api.container.event_log, "append_transition"),
        (api.container.runner._executor, "answer"),
    ):
        monkeypatch.setattr(target, method, denied)
    saved = api.create()
    path = f"/collections/{saved['collection_id']}"
    assert api.client.get("/collections").status_code == 200
    assert api.client.get(path).status_code == 200
    assert (
        api.client.put(
            path, json={"name": "Updated", "document_ids": [api.second], "expected_revision": 1}
        ).status_code
        == 200
    )
    preview = api.client.post(
        "/retrieve", json={"query": "GraphRAG", "collection_id": saved["collection_id"]}
    )
    assert preview.status_code == 200
    assert preview.json()["plan"]["observation"]["document_ids"] == [api.second]
    assert api.client.delete(path, params={"expected_revision": 2}).status_code == 204
    assert api.container.event_log.list_events() == []


@pytest.mark.parametrize(
    "raw",
    [
        "null",
        "[]",
        '["missing", "missing"]',
        '[" leading"]',
        '[""]',
        '{"document_ids":["missing"]}',
        '["' + "x" * 129 + '"]',
        "not-json",
        '["missing"]' + " " * 160000,
    ],
)
def test_corrupt_membership_fails_closed_and_can_be_replaced(
    collection_api: CollectionAPI, monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    api = collection_api
    saved = api.create()
    path = f"/collections/{saved['collection_id']}"
    with sqlite3.connect(api.path) as connection:
        connection.execute(
            "UPDATE paper_collections SET document_ids = ? WHERE collection_id = ?",
            (raw, saved["collection_id"]),
        )
    monkeypatch.setattr(api.container.runner, "run", denied)
    monkeypatch.setattr(api.container.runner, "preview", denied)
    responses = [api.client.get(path), api.client.get("/collections")]
    responses.extend(
        api.client.post(
            endpoint, json={"query": "GraphRAG", "collection_id": saved["collection_id"]}
        )
        for endpoint in ("/query", "/retrieve")
    )
    for response in responses:
        assert response.status_code == 409
        assert response.json()["detail"]["code"] == "invalid_collection_record"
        assert_private(response)
    assert (
        api.client.put(
            path, json={"name": "Repaired", "document_ids": [api.second], "expected_revision": 1}
        ).status_code
        == 200
    )
    assert api.client.get(path).json()["document_ids"] == [api.second]
    assert api.container.event_log.list_events() == []


def test_sqlite_failures_are_sanitized_and_transactions_roll_back(
    collection_api: CollectionAPI, caplog: pytest.LogCaptureFixture
) -> None:
    api = collection_api
    saved = api.create()
    with sqlite3.connect(api.path) as connection:
        connection.executescript("""
            CREATE TRIGGER abort_collection_create BEFORE INSERT ON paper_collections
            BEGIN SELECT RAISE(ABORT, 'private-sqlite-diagnostic'); END;
            CREATE TRIGGER abort_collection_update BEFORE UPDATE ON paper_collections
            BEGIN SELECT RAISE(ABORT, 'private-sqlite-diagnostic'); END;
            CREATE TRIGGER abort_collection_delete BEFORE DELETE ON paper_collections
            BEGIN SELECT RAISE(ABORT, 'private-sqlite-diagnostic'); END;
        """)
    path = f"/collections/{saved['collection_id']}"
    responses = [
        api.client.post("/collections", json={"name": "New", "document_ids": [api.second]}),
        api.client.put(
            path,
            json={
                "name": "Changed",
                "document_ids": [api.second],
                "expected_revision": 1,
            },
        ),
        api.client.delete(path, params={"expected_revision": 1}),
    ]
    for response in responses:
        assert response.status_code == 503
        assert response.json()["detail"]["code"] == "collection_storage_error"
        assert "private-sqlite-diagnostic" not in response.text
        assert_private(response)
    assert api.client.get(path).json() == saved
    assert len(api.client.get("/collections").json()["collections"]) == 1
    assert "collection_storage_error" in caplog.text
    assert "private-sqlite-diagnostic" not in caplog.text
    assert api.container.event_log.list_events() == []


def test_unavailable_storage_never_starts_scoped_work(
    collection_api: CollectionAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = collection_api
    saved = api.create()
    with sqlite3.connect(api.path) as connection:
        connection.execute("DROP TABLE paper_collections")
    monkeypatch.setattr(api.container.runner, "run", denied)
    monkeypatch.setattr(api.container.runner, "preview", denied)
    for endpoint in ("/query", "/retrieve"):
        response = api.client.post(
            endpoint, json={"query": "GraphRAG", "collection_id": saved["collection_id"]}
        )
        assert response.status_code == 503
        assert response.json()["detail"]["code"] == "collection_storage_error"
        assert "no such table" not in response.text
    assert api.client.get("/collections").status_code == 503
    assert api.client.get(f"/collections/{saved['collection_id']}").status_code == 503
    assert api.container.event_log.list_events() == []


def test_openapi_documents_shared_scope_and_revision_contract(
    collection_api: CollectionAPI,
) -> None:
    schemas = collection_api.client.get("/openapi.json").json()["components"]["schemas"]
    for model in ("QueryRequest", "RetrievalRequest"):
        field = schemas[model]["properties"]["collection_id"]
        assert field["type"] == "string"
        assert field["pattern"] == "^col_[0-9a-f]{32}$"
        assert "collection_id" not in schemas[model]["required"]
    for model in ("CollectionSelection", "CollectionReplacement"):
        assert schemas[model]["additionalProperties"] is False
        assert schemas[model]["properties"]["document_ids"]["maxItems"] == 100
    assert "expected_revision" in schemas["CollectionReplacement"]["required"]
