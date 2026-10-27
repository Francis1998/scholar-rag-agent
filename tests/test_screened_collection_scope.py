"""Screened API scopes fail before work and retain exact membership through awaits and export."""

import asyncio
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import FastAPI
from pydantic import ValidationError
from scripts.demo_evidence_export import offline_settings

from api.dependencies import AppContainer
from api.schemas import QueryRequest
from retrieval.models import Chunk, Document, Entity, EntityEdge
from storage.paper_screening import ScreeningSubmission
from tests.test_paper_collections_api import CollectionAPI, assert_private, denied
from tests.test_paper_collections_api import collection_api as collection_api

UNKNOWN = "col_" + "0" * 32


def review(
    api: CollectionAPI,
    collection_id: str,
    document_id: str,
    decision: str = "include",
    *,
    revision: int = 0,
    collection_revision: int = 1,
) -> None:
    api.container.paper_screening.submit(
        collection_id,
        document_id,
        ScreeningSubmission.model_validate(
            {
                "collection_revision": collection_revision,
                "expected_decision_revision": revision,
                "decision": decision,
                "reason": "Human opinion on synthetic evidence; not scientific validation.",
            }
        ),
    )


def selection(api: CollectionAPI) -> dict[str, object]:
    saved = api.create(ids=[api.first, api.second])
    collection_id = saved["collection_id"]
    review(api, collection_id, api.first)
    review(api, collection_id, api.second, "exclude")
    return {"collection_id": collection_id, "collection_revision": 1}


@contextmanager
def no_work(api: CollectionAPI, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    before = api.path.read_bytes()
    with monkeypatch.context() as patch:
        for target, method in (
            (api.container.runner, "run"),
            (api.container.runner, "preview"),
            (api.container.runner._analyzer, "analyze"),
            (api.container.runner._planner, "plan"),
            (api.container.hybrid_retriever, "retrieve"),
            (api.container.runner._executor._multihop_retriever, "retrieve"),
            (api.container.llm, "generate"),
            (api.container.event_log, "append_event"),
            (api.container.event_log, "append_transition"),
        ):
            patch.setattr(target, method, denied)
        yield
    assert api.path.read_bytes() == before


def test_nested_scope_is_validated_and_immutable() -> None:
    supplied = {"collection_id": UNKNOWN, "collection_revision": 1}
    request = QueryRequest.model_validate({"query": "GraphRAG", "screened_collection": supplied})
    assert request.screened_collection is not None
    assert request.screened_collection.collection_revision == 1
    supplied["collection_revision"] = 2
    assert request.screened_collection.collection_revision == 1
    with pytest.raises(ValidationError):
        request.screened_collection.collection_revision = 2
    with pytest.raises(ValidationError):
        request.screened_collection = None


@pytest.mark.parametrize("endpoint", ["/query", "/retrieve"])
@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        {},
        "",
        True,
        42,
        UNKNOWN,
        {"collection_id": UNKNOWN},
        {"collection_revision": 1},
        {"collection_id": None, "collection_revision": 1},
        {"collection_id": 1, "collection_revision": 1},
        {"collection_id": "not-a-collection", "collection_revision": 1},
        {"collection_id": "\ud800", "collection_revision": 1},
        *[
            {"collection_id": UNKNOWN, "collection_revision": revision}
            for revision in (
                None,
                True,
                False,
                0,
                -1,
                "1",
                1.0,
                2**63,
                [],
                {},
                float("nan"),
                float("inf"),
            )
        ],
        {"collection_id": UNKNOWN, "collection_revision": 1, "decision": "exclude"},
    ],
)
def test_invalid_screened_scope_never_starts_work(
    collection_api: CollectionAPI, monkeypatch: pytest.MonkeyPatch, endpoint: str, value: object
) -> None:
    api = collection_api
    with no_work(api, monkeypatch):
        response = api.client.post(
            endpoint,
            content=json.dumps({"query": "GraphRAG", "screened_collection": value}),
            headers={"content-type": "application/json"},
        )
    assert response.status_code == 422, response.text


@pytest.mark.parametrize("endpoint", ["/query", "/retrieve"])
@pytest.mark.parametrize(
    ("root_path", "mounts"),
    [("", ()), ("/proxy", ()), ("/proxy", ("/outer", "/library"))],
)
@pytest.mark.parametrize(
    "value",
    [
        {"collection_id": UNKNOWN, "collection_revision": float("nan")},
        {"collection_id": "\ud800", "collection_revision": 1},
    ],
)
async def test_invalid_screened_scope_is_safe_under_mounts(
    collection_api: CollectionAPI,
    monkeypatch: pytest.MonkeyPatch,
    endpoint: str,
    root_path: str,
    mounts: tuple[str, ...],
    value: object,
) -> None:
    api = collection_api
    application = api.application
    for mount_path in reversed(mounts):
        parent = FastAPI()
        parent.mount(mount_path, application, name="child")
        application = parent
    prefix = root_path + "".join(mounts)
    transport = httpx.ASGITransport(
        app=application, root_path=root_path, raise_app_exceptions=False
    )
    with no_work(api, monkeypatch):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                prefix + endpoint,
                content=json.dumps({"query": "GraphRAG", "screened_collection": value}),
                headers={"content-type": "application/json"},
            )
    assert response.status_code == 422, response.text
    assert_private(response)
    assert response.json()["detail"]
    assert all("input" not in error for error in response.json()["detail"])


@pytest.mark.parametrize("endpoint", ["/query", "/retrieve"])
@pytest.mark.parametrize(
    "extra",
    [
        {"document_ids": ["selected"]},
        {"document_ids": []},
        {"document_ids": None},
        {"collection_id": UNKNOWN},
        {"collection_id": None},
        {"collection_id": UNKNOWN, "document_ids": ["selected"]},
    ],
)
def test_scope_variants_are_mutually_exclusive(
    collection_api: CollectionAPI,
    monkeypatch: pytest.MonkeyPatch,
    endpoint: str,
    extra: dict[str, object],
) -> None:
    api = collection_api
    scope = selection(api)
    with no_work(api, monkeypatch):
        response = api.client.post(
            endpoint, json={"query": "GraphRAG", "screened_collection": scope, **extra}
        )
    assert response.status_code == 422, response.text


@pytest.mark.parametrize("endpoint", ["/query", "/retrieve"])
@pytest.mark.parametrize(
    ("fault", "status", "code"),
    [
        ("unknown", 404, "collection_not_found"),
        ("stale", 409, "collection_revision_conflict"),
        ("empty", 409, "screened_collection_empty"),
        ("no_current_includes", 409, "screened_collection_empty"),
        ("missing_include", 409, "collection_documents_missing"),
        ("missing_exclude", 409, "collection_documents_missing"),
        ("broken_collection", 409, "invalid_collection_record"),
        ("broken_review", 409, "invalid_screening_record"),
        ("storage", 503, "collection_storage_error"),
    ],
)
def test_rejected_selection_has_private_errors_and_no_work(
    collection_api: CollectionAPI,
    monkeypatch: pytest.MonkeyPatch,
    endpoint: str,
    fault: str,
    status: int,
    code: str,
) -> None:
    api = collection_api
    scope = selection(api)
    collection_id = str(scope["collection_id"])
    if fault == "unknown":
        scope["collection_id"] = UNKNOWN
    elif fault in ("stale", "no_current_includes"):
        api.container.paper_collections.replace(
            collection_id,
            name="New criteria",
            document_ids=[api.first, api.second],
            expected_revision=1,
        )
        if fault == "no_current_includes":
            scope["collection_revision"] = 2
    elif fault == "empty":
        review(api, collection_id, api.first, "unsure", revision=1)
    else:
        with sqlite3.connect(api.path) as connection:
            if fault.startswith("missing_"):
                identifier = api.first if fault == "missing_include" else api.second
                connection.execute("DELETE FROM documents WHERE document_id = ?", (identifier,))
            elif fault == "broken_collection":
                connection.execute("UPDATE paper_collections SET document_ids = '[]'")
            elif fault == "broken_review":
                connection.execute(
                    "UPDATE paper_screening_decisions SET record = ? WHERE document_id = ?",
                    ("private-malformed-review", api.second),
                )
            elif fault == "storage":
                connection.execute("DROP TABLE paper_screening_decisions")
    with no_work(api, monkeypatch):
        response = api.client.post(
            endpoint, json={"query": "GraphRAG", "screened_collection": scope}
        )
    assert response.status_code == status, response.text
    assert response.json()["detail"]["code"] == code
    assert_private(response)
    assert "private-malformed-review" not in response.text
    assert "paper_screening_decisions" not in response.text
    assert api.container.event_log.list_events() == []


def test_preview_query_and_export_preserve_the_entire_resolved_selection(
    collection_api: CollectionAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = collection_api
    api.container.document_store.add_documents(
        [
            Document(
                document_id="catalog-only",
                title="Synthetic unchunked note",
                text="Stored but not indexed.",
                source="synthetic",
            )
        ],
        [],
    )
    saved = api.create(ids=["catalog-only", api.first, api.second])
    for identifier in ("catalog-only", api.first):
        review(api, saved["collection_id"], identifier)
    review(api, saved["collection_id"], api.second, "exclude")
    body = {
        "query": "Compare GraphRAG versus Retrieval",
        "screened_collection": {"collection_id": saved["collection_id"], "collection_revision": 1},
        "max_chunks_per_document": 1,
        "min_evidence_documents": 1,
    }
    before = api.path.read_bytes()
    with monkeypatch.context() as patch:
        for target, method in (
            (api.container.llm, "generate"),
            (api.container.event_log, "append_event"),
            (api.container.event_log, "append_transition"),
        ):
            patch.setattr(target, method, denied)
        preview = api.client.post("/retrieve", json=body)
    assert preview.status_code == 200, preview.text
    assert_private(preview)
    assert api.path.read_bytes() == before
    expected = ["catalog-only", api.first]
    assert preview.json()["plan"]["observation"]["document_ids"] == expected
    assert {s["chunk"]["document_id"] for s in preview.json()["sources"]} == {api.first}
    assert "EXCLUDED_MARKER" not in preview.text
    result = api.client.post("/query", json=body)
    assert result.status_code == 200, result.text
    assert_private(result)
    run = result.json()["result"]
    assert run["state"] == "DONE", run
    assert run["plan"]["observation"]["document_ids"] == expected
    path = f"/runs/{run['run_id']}/export"
    exported = api.client.get(path)
    assert exported.status_code == 200, exported.text
    bundle = exported.json()
    assert bundle["plan"]["observation"]["document_ids"] == expected
    assert bundle["events"][0]["payload"]["payload"]["document_ids"] == expected
    assert bundle["snapshot"]["sources"] == preview.json()["sources"]
    assert bundle["snapshot"]["request"]["context"] == preview.json()["context"]
    assert bundle["generation"]["provider"] == "fake"
    review(api, saved["collection_id"], api.first, "exclude", revision=1)
    api.container.paper_collections.delete(saved["collection_id"], expected_revision=1)
    api.application.state.container = AppContainer(offline_settings(api.path))
    assert api.client.get(path).content == exported.content


@pytest.mark.parametrize("endpoint", ["/query", "/retrieve"])
async def test_resolution_is_once_before_await_and_isolated_from_new_decisions(
    collection_api: CollectionAPI, monkeypatch: pytest.MonkeyPatch, endpoint: str
) -> None:
    api = collection_api
    scope = selection(api)
    collection_id = str(scope["collection_id"])
    selected = Mock(wraps=api.container.paper_screening.select_included)
    monkeypatch.setattr(api.container.paper_screening, "select_included", selected)
    entered, release = asyncio.Event(), asyncio.Event()
    hyde = api.container.hybrid_retriever._hyde_expander
    original = hyde.expand

    async def held(query: str) -> str:
        if query == "first GraphRAG":
            entered.set()
            await release.wait()
        return await original(query)

    monkeypatch.setattr(hyde, "expand", held)
    transport = httpx.ASGITransport(app=api.application)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        pending = asyncio.create_task(
            client.post(endpoint, json={"query": "first GraphRAG", "screened_collection": scope})
        )
        try:
            await asyncio.wait_for(entered.wait(), timeout=3)
            selected.assert_called_once_with(collection_id, collection_revision=1)
            review(api, collection_id, api.first, "exclude", revision=1)
            review(api, collection_id, api.second, revision=1)
            later, ordinary, unscoped = await asyncio.wait_for(
                asyncio.gather(
                    client.post(endpoint, json={"query": "GraphRAG", "screened_collection": scope}),
                    client.post(
                        endpoint, json={"query": "GraphRAG", "collection_id": collection_id}
                    ),
                    client.post(endpoint, json={"query": "GraphRAG"}),
                ),
                timeout=5,
            )
            assert not pending.done()
            assert selected.call_count == 2
        finally:
            release.set()
        first = await asyncio.wait_for(pending, timeout=3)
    for response, expected in zip(
        (first, later, ordinary, unscoped),
        ([api.first], [api.second], [api.first, api.second], None),
        strict=True,
    ):
        assert response.status_code == 200, response.text
        result = response.json()["result"] if endpoint == "/query" else response.json()
        assert result["plan"]["observation"]["document_ids"] == expected
        if endpoint == "/query":
            assert result["state"] == "DONE", result
            exported = api.client.get(f"/runs/{result['run_id']}/export").json()
            assert exported["plan"]["observation"]["document_ids"] == expected
            sources = exported["snapshot"]["sources"]
        else:
            sources = result["sources"]
        assert {s["chunk"]["document_id"] for s in sources} == set(
            expected if expected is not None else [api.first, api.second]
        )


@pytest.mark.parametrize("endpoint", ["/query", "/retrieve"])
def test_screened_graph_cannot_cross_an_excluded_bridge(
    collection_api: CollectionAPI, monkeypatch: pytest.MonkeyPatch, endpoint: str
) -> None:
    api = collection_api
    scope = selection(api)
    graph = api.container.graph_store
    for chunk_id, document_id, text, entities in (
        ("start", api.first, "Alpha evidence", ["Alpha"]),
        ("bridge", api.second, "Alpha Beta EXCLUDED_MARKER", ["Alpha", "Beta"]),
        ("tail", api.first, "Beta Gamma otherwise allowed evidence", ["Beta", "Gamma"]),
    ):
        graph.add_mentions(
            Chunk(
                chunk_id=chunk_id,
                document_id=document_id,
                title="Synthetic graph",
                text=text,
                source="synthetic:screened-scope",
            ),
            [Entity(name=name) for name in entities],
        )
    graph.add_edges(
        [
            EntityEdge(source="Alpha", target="Beta", chunk_id="bridge"),
            EntityEdge(source="Beta", target="Gamma", chunk_id="tail"),
        ]
    )
    # Isolate real graph traversal from independent hybrid discovery of the tail.
    monkeypatch.setattr(api.container.hybrid_retriever, "retrieve", AsyncMock(return_value=[]))
    baseline = api.client.post(
        "/retrieve", json={"query": "summarize Alpha", "collection_id": scope["collection_id"]}
    )
    assert baseline.status_code == 200, baseline.text
    assert {s["chunk"]["chunk_id"] for s in baseline.json()["sources"]} == {
        "start",
        "bridge",
        "tail",
    }
    response = api.client.post(
        endpoint, json={"query": "summarize Alpha", "screened_collection": scope}
    )
    assert response.status_code == 200, response.text
    if endpoint == "/query":
        run = response.json()["result"]
        assert run["state"] == "DONE", run
        sources = api.client.get(f"/runs/{run['run_id']}/export").json()["snapshot"]["sources"]
    else:
        sources = response.json()["sources"]
    assert [s["chunk"]["chunk_id"] for s in sources] == ["start"]
    assert sources[0]["path"] == ["alpha", "multihop"]


@pytest.mark.parametrize("endpoint", ["/query", "/retrieve"])
def test_ordinary_scopes_ignore_screening_even_when_its_records_are_broken(
    collection_api: CollectionAPI, monkeypatch: pytest.MonkeyPatch, endpoint: str
) -> None:
    api = collection_api
    scope = selection(api)
    with sqlite3.connect(api.path) as connection:
        connection.execute("UPDATE paper_screening_decisions SET record = 'invalid'")
    monkeypatch.setattr(api.container.paper_screening, "_snapshot", denied)
    for fields, expected in (
        ({}, None),
        ({"collection_id": scope["collection_id"]}, [api.first, api.second]),
        ({"document_ids": [api.first]}, [api.first]),
    ):
        response = api.client.post(endpoint, json={"query": "GraphRAG", **fields})
        assert response.status_code == 200, response.text
        result = response.json()["result"] if endpoint == "/query" else response.json()
        assert result["plan"]["observation"]["document_ids"] == expected


@pytest.mark.parametrize(
    ("endpoint", "body"),
    [
        ("/research/worksheet", {"questions": ["GraphRAG"], "document_ids": ["selected"]}),
        ("/research/compare-retrieval", {"query": "GraphRAG", "baseline": {}, "candidate": {}}),
    ],
)
def test_other_request_contracts_reject_screened_scope_instead_of_ignoring_it(
    collection_api: CollectionAPI,
    monkeypatch: pytest.MonkeyPatch,
    endpoint: str,
    body: dict[str, object],
) -> None:
    api = collection_api
    with no_work(api, monkeypatch):
        response = api.client.post(
            endpoint,
            json={
                **body,
                "screened_collection": {"collection_id": UNKNOWN, "collection_revision": 1},
            },
        )
    assert response.status_code == 422, response.text
    assert any(error["type"] == "extra_forbidden" for error in response.json()["detail"])


def test_openapi_documents_opt_in_nonnull_nested_scope(collection_api: CollectionAPI) -> None:
    schemas = collection_api.client.get("/openapi.json").json()["components"]["schemas"]
    for name in ("QueryRequest", "RetrievalRequest"):
        scope = schemas[name]["properties"]["screened_collection"]
        assert "anyOf" not in scope
        assert "screened_collection" not in schemas[name]["required"]
        nested = schemas[scope["$ref"].rsplit("/", 1)[1]]
        assert set(nested["required"]) == {"collection_id", "collection_revision"}
        assert nested["additionalProperties"] is False
        assert nested["properties"]["collection_revision"]["minimum"] == 1
