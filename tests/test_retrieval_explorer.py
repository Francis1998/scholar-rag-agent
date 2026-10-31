"""Native retrieval inspection must reuse preparation, never generation or persistence."""

import json
import socket
from collections.abc import Iterator
from unittest.mock import AsyncMock, Mock
from urllib.parse import urlsplit

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from agent.retrieval_preview import RetrievalPreviewError
from api.retrieval_explorer import explore_retrieval
from storage.paper_collections import CollectionError
from tests.test_corpus_explorer import (
    HOSTILE,
    ExplorerDOM,
    assert_headers,
    href,
    identities,
    values,
)
from tests.test_retrieval_preview import PreviewAPI, denied
from tests.test_retrieval_preview import preview_api as preview_api

ENDPOINT = "/explore/retrieve"
ORIGIN = {"Origin": "http://testserver"}
FORM = {
    "query": "GraphRAG retrieval",
    "scope": "documents",
    "document_ids": '["a", "b"]',
    "collection_id": "",
    "max_chunks_per_document": "1",
    "min_evidence_documents": "2",
    "near_duplicate_threshold": "",
}


@pytest.fixture
def workbench(preview_api: PreviewAPI, monkeypatch: pytest.MonkeyPatch) -> Iterator[PreviewAPI]:
    api = preview_api
    api.add("a", "GraphRAG retrieval uses local terms.")
    api.add("b", "GraphRAG retrieval follows graph connections.")
    api.add("excluded", "EXCLUDED_MARKER retrieval.")
    guards = []
    for target, method in (
        (api.container.runner, "run"),
        (api.container.runner._executor, "answer"),
        (api.container.llm, "generate"),
        (api.container.event_log, "append_event"),
        (api.container.event_log, "append_transition"),
        (api.container.ingestion_pipeline, "ingest_documents"),
        (socket, "create_connection"),
        (socket.socket, "connect"),
    ):
        guard = Mock(side_effect=denied)
        monkeypatch.setattr(target, method, guard)
        guards.append(guard)
    before = api.database_path.read_bytes()
    yield api
    assert api.database_path.read_bytes() == before
    assert api.container.event_log.list_events() == []
    assert all(guard.call_count == 0 for guard in guards)


def controls(html: str) -> dict[str, str]:
    root = ExplorerDOM(html).root
    form = root.by_id("retrieval-form")
    fields = {node.attrs["name"]: node.attrs.get("value", "") for node in form.by_tag("input")}
    fields.update(
        (node.attrs["name"], node.text().removeprefix("\n")) for node in form.by_tag("textarea")
    )
    fields["scope"] = next(
        node.attrs["value"] for node in form.by_tag("option") if "selected" in node.attrs
    )
    return fields


def failure(response: httpx.Response, status: int) -> None:
    assert response.status_code == status, response.text
    assert_headers(response, referrer_policy="same-origin")
    root = ExplorerDOM(response.text).root
    assert not root.by_class("prepared-source")
    assert not root.by_tag("form")
    assert "No partial preview" in root.text()
    assert "PRIVATE_" not in response.text and "EXCLUDED_MARKER" not in response.text


def test_opening_only_shows_controls_and_integrated_navigation(
    workbench: PreviewAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    preview = AsyncMock(side_effect=denied)
    monkeypatch.setattr(workbench.container.runner, "preview", preview)
    catalog = ExplorerDOM(workbench.client.get("/explore?limit=1").text).root
    response = workbench.client.get(href(catalog, "retrieval-preview"))
    assert response.status_code == 200
    assert_headers(response, referrer_policy="same-origin")
    root = ExplorerDOM(response.text).root
    assert root.by_id("retrieval-form").attrs["method"] == "post"
    assert href(root, "back-to-catalog") == "/explore?limit=1"
    assert controls(response.text)["scope"] == "all"
    document = ExplorerDOM(workbench.client.get(href(catalog, "inspect-document-1")).text).root
    selected = workbench.client.get(href(document, "preview-document"))
    assert json.loads(controls(selected.text)["document_ids"]) == ["a"]
    assert controls(selected.text)["scope"] == "documents"
    for node in (*root.by_tag("textarea"), *root.by_tag("select"), *root.by_tag("input")):
        if node.attrs.get("type") != "hidden":
            assert node.attrs["id"] in {label.attrs["for"] for label in root.by_tag("label")}
    preview.assert_not_called()


@pytest.mark.parametrize("identifier", ["nul\0id", "line\r\nid", 'paper"</textarea>/\u7814'])
def test_prefill_preserves_control_character_ids_as_json_without_computing(
    workbench: PreviewAPI, monkeypatch: pytest.MonkeyPatch, identifier: str
) -> None:
    preview = AsyncMock(side_effect=denied)
    monkeypatch.setattr(workbench.container.runner, "preview", preview)
    response = workbench.client.get(ENDPOINT, params={"document_id": identifier})
    assert response.status_code == 200
    assert json.loads(controls(response.text)["document_ids"]) == [identifier]
    assert "\0" not in response.text
    preview.assert_not_called()


def test_post_displays_exact_api_preparation_and_failed_minimum_without_discarding(
    workbench: PreviewAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = workbench.preview(
        FORM["query"], document_ids=["a", "b"], max_chunks_per_document=1, min_evidence_documents=3
    )
    preview = AsyncMock(wraps=workbench.container.runner.preview)
    monkeypatch.setattr(workbench.container.runner, "preview", preview)
    response = workbench.client.post(
        ENDPOINT, data={**FORM, "min_evidence_documents": "3"}, headers=ORIGIN
    )
    assert response.status_code == 200, response.text
    assert_headers(response, referrer_policy="same-origin")
    root = ExplorerDOM(response.text).root
    assert values(root, "prepared-context") == [expected["context"]]
    assert values(root, "context-sha256") == [expected["context_sha256"]]
    assert identities(root, "source-document-id") == [
        source["chunk"]["document_id"] for source in expected["sources"]
    ]
    assert values(root, "passage-text") == [
        source["chunk"]["text"] for source in expected["sources"]
    ]
    assert json.loads(values(root, "preview-plan")[0]) == expected["plan"]
    assert json.loads(values(root, "preview-configuration")[0]) == expected["configuration"]
    assert json.loads(values(root, "evidence-assessment")[0]) == expected["evidence_assessment"]
    assert "Minimum not met" in root.text()
    assert "EXCLUDED_MARKER" not in root.text()
    preview.assert_awaited_once_with(
        FORM["query"],
        document_ids=("a", "b"),
        max_chunks_per_document=1,
        min_evidence_documents=3,
    )


@pytest.mark.parametrize(
    "updates",
    [
        {"query": ""},
        {"query": "\u2003"},
        {"query": "x" * 2001},
        {"query": "\0PRIVATE_QUERY"},
        {"scope": ""},
        {"scope": "all"},
        {"scope": "collection"},
        {"scope": "screened"},
        {"document_ids": ""},
        {"document_ids": "[]"},
        {"document_ids": "null"},
        {"document_ids": "{}"},
        {"document_ids": "[true]"},
        {"document_ids": '["\\ud800"]'},
        {"document_ids": '["x"'},
        {"document_ids": "[" * 1100},
        {"document_ids": json.dumps(["a"] * 101)},
        {"document_ids": json.dumps(["x" * 129])},
        {"collection_id": "col_" + "0" * 32},
        {"max_chunks_per_document": "0"},
        {"max_chunks_per_document": "51"},
        {"max_chunks_per_document": "1.0"},
        {"max_chunks_per_document": "true"},
        {"min_evidence_documents": "+1"},
        {"min_evidence_documents": " 1"},
        {"near_duplicate_threshold": "NaN"},
        {"near_duplicate_threshold": "Infinity"},
        {"near_duplicate_threshold": "0"},
        {"near_duplicate_threshold": "1.1"},
        {"near_duplicate_threshold": "true"},
        {"near_duplicate_threshold": '"0.5"'},
        {"model": "PRIVATE_MODEL"},
        {"navigation": '{"return_url":"https://invalid.example/PRIVATE_REDIRECT"}'},
        {"navigation": '{"source":"\\ud800"}'},
        {"navigation": '{"catalog_limit":1,"catalog_limit":2}'},
    ],
)
def test_invalid_fields_never_start_preview(
    workbench: PreviewAPI, monkeypatch: pytest.MonkeyPatch, updates: dict[str, str]
) -> None:
    preview = AsyncMock(side_effect=denied)
    monkeypatch.setattr(workbench.container.runner, "preview", preview)
    failure(workbench.client.post(ENDPOINT, data={**FORM, **updates}, headers=ORIGIN), 422)
    preview.assert_not_called()


@pytest.mark.parametrize(
    "wire",
    [
        b"query=x&query=y",
        b"query=x&%71uery=y",
        b"query=%FF",
        b"query=%C3",
        b"query=%",
        b"query=%GG",
        b"query=\xff",
        b"query",
        b"query=x&&scope=all",
    ],
)
def test_malformed_wire_fails_closed(
    workbench: PreviewAPI, monkeypatch: pytest.MonkeyPatch, wire: bytes
) -> None:
    preview = AsyncMock(side_effect=denied)
    monkeypatch.setattr(workbench.container.runner, "preview", preview)
    failure(
        workbench.client.post(
            ENDPOINT,
            content=wire,
            headers={**ORIGIN, "Content-Type": "application/x-www-form-urlencoded"},
        ),
        422,
    )
    failure(workbench.client.get(ENDPOINT + "?" + wire.decode("latin1")), 422)
    preview.assert_not_called()


@pytest.mark.parametrize(
    "headers,status",
    [
        ({}, 403),
        ({"Origin": "null"}, 403),
        ({"Origin": "https://evil.invalid"}, 403),
        ({**ORIGIN, "Sec-Fetch-Site": "cross-site"}, 403),
        ({**ORIGIN, "Content-Type": "application/json"}, 415),
        ({**ORIGIN, "Content-Encoding": "gzip"}, 415),
        ({**ORIGIN, "Content-Length": "65537"}, 413),
        ({**ORIGIN, "Content-Length": "1"}, 422),
    ],
)
def test_origin_and_transport_guards(
    workbench: PreviewAPI, monkeypatch: pytest.MonkeyPatch, headers: dict[str, str], status: int
) -> None:
    preview = AsyncMock(side_effect=denied)
    monkeypatch.setattr(workbench.container.runner, "preview", preview)
    failure(workbench.client.post(ENDPOINT, data=FORM, headers=headers), status)
    preview.assert_not_called()


def test_size_methods_and_query_controls_do_not_compute(
    workbench: PreviewAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    preview = AsyncMock(side_effect=denied)
    monkeypatch.setattr(workbench.container.runner, "preview", preview)
    failure(
        workbench.client.post(
            ENDPOINT,
            content=iter([b"x" * 32768, b"x" * 32769]),
            headers={**ORIGIN, "Content-Type": "application/x-www-form-urlencoded"},
        ),
        413,
    )
    failure(workbench.client.post(ENDPOINT + "?query=x", data=FORM, headers=ORIGIN), 422)
    failure(workbench.client.request("GET", ENDPOINT, content=b"query=x"), 422)
    failure(workbench.client.get(ENDPOINT, params={"query": "PRIVATE_QUERY"}), 422)
    failure(workbench.client.put(ENDPOINT), 405)
    preview.assert_not_called()


async def test_real_query_byte_cap_before_any_request_parsing() -> None:
    response = await explore_retrieval(
        Request({"type": "http", "method": "GET", "query_string": b"x" * 65537})
    )
    assert response.status_code == 422
    assert b"65,536" in response.body


def test_collection_prefill_resolution_and_no_implicit_screening(
    preview_api: PreviewAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = preview_api
    api.add("a", "GraphRAG retrieval methods.")
    api.add("b", "GraphRAG retrieval limitations.")
    collection = api.container.paper_collections.create(name="Synthetic", document_ids=["b", "a"])
    before = api.database_path.read_bytes()
    monkeypatch.setattr(api.container.runner, "run", denied)
    monkeypatch.setattr(api.container.llm, "generate", denied)
    monkeypatch.setattr(api.container.event_log, "append_event", denied)
    monkeypatch.setattr(api.container.paper_screening, "select_included", denied)
    resolve = Mock(wraps=api.container.paper_collections.resolve)
    preview = AsyncMock(wraps=api.container.runner.preview)
    monkeypatch.setattr(api.container.paper_collections, "resolve", resolve)
    monkeypatch.setattr(api.container.runner, "preview", preview)
    collection_page = api.client.get(
        "/explore/collection",
        params={
            "collection_id": collection.collection_id,
            "collection_revision": collection.revision,
        },
    )
    assert collection_page.status_code == 200
    opened = api.client.get(href(ExplorerDOM(collection_page.text).root, "preview-collection"))
    assert opened.status_code == 200
    assert controls(opened.text)["scope"] == "collection"
    preview.assert_not_called()
    response = api.client.post(
        ENDPOINT,
        data={**controls(opened.text), "query": "retrieval", "near_duplicate_threshold": "0.9"},
        headers=ORIGIN,
    )
    assert response.status_code == 200, response.text
    root = ExplorerDOM(response.text).root
    assert set(identities(root, "source-document-id")) == {"a", "b"}
    assert json.loads(values(root, "resolved-scope")[0]) == list(collection.document_ids)
    preview.assert_awaited_once_with(
        "retrieval", document_ids=tuple(collection.document_ids), near_duplicate_threshold=0.9
    )
    resolve.assert_called_once_with(collection.collection_id)
    assert api.database_path.read_bytes() == before
    assert api.container.event_log.list_events() == []


@pytest.mark.parametrize(
    "code,status",
    [
        ("collection_not_found", 404),
        ("invalid_collection_record", 409),
        ("collection_documents_missing", 409),
        ("collection_storage_error", 503),
    ],
)
def test_collection_errors_fail_before_runner(
    workbench: PreviewAPI, monkeypatch: pytest.MonkeyPatch, code: str, status: int
) -> None:
    preview = AsyncMock(side_effect=denied)
    monkeypatch.setattr(workbench.container.runner, "preview", preview)
    monkeypatch.setattr(
        workbench.container.paper_collections,
        "resolve",
        Mock(side_effect=CollectionError(code, "PRIVATE_STORAGE", status)),
    )
    failure(
        workbench.client.post(
            ENDPOINT,
            data={
                **FORM,
                "scope": "collection",
                "document_ids": "",
                "collection_id": "col_" + "0" * 32,
            },
            headers=ORIGIN,
        ),
        status,
    )
    preview.assert_not_called()


@pytest.mark.parametrize(
    "text,status",
    [
        ("GraphRAG " + "<" * 140_000, 413),
        ("GraphRAG \0PRIVATE_TEXT", 409),
        ("GraphRAG " + "x" * 262_144, 500),
    ],
)
def test_complete_html_and_capture_limits_never_return_partial_success(
    preview_api: PreviewAPI, monkeypatch: pytest.MonkeyPatch, text: str, status: int
) -> None:
    api = preview_api
    api.add("a", text)
    before = api.database_path.read_bytes()
    monkeypatch.setattr(api.container.runner, "run", denied)
    monkeypatch.setattr(api.container.llm, "generate", denied)
    monkeypatch.setattr(api.container.event_log, "append_event", denied)
    failure(
        api.client.post(
            ENDPOINT,
            data={**FORM, "document_ids": '["a"]'},
            headers=ORIGIN,
        ),
        status,
    )
    assert api.database_path.read_bytes() == before
    assert api.container.event_log.list_events() == []


def test_policies_and_omissions_reuse_exact_json_contract(workbench: PreviewAPI) -> None:
    submitted = {
        **FORM,
        "max_chunks_per_document": "1",
        "min_evidence_documents": "",
        "near_duplicate_threshold": "1",
    }
    expected = workbench.preview(
        FORM["query"],
        document_ids=["a", "b"],
        max_chunks_per_document=1,
        near_duplicate_threshold=1,
    )
    response = workbench.client.post(ENDPOINT, data=submitted, headers=ORIGIN)
    assert response.status_code == 200
    root = ExplorerDOM(response.text).root
    assert values(root, "prepared-context") == [expected["context"]]
    assert (
        json.loads(values(root, "evidence-policy")[0])
        == expected["plan"]["observation"]["evidence_policy"]
    )
    assert not root.by_class("evidence-assessment")
    assert "No minimum-document requirement" in root.text()
    assert values(root, "source-retriever") == [
        source["retriever"] for source in expected["sources"]
    ]
    assert [json.loads(path) for path in values(root, "retrieval-path")] == [
        source["path"] for source in expected["sources"]
    ]


@pytest.mark.parametrize(
    "code,status",
    [
        ("planning_failed", 500),
        ("retrieval_failed", 500),
        ("context_preparation_failed", 500),
        ("retrieval_timeout", 504),
        ("context_preparation_timeout", 504),
        ("generative_retrieval", 409),
    ],
)
def test_operational_errors_are_not_empty_success(
    workbench: PreviewAPI, monkeypatch: pytest.MonkeyPatch, code: str, status: int
) -> None:
    monkeypatch.setattr(
        workbench.container.runner,
        "preview",
        AsyncMock(side_effect=RetrievalPreviewError(code, "PRIVATE_DIAGNOSTIC", status)),
    )
    failure(workbench.client.post(ENDPOINT, data=FORM, headers=ORIGIN), status)


def test_whole_corpus_and_unknown_ids_have_distinct_explicit_scope(workbench: PreviewAPI) -> None:
    all_response = workbench.client.post(
        ENDPOINT, data={**FORM, "scope": "all", "document_ids": ""}, headers=ORIGIN
    )
    assert all_response.status_code == 200
    all_root = ExplorerDOM(all_response.text).root
    assert json.loads(values(all_root, "resolved-scope")[0]) is None
    unknown = workbench.client.post(
        ENDPOINT, data={**FORM, "document_ids": '["not-ingested"]'}, headers=ORIGIN
    )
    assert unknown.status_code == 200
    root = ExplorerDOM(unknown.text).root
    assert json.loads(values(root, "resolved-scope")[0]) == ["not-ingested"]
    assert not root.by_class("prepared-source")
    assert "No passages returned" in root.text()
    assert "Minimum not met" in root.text()


@pytest.mark.parametrize("prefix,mounted", [("/library", False), ("/prefix/library", True)])
def test_mounts_and_escaped_exact_ids(
    preview_api: PreviewAPI, monkeypatch: pytest.MonkeyPatch, prefix: str, mounted: bool
) -> None:
    api = preview_api
    identifier = "../paper/\u7814?&\r\n"
    identifier = identifier.rstrip() + "/tail"
    api.add(identifier, "\n" + HOSTILE + "\r\nGraphRAG")
    parent = api.application
    if mounted:
        parent = FastAPI()
        parent.mount("/library", api.application)
    monkeypatch.setattr(api.container.runner, "run", denied)
    monkeypatch.setattr(api.container.llm, "generate", denied)
    monkeypatch.setattr(api.container.event_log, "append_event", denied)
    with TestClient(parent, root_path="/prefix" if mounted else prefix) as client:
        opened = client.get(prefix + ENDPOINT, params={"document_id": identifier})
        assert opened.status_code == 200
        fields = controls(opened.text)
        response = client.post(
            ExplorerDOM(opened.text).root.by_id("retrieval-form").attrs["action"],
            data={**fields, "query": HOSTILE + " GraphRAG"},
            headers=ORIGIN,
        )
        assert response.status_code == 200, response.text
        root = ExplorerDOM(response.text).root
        assert identities(root, "source-document-id") == [identifier]
        assert values(root, "passage-text") == ["\n" + HOSTILE + "\r\nGraphRAG"]
        assert not any(root.by_tag(tag) for tag in ("script", "img", "iframe", "base"))
        for node in root.by_tag("a"):
            parsed = urlsplit(node.attrs["href"])
            assert not parsed.netloc and not parsed.scheme
            assert not parsed.path or parsed.path.startswith(prefix + "/")
        context = client.get(href(root, "preview-source-1"))
        # This fixture has no chunk_index; the existing reader must not guess one.
        assert context.status_code == 409
        error = client.post(prefix + ENDPOINT, data={"query": "PRIVATE_QUERY"}, headers=ORIGIN)
        failure(error, 422)
        assert href(ExplorerDOM(error.text).root, "reset-preview") == prefix + ENDPOINT
