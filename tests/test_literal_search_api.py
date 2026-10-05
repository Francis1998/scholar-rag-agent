"""Offline HTTP contracts for literal search, independent of ranked retrieval."""

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import NoReturn

import httpx
import pytest
from fastapi.testclient import TestClient
from scripts.demo_evidence_export import offline_settings

from api.application import create_app
from retrieval.models import Chunk, Document
from storage.literal_search import LiteralSearchPage, LiteralSearchRequest
from tests.test_literal_search_storage import seed


def no_work(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("Literal search must not retrieve, generate, or write.")


def test_literal_search_finds_a_late_phrase_without_agent_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "corpus.sqlite3"
    app = create_app(offline_settings(path))
    container = app.state.container
    text = "Synthetic padding. " * 60 + "exact phrase" + " trailing context."
    container.document_store.add_documents(
        [Document(document_id="paper", title="Synthetic", text="PRIVATE_BODY", source="local")],
        [
            Chunk(
                chunk_id="chunk",
                document_id="paper",
                title="Synthetic",
                source="local",
                text=text,
                metadata={"private": "PRIVATE_METADATA"},
            )
        ],
    )
    before = path.read_bytes()
    for target, method in (
        (container.document_store, "list_chunks"),
        (container.document_store, "add_documents"),
        (container.hybrid_retriever, "retrieve"),
        (container.hybrid_retriever, "add_chunks"),
        (container.llm, "generate"),
        (container.runner, "run"),
        (container.runner, "preview"),
        (container.event_log, "append_event"),
        (container.event_log, "list_events"),
    ):
        monkeypatch.setattr(target, method, no_work)
    with TestClient(app) as client:
        response = client.post("/research/search", json={"query": "exact phrase"})
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    result = response.json()
    assert result["query"] == "exact phrase"
    assert len(result["matches"]) == 1
    match = result["matches"][0]
    assert match["document_id"] == "paper" and match["chunk_id"] == "chunk"
    assert match["match_start"] == text.index("exact phrase") > 800
    assert text[match["match_start"] : match["match_end"]] == "exact phrase"
    assert match["excerpt"] == text[match["excerpt_start"] : match["excerpt_end"]]
    assert "exact phrase" in match["excerpt"] and len(match["excerpt"]) <= 800
    assert match["excerpt_truncated_before"] is True
    assert match["excerpt_truncated_after"] is False
    assert match["inspection_url"] == "/documents/paper/chunks"
    assert "score" not in match and "rank" not in match
    assert "PRIVATE_" not in response.text
    assert result["next_cursor"] is None
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"query": ""},
        {"query": " \t\n"},
        {"query": 1},
        {"query": "x" * 201},
        {"query": "\ud800PRIVATE_QUERY"},
        {"query": "x", "document_ids": None},
        {"query": "x", "document_ids": []},
        {"query": "x", "document_ids": ["\ud800"]},
        {"query": "x", "document_ids": "paper"},
        {"query": "x", "document_ids": ["paper"] * 101},
        {"query": "x", "collection_id": None},
        {"query": "x", "collection_id": ""},
        {"query": "x", "collection_id": "col_" + "0" * 32, "document_ids": ["paper"]},
        {"query": "x", "limit": True},
        {"query": "x", "limit": "1"},
        {"query": "x", "limit": 1.0},
        {"query": "x", "limit": 0},
        {"query": "x", "limit": 51},
        {"query": "x", "cursor": None},
        {"query": "x", "cursor": ""},
        {"query": "x", "cursor": "a" * 4097},
        {"query": "x", "PRIVATE_FIELD": "PRIVATE_INPUT"},
    ],
)
def test_http_strict_validation_is_private_and_non_caching(
    tmp_path: Path, payload: dict[str, object]
) -> None:
    with TestClient(create_app(offline_settings(tmp_path / "corpus.sqlite3"))) as client:
        response = client.post(
            "/research/search",
            content=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
    assert response.status_code == 422, response.text
    assert response.json()["detail"]["code"] == "invalid_search_request"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "PRIVATE" not in response.text


@pytest.mark.parametrize(
    "content",
    [
        b'{"query":"\x80PRIVATE"}',
        '{"query":"x"}'.encode("utf-16"),
        pytest.param('{"query":"PRIVATE_QUERY"}'.encode("utf-16-le"), id="bomless-utf16le"),
        pytest.param('{"query":"PRIVATE_QUERY"}'.encode("utf-16-be"), id="bomless-utf16be"),
        pytest.param('{"query":"PRIVATE_QUERY"}'.encode("utf-32-le"), id="bomless-utf32le"),
        pytest.param('{"query":"PRIVATE_QUERY"}'.encode("utf-32-be"), id="bomless-utf32be"),
        b"{",
        b"[]",
    ],
)
def test_malformed_or_non_utf8_json_is_an_explicit_sanitized_validation_error(
    tmp_path: Path, content: bytes
) -> None:
    with TestClient(create_app(offline_settings(tmp_path / "corpus.sqlite3"))) as client:
        response = client.post(
            "/research/search", content=content, headers={"Content-Type": "application/json"}
        )
    assert response.status_code == 422, response.text
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "PRIVATE" not in response.text


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig"])
def test_utf8_json_preserves_unicode_queries_and_optional_bom(
    tmp_path: Path, encoding: str
) -> None:
    query = "\u7814\u7a76"
    content = json.dumps({"query": query}, ensure_ascii=False).encode(encoding)
    with TestClient(create_app(offline_settings(tmp_path / "corpus.sqlite3"))) as client:
        response = client.post(
            "/research/search", content=content, headers={"Content-Type": "application/json"}
        )
    assert response.status_code == 200, response.text
    assert response.json()["query"] == query


@pytest.mark.parametrize(
    "identifier", ["https://doi.org/10.1/'quoted'", "percent%?#", "\u7814-\U0001f52c", "nul\x00id"]
)
def test_http_python_parity_restart_and_exact_existing_inspection_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, identifier: str
) -> None:
    path = tmp_path / "corpus.sqlite3"
    seed(path, identifier, ("a", "b"))
    settings = offline_settings(path)
    app = create_app(settings)
    before = path.read_bytes()
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", no_work)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", no_work)
    with TestClient(app) as client:
        first = client.post("/research/search", json={"query": "exact phrase", "limit": 1})
        assert first.status_code == 200
        result = LiteralSearchPage.model_validate_json(first.content)
        request = LiteralSearchRequest(query="exact phrase", limit=1)
        assert (
            app.state.container.literal_search.search(request).to_json().encode() == first.content
        )
        inspected = client.get(result.matches[0].inspection_url)
        assert inspected.status_code == 200
        assert inspected.json()["document_id"] == identifier
    with TestClient(create_app(settings)) as client:
        assert client.post(
            "/research/search", json={"query": "exact phrase", "limit": 1}
        ).content == (first.content)
        continued = client.post(
            "/research/search", json={"query": "exact phrase", "cursor": result.next_cursor}
        )
        assert [match["chunk_id"] for match in continued.json()["matches"]] == ["b"]
        assert (
            client.post(
                "/research/search", json={"query": "changed", "cursor": result.next_cursor}
            ).status_code
            == 422
        )
    assert path.read_bytes() == before


@pytest.mark.parametrize("identifier", [".", "..", "paper/../source", "paper/./source"])
def test_nonrepresentable_url_is_explicitly_null(tmp_path: Path, identifier: str) -> None:
    path = tmp_path / "corpus.sqlite3"
    seed(path, identifier, ("a",))
    with TestClient(create_app(offline_settings(path))) as client:
        response = client.post("/research/search", json={"query": "exact phrase"})
    assert response.status_code == 200
    assert response.json()["matches"][0]["inspection_url"] is None


def test_http_collection_and_storage_errors_are_sanitized(tmp_path: Path) -> None:
    path = tmp_path / "PRIVATE_STORAGE.sqlite3"
    app = create_app(offline_settings(path))
    seed(path)
    with TestClient(app) as client:
        missing = client.post(
            "/research/search", json={"query": "PRIVATE_QUERY", "collection_id": "col_" + "0" * 32}
        )
        assert missing.status_code == 404
        assert missing.json()["detail"]["code"] == "collection_not_found"
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.execute("UPDATE chunks SET text = zeroblob(10)")
        corrupt = client.post("/research/search", json={"query": "PRIVATE_QUERY"})
        assert corrupt.status_code == 409
        assert corrupt.json()["detail"]["code"] == "invalid_chunk_record"
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.execute("DROP TABLE chunks")
        unavailable = client.post("/research/search", json={"query": "PRIVATE_QUERY"})
        assert unavailable.status_code == 503
        assert unavailable.json()["detail"]["code"] == "search_storage_unavailable"
    for response in (missing, corrupt, unavailable):
        assert "PRIVATE" not in response.text
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["x-content-type-options"] == "nosniff"


def test_openapi_exposes_strict_limits_response_offsets_and_explicit_errors(tmp_path: Path) -> None:
    with TestClient(create_app(offline_settings(tmp_path / "corpus.sqlite3"))) as client:
        schema = client.get("/openapi.json").json()
    operation = schema["paths"]["/research/search"]["post"]
    assert {"200", "404", "409", "413", "422", "503", "504"} <= set(operation["responses"])
    models = schema["components"]["schemas"]
    fields = models["LiteralSearchRequest"]["properties"]
    assert fields["query"]["maxLength"] == 200
    assert fields["limit"]["maximum"] == 50
    assert fields["document_ids"]["minItems"] == 1
    assert fields["document_ids"]["maxItems"] == 100
    assert "anyOf" not in fields["collection_id"]
    assert models["LiteralSearchRequest"]["additionalProperties"] is False
    assert models["LiteralSearchPage"]["properties"]["matches"]["maxItems"] == 50
    assert models["LiteralMatch"]["properties"]["excerpt"]["maxLength"] == 800
