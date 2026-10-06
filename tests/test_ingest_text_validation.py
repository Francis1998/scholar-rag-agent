"""Text ingestion rejects unusable API input without changing valid content or storage."""

import json
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from typing import NoReturn
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from scripts.demo_evidence_export import offline_settings

import api.routes as routes_module
from api.application import create_app
from api.dependencies import AppContainer
from api.schemas import IngestTextRequest
from ingestion.chunking import stable_id
from llm.providers import AnthropicAdapter, GeminiAdapter, KimiAdapter, OpenAIAdapter
from retrieval.models import Chunk

PADDED_TEXT = "\t \u00a0GraphRAG  connects\nAlpha and Beta. caf\u00e9 \u7814\u7a76\u2003\n"
NORMALIZED_TEXT = "GraphRAG connects Alpha and Beta. caf\u00e9 \u7814\u7a76"


def deny_external_work(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("Text ingestion validation tests must not call models or the network.")


@pytest.fixture
def ingest_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[FastAPI]:
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", deny_external_work)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", deny_external_work)
    application = create_app(offline_settings(tmp_path / "ingest-text-validation.sqlite3"))
    container: AppContainer = application.state.container
    model_guards: list[AsyncMock] = []
    for owner in (
        container.llm,
        container.llm._router._adapters["fake"],
        OpenAIAdapter,
        AnthropicAdapter,
        GeminiAdapter,
        KimiAdapter,
    ):
        guard = AsyncMock(side_effect=deny_external_work)
        monkeypatch.setattr(owner, "generate", guard)
        model_guards.append(guard)
    yield application
    for guard in model_guards:
        guard.assert_not_called()


@pytest.fixture
def ingestion_spies(ingest_app: FastAPI, monkeypatch: pytest.MonkeyPatch) -> list[Mock]:
    container: AppContainer = ingest_app.state.container
    hybrid = container.hybrid_retriever
    spies: list[Mock] = []
    for owner, method in (
        (routes_module, "stable_id"),
        (container.ingestion_pipeline, "ingest_documents"),
        (container.ingestion_pipeline._chunker, "chunk"),
        (container.document_store, "add_documents"),
        (hybrid, "add_chunks"),
        (hybrid._dense_retriever, "add_chunks"),
        (hybrid._sparse_retriever, "add_chunks"),
        (container.graph_builder, "index_chunks"),
        (container.graph_builder._extractor, "extract"),
        (container.graph_store, "replace_chunk"),
        (container.graph_store, "add_mentions"),
        (container.graph_store, "add_edges"),
        (container.event_log, "append_transition"),
        (container.event_log, "append_event"),
    ):
        spy = Mock(name=method, wraps=getattr(owner, method))
        monkeypatch.setattr(owner, method, spy)
        spies.append(spy)
    for owner in (hybrid, hybrid._dense_retriever, hybrid._sparse_retriever):
        spy = AsyncMock(name="retrieve", wraps=owner.retrieve)
        monkeypatch.setattr(owner, "retrieve", spy)
        spies.append(spy)
    return spies


def database_dump(database_path: Path) -> list[str]:
    with closing(sqlite3.connect(database_path)) as connection:
        return list(connection.iterdump())


@pytest.mark.parametrize(
    ("text_fields", "error_type"),
    [
        pytest.param({"text": ""}, "string_too_short", id="empty"),
        pytest.param({"text": "   "}, "value_error", id="spaces"),
        pytest.param({"text": "\t\r\n\v\f"}, "value_error", id="ascii-whitespace"),
        pytest.param({"text": "\u00a0"}, "value_error", id="nonbreaking-space"),
        pytest.param({"text": "\u0085"}, "value_error", id="unicode-next-line"),
        pytest.param({"text": "\u2003"}, "value_error", id="em-space"),
        pytest.param({"text": "\u2028\u2029"}, "value_error", id="unicode-line-separators"),
        pytest.param({"text": "\u3000"}, "value_error", id="ideographic-space"),
        pytest.param(
            {"text": "\u1680\u2000\u2009\u202f\u205f"},
            "value_error",
            id="unicode-space-characters",
        ),
        pytest.param({"text": " \t\u00a0\u2003\u3000\n"}, "value_error", id="mixed-whitespace"),
        pytest.param({}, "missing", id="missing"),
        pytest.param({"text": None}, "string_type", id="null"),
        pytest.param({"text": 12}, "string_type", id="integer"),
        pytest.param({"text": 1.5}, "string_type", id="float"),
        pytest.param({"text": False}, "string_type", id="boolean"),
        pytest.param({"text": []}, "string_type", id="list"),
        pytest.param({"text": {}}, "string_type", id="object"),
    ],
)
def test_invalid_text_returns_422_before_ingestion(
    ingest_app: FastAPI,
    ingestion_spies: list[Mock],
    tmp_path: Path,
    text_fields: dict[str, object],
    error_type: str,
) -> None:
    database_path = tmp_path / "ingest-text-validation.sqlite3"
    before = database_dump(database_path)
    with TestClient(ingest_app) as client:
        response = client.post(
            "/ingest/text", json={"title": "Synthetic validation note", **text_fields}
        )

    assert response.status_code == 422, response.text
    errors = response.json()["detail"]
    assert len(errors) == 1
    assert errors[0]["loc"] == ["body", "text"]
    assert errors[0]["type"] == error_type
    if error_type == "value_error":
        assert "text must not be blank" in errors[0]["msg"]
    for spy in ingestion_spies:
        spy.assert_not_called()
    assert database_dump(database_path) == before


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("", id="empty"),
        pytest.param(" \t\r\n", id="ascii-whitespace"),
        pytest.param("\u00a0\u2003\u3000", id="unicode-whitespace"),
    ],
)
def test_blank_text_leaves_existing_corpus_and_events_unchanged(
    ingest_app: FastAPI, tmp_path: Path, text: str
) -> None:
    container: AppContainer = ingest_app.state.container
    database_path = tmp_path / "ingest-text-validation.sqlite3"
    with TestClient(ingest_app) as client:
        seeded = client.post(
            "/ingest/text", json={"title": "Existing note", "text": "GraphRAG Alpha Beta."}
        )
        assert seeded.status_code == 200, seeded.text
        container.event_log.append_event("test-agent", "existing-run", "test", {"seed": True})
        before = database_dump(database_path)
        response = client.post("/ingest/text", json={"title": "Blank note", "text": text})

    # Compare all persisted rows, including documents, chunks, graph tables, and events.
    assert database_dump(database_path) == before
    assert response.status_code == 422, response.text


@pytest.mark.parametrize(
    ("title", "source_fields"),
    [
        pytest.param("Synthetic ingestion note", {}, id="default-source"),
        pytest.param(
            "  Padded title\t", {"source": "  synthetic:ingest\t"}, id="padded-title-source"
        ),
        pytest.param("", {"source": ""}, id="empty-title-source"),
    ],
)
async def test_nonblank_text_preserves_document_id_content_and_chunks(
    ingest_app: FastAPI, tmp_path: Path, title: str, source_fields: dict[str, str]
) -> None:
    source = source_fields.get("source", "api")
    payload = {"title": title, "text": PADDED_TEXT, **source_fields}
    document_id = stable_id(
        json.dumps([source, title, PADDED_TEXT], ensure_ascii=False, separators=(",", ":")),
        "doc-api-v2",
    )
    expected_chunk = Chunk(
        chunk_id=stable_id(f"{document_id}:0:{NORMALIZED_TEXT}", "chunk"),
        document_id=document_id,
        title=title,
        text=NORMALIZED_TEXT,
        source=source,
        metadata={"source_type": "api", "chunk_index": "0"},
    )
    assert IngestTextRequest.model_validate(payload).text == PADDED_TEXT
    assert document_id != stable_id(
        json.dumps([source, title, PADDED_TEXT.strip()], ensure_ascii=False, separators=(",", ":")),
        "doc-api-v2",
    )
    with TestClient(ingest_app) as client:
        response = client.post("/ingest/text", json=payload)
    assert response.status_code == 200, response.text
    assert response.json() == {
        "document_id": document_id,
        "chunk_ids": [expected_chunk.chunk_id],
    }
    with closing(sqlite3.connect(tmp_path / "ingest-text-validation.sqlite3")) as connection:
        assert connection.execute(
            "SELECT document_id, title, text, source, metadata FROM documents"
        ).fetchall() == [(document_id, title, PADDED_TEXT, source, '{"source_type": "api"}')]
        assert connection.execute("SELECT chunk_id, text FROM graph_chunks").fetchall() == [
            (expected_chunk.chunk_id, NORMALIZED_TEXT)
        ]
    container: AppContainer = ingest_app.state.container
    assert container.document_store.list_chunks() == [expected_chunk]
    for retriever in (
        container.hybrid_retriever,
        container.hybrid_retriever._dense_retriever,
        container.hybrid_retriever._sparse_retriever,
    ):
        results = await retriever.retrieve("GraphRAG")
        assert [result.chunk for result in results] == [expected_chunk]
    assert container.event_log.list_events() == []


def test_ingest_openapi_documents_nonblank_text(ingest_app: FastAPI) -> None:
    with TestClient(ingest_app) as client:
        response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()["components"]["schemas"]["IngestTextRequest"]
    assert "text" in schema["required"]
    assert schema["properties"]["text"]["type"] == "string"
    assert schema["properties"]["text"]["minLength"] == 1
    assert "non-whitespace" in schema["properties"]["text"]["description"]
