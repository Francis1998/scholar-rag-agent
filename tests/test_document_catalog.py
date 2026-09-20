"""Offline acceptance tests for discovering an existing scientific corpus."""

import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError
from scripts.demo_evidence_export import offline_settings

from api.application import create_app
from api.dependencies import AppContainer
from retrieval.models import Chunk, Document


@dataclass
class CatalogAPI:
    application: FastAPI
    client: TestClient
    container: AppContainer
    path: Path


def _no_work(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("Document discovery must not retrieve, generate, or use the network.")


@pytest.fixture
def catalog_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[CatalogAPI]:
    path = tmp_path / "corpus.sqlite3"
    application = create_app(offline_settings(path))
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _no_work)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", _no_work)
    with TestClient(application) as client:
        yield CatalogAPI(application, client, application.state.container, path)


def _seed(
    api: CatalogAPI,
    document_id: str,
    *,
    title: str = "Synthetic methods",
    source: str = "synthetic:catalog",
    chunks: int = 1,
) -> None:
    document = Document(
        document_id=document_id,
        title=title,
        source=source,
        text="private-body-sentinel",
        metadata={"private": "private-metadata-sentinel"},
    )
    api.container.document_store.add_documents(
        [document],
        [
            Chunk(
                chunk_id=f"{document_id}-{index}",
                document_id=document_id,
                title=title,
                text=document.text,
                source=source,
                metadata=document.metadata,
            )
            for index in range(chunks)
        ],
    )


def _page(api: CatalogAPI, **params: str | int) -> dict[str, Any]:
    response = api.client.get("/documents", params=params)
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    result: dict[str, Any] = response.json()
    assert set(result) == {"documents", "next_cursor"}
    assert "private-body-sentinel" not in response.text
    assert "private-metadata-sentinel" not in response.text
    return result


def test_empty_document_catalog(catalog_api: CatalogAPI) -> None:
    assert _page(catalog_api) == {"documents": [], "next_cursor": None}


def test_catalog_reports_bounded_summaries_without_running_the_agent(
    catalog_api: CatalogAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(catalog_api, "b", chunks=3)
    _seed(catalog_api, "a", chunks=0)
    before = catalog_api.path.read_bytes()
    for target, method in (
        (catalog_api.container.document_store, "list_chunks"),
        (catalog_api.container.hybrid_retriever, "retrieve"),
        (catalog_api.container.llm, "generate"),
        (catalog_api.container.runner, "run"),
        (catalog_api.container.event_log, "list_events"),
        (catalog_api.container.event_log, "append_event"),
    ):
        monkeypatch.setattr(target, method, _no_work)
    page = _page(catalog_api)
    assert [record["document_id"] for record in page["documents"]] == ["a", "b"]
    assert [record["chunk_count"] for record in page["documents"]] == [0, 3]
    assert page["documents"][0] == {
        "document_id": "a",
        "title": "Synthetic methods",
        "title_truncated": False,
        "source": "synthetic:catalog",
        "source_truncated": False,
        "chunk_count": 0,
    }
    assert page["next_cursor"] is None
    assert catalog_api.path.read_bytes() == before


def test_discovered_ids_work_in_scoped_queries_after_restart(catalog_api: CatalogAPI) -> None:
    api = catalog_api
    ingested = api.client.post(
        "/ingest/text",
        json={
            "title": "Synthetic GraphRAG methods",
            "text": "Synthetic data only. GraphRAG retrieves connected research passages.",
            "source": "synthetic:restart",
        },
    )
    assert ingested.status_code == 200
    api.client.post(
        "/ingest/text",
        json={"title": "Unselected", "text": "Unrelated content.", "source": "synthetic:other"},
    ).raise_for_status()
    api.application.state.container = AppContainer(offline_settings(api.path))
    page = _page(api, source="synthetic:restart", title="graphrag")
    assert len(page["documents"]) == 1
    identifier = page["documents"][0]["document_id"]
    assert identifier == ingested.json()["document_id"]
    query = api.client.post(
        "/query", json={"query": "GraphRAG research", "document_ids": [identifier]}
    )
    assert query.status_code == 200
    result = query.json()["result"]
    assert result["state"] == "DONE"
    exported = api.client.get(f"/runs/{result['run_id']}/export")
    assert exported.status_code == 200
    assert {s["chunk"]["document_id"] for s in exported.json()["snapshot"]["sources"]} == {
        identifier
    }


def test_keyset_pagination_is_exclusive_and_does_not_repeat_reingested_ids(
    catalog_api: CatalogAPI,
) -> None:
    api = catalog_api
    for identifier in ("a", "b", "c", "d", "e"):
        _seed(api, identifier)
    first = _page(api, limit=2)
    assert [r["document_id"] for r in first["documents"]] == ["a", "b"]
    assert first["next_cursor"] == "b"
    _seed(api, "a", title="Updated title")
    _seed(api, "aa", title="Inserted behind the cursor")
    _seed(api, "f")
    second = _page(api, limit=2, cursor=first["next_cursor"])
    third = _page(api, limit=2, cursor=second["next_cursor"])
    assert [r["document_id"] for r in second["documents"]] == ["c", "d"]
    assert [r["document_id"] for r in third["documents"]] == ["e", "f"]
    assert third["next_cursor"] is None
    assert _page(api, cursor="z") == {"documents": [], "next_cursor": None}


def test_source_and_literal_title_filters_apply_before_pagination(catalog_api: CatalogAPI) -> None:
    api = catalog_api
    _seed(api, "a", source="wrong", title="100%_Graph methods")
    _seed(api, "b", source="selected", title="Other title")
    _seed(api, "c", source="selected", title="100%_Graph methods")
    _seed(api, "d", source="selected", title="100%_GRAPH limitations")
    _seed(api, "e", source="SELECTED", title="100%_Graph methods")
    first = _page(api, limit=1, source="selected", title="%_graph")
    assert [r["document_id"] for r in first["documents"]] == ["c"]
    assert first["next_cursor"] == "c"
    second = _page(api, limit=1, cursor="c", source="selected", title="%_graph")
    assert [r["document_id"] for r in second["documents"]] == ["d"]
    assert second["next_cursor"] is None
    assert _page(api, source="' OR 1=1 --")["documents"] == []
    assert _page(api, title="' OR 1=1 --")["documents"] == []
    assert _page(api, cursor="' OR 1=1 --", source="missing")["documents"] == []


def test_unicode_and_nul_prefixes_are_bounded_without_reading_metadata(
    catalog_api: CatalogAPI,
) -> None:
    title = "a\x00" + "\U0001f52c" * 400
    source = "\u7814" * 600
    _seed(catalog_api, "unicode", title=title, source=source)
    with sqlite3.connect(catalog_api.path) as connection:
        connection.execute("UPDATE documents SET metadata = 'not-json'")
    summary = _page(catalog_api)["documents"][0]
    assert summary["title"] == title[:300]
    assert summary["title_truncated"] is True
    assert summary["source"] == source[:512]
    assert summary["source_truncated"] is True


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("limit", "0"),
        ("limit", "101"),
        ("limit", "1.5"),
        ("limit", "true"),
        ("cursor", ""),
        ("cursor", " a"),
        ("cursor", "a" * 129),
        ("source", ""),
        ("source", "a" * 513),
        ("title", ""),
        ("title", "a" * 301),
    ],
)
def test_invalid_pagination_and_filters_are_rejected(
    catalog_api: CatalogAPI, parameter: str, value: str
) -> None:
    response = catalog_api.client.get("/documents", params={parameter: value})
    assert response.status_code == 422


@pytest.mark.parametrize("identifier", ["", " leading", "trailing ", "x" * 129])
def test_unselectable_stored_ids_fail_instead_of_being_truncated(
    catalog_api: CatalogAPI, identifier: str
) -> None:
    _seed(catalog_api, identifier)
    response = catalog_api.client.get("/documents")
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "invalid_document_record"
    assert "private-body-sentinel" not in response.text
    assert response.headers["cache-control"] == "no-store"


def test_python_catalog_validates_requests_and_does_not_create_missing_database(
    tmp_path: Path, catalog_api: CatalogAPI
) -> None:
    from storage.document_catalog import SQLiteDocumentCatalog

    reader = SQLiteDocumentCatalog(catalog_api.path)
    for value in (0, 101, True, 1.5, "2"):
        with pytest.raises(ValidationError):
            reader.list_documents(limit=value)  # type: ignore[arg-type]
    missing = tmp_path / "missing.sqlite3"
    with pytest.raises(sqlite3.OperationalError):
        SQLiteDocumentCatalog(missing).list_documents()
    assert not missing.exists()


def test_document_catalog_is_independent_of_chunks_and_has_index(catalog_api: CatalogAPI) -> None:
    _seed(catalog_api, "no-chunks", chunks=0)
    assert _page(catalog_api)["documents"][0]["chunk_count"] == 0
    with sqlite3.connect(catalog_api.path) as connection:
        indexes = {row[1] for row in connection.execute("PRAGMA index_list(chunks)")}
    assert "idx_chunks_document_id" in indexes


def test_page_size_cap_and_exact_label_boundaries(catalog_api: CatalogAPI) -> None:
    for index in range(101):
        _seed(catalog_api, f"doc-{index:03}", title="t" * 300, source="s" * 512, chunks=0)
    page = _page(catalog_api, limit=100)
    assert len(page["documents"]) == 100
    assert page["next_cursor"] == "doc-099"
    assert all(
        not row["title_truncated"] and not row["source_truncated"] for row in page["documents"]
    )
    assert len(_page(catalog_api, limit=100, cursor=page["next_cursor"])["documents"]) == 1


def test_catalog_decodes_utf16_database_and_does_not_change_encoding(tmp_path: Path) -> None:
    from storage.document_catalog import SQLiteDocumentCatalog
    from storage.document_store import SQLiteDocumentStore

    path = tmp_path / "utf16.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA encoding = 'UTF-16le'")
        connection.execute("CREATE TABLE marker (value TEXT)")
    store = SQLiteDocumentStore(path)
    title = "\U0001f52c" * 301
    store.add_documents(
        [Document(document_id="selected", title=title, text="private", source="synthetic")], []
    )
    before = path.read_bytes()
    summary = SQLiteDocumentCatalog(path).list_documents().documents[0]
    assert summary.title == title[:300] and summary.title_truncated
    assert path.read_bytes() == before
