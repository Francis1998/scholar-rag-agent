"""Real-SQLite regressions for empty labels across read-only corpus surfaces."""

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from scripts.demo_corpus_explorer import browsing_guards
from scripts.demo_evidence_export import offline_settings

from api.dependencies import AppContainer
from retrieval.models import Chunk, Document
from storage.document_catalog import DocumentCatalogError, SQLiteDocumentCatalog, _prefix
from storage.document_chunks import DocumentChunksError, SQLiteDocumentChunks
from storage.document_store import SQLiteDocumentStore
from storage.literal_search import LiteralSearchRequest, SQLiteLiteralSearch
from storage.source_context import SQLiteSourceContext
from tests.test_corpus_explorer import ExplorerAPI, ExplorerDOM, assert_headers, values
from tests.test_corpus_explorer import api as api

LABELS = [("", "synthetic"), ("Synthetic \u7814", ""), ("", "")]
SURFACES = ["catalog", "chunks", "context", "search"]
TEXT = "Synthetic \U0001f52c passage."


def seed(path: Path, title: str, source: str) -> None:
    SQLiteDocumentStore(path).add_documents(
        [
            Document(document_id=identifier, title=title, source=source, text="PRIVATE_BODY")
            for identifier in ("d0", "d1")
        ],
        [
            Chunk(
                chunk_id=f"{identifier}:c{index + 1}",
                document_id=identifier,
                title=title,
                source=source,
                text=TEXT if identifier == "d0" else "EXCLUDED passage.",
                metadata={"chunk_index": str(index), "private": "PRIVATE_METADATA"},
            )
            for identifier, index in (("d0", 1), ("d0", 0), ("d1", 0))
        ],
    )


@pytest.mark.parametrize(("title", "source"), LABELS)
@pytest.mark.parametrize("encoding", ["UTF-8", "UTF-16le", "UTF-16be"])
@pytest.mark.parametrize("surface", SURFACES)
def test_empty_labels_preserve_python_pages_offsets_and_restart(
    tmp_path: Path, title: str, source: str, encoding: str, surface: str
) -> None:
    path = tmp_path / "corpus.sqlite3"
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(f"PRAGMA encoding = '{encoding}'")
        connection.execute("CREATE TABLE marker (value TEXT)")
    seed(path, title, source)
    before = path.read_bytes()
    if surface == "catalog":
        first = SQLiteDocumentCatalog(path).list_documents(limit=1)
        assert first.next_cursor == "d0"
        second = SQLiteDocumentCatalog(path).list_documents(limit=1, cursor=first.next_cursor)
        records = first.documents + second.documents
        assert [record.document_id for record in records] == ["d0", "d1"]
        assert [record.chunk_count for record in records] == [2, 1]
        assert second.next_cursor is None
        assert SQLiteDocumentCatalog(path).list_documents(limit=1) == first
    elif surface == "chunks":
        first = SQLiteDocumentChunks(path).list_chunks("d0", limit=1)
        assert first.next_cursor is not None
        second = SQLiteDocumentChunks(path).list_chunks("d0", limit=1, cursor=first.next_cursor)
        records = first.chunks + second.chunks
        assert [record.chunk_id for record in records] == ["d0:c1", "d0:c2"]
        assert [record.chunk_index for record in records] == [0, 1]
        assert second.next_cursor is None
        assert SQLiteDocumentChunks(path).list_chunks("d0", limit=1) == first
    elif surface == "context":
        window = SQLiteSourceContext(path).read("d0", "d0:c1", before=0, after=1)
        records = window.chunks
        assert [record.chunk_id for record in records] == ["d0:c1", "d0:c2"]
        assert [record.chunk_index for record in records] == [0, 1]
        assert [record.is_anchor for record in records] == [True, False]
        assert window.returned_before == 0 and window.returned_after == 1
        assert not window.has_more_before and not window.has_more_after
        assert SQLiteSourceContext(path).read("d0", "d0:c1", before=0, after=1) == window
    else:
        request = LiteralSearchRequest(query="passage", document_ids=["d0"], limit=1)
        first = SQLiteLiteralSearch(path).search(request)
        assert first.next_cursor is not None
        second = SQLiteLiteralSearch(path).search(
            LiteralSearchRequest(
                query="passage", document_ids=["d0"], limit=1, cursor=first.next_cursor
            )
        )
        records = first.matches + second.matches
        assert [record.chunk_id for record in records] == ["d0:c1", "d0:c2"]
        assert first.document_ids == second.document_ids == ("d0",)
        assert second.next_cursor is None
        assert SQLiteLiteralSearch(path).search(request).to_json() == first.to_json()
        for record in records:
            assert (record.match_start, record.match_end) == (TEXT.index("passage"), len(TEXT) - 1)
            assert (record.excerpt_start, record.excerpt_end) == (0, len(TEXT))
            assert record.excerpt == TEXT and record.text_characters == len(TEXT)
            assert not record.excerpt_truncated_before and not record.excerpt_truncated_after
    for record in records:
        assert record.title == title and record.source == source
        assert not record.title_truncated and not record.source_truncated
        if surface in {"chunks", "context"}:
            assert record.document_id == "d0"
            assert record.text == TEXT and not record.text_truncated
    assert path.read_bytes() == before
    assert set(tmp_path.iterdir()) == {path}


@pytest.mark.parametrize(("title", "source"), LABELS)
@pytest.mark.parametrize("surface", SURFACES)
def test_empty_labels_remain_exact_in_json_and_browser_after_app_restart(
    api: ExplorerAPI, title: str, source: str, surface: str
) -> None:
    api.seed(chunk_ids=("a", "b"), title=title, source=source, text=TEXT)
    api.seed("z-excluded", ("excluded",), text="EXCLUDED passage.")
    original = None
    for restart in (False, True):
        if restart:
            api.app.state.container = AppContainer(offline_settings(api.path))
        before = api.path.read_bytes()
        with browsing_guards(api.app.state.container) as guards:
            if surface == "catalog":
                response = api.client.get("/documents", params={"limit": 1})
                browser = api.client.get("/explore", params={"limit": 1})
                field, title_class, source_class = "documents", "document-title", "document-source"
            elif surface == "chunks":
                response = api.client.get("/documents/selected/chunks", params={"limit": 1})
                browser = api.client.get(
                    "/explore/document", params={"document_id": "selected", "limit": 1}
                )
                field, title_class, source_class = "chunks", "chunk-title", "chunk-source"
            elif surface == "context":
                params = {"document_id": "selected", "chunk_id": "a", "before": 0, "after": 0}
                response = api.client.get("/documents/context", params=params)
                browser = api.client.get("/explore/context", params=params)
                field, title_class, source_class = "chunks", "chunk-title", "chunk-source"
            else:
                response = api.client.post(
                    "/research/search",
                    json={"query": "passage", "document_ids": ["selected"], "limit": 1},
                )
                browser = api.client.get(
                    "/explore/search",
                    params={
                        "query": "passage",
                        "scope": "document",
                        "scope_id": "selected",
                        "limit": 1,
                    },
                )
                field, title_class, source_class = "matches", "chunk-title", "chunk-source"
            assert response.status_code == browser.status_code == 200, response.text
            assert response.headers["cache-control"] == "no-store"
            assert response.headers["x-content-type-options"] == "nosniff"
            assert_headers(browser)
            records = response.json()[field]
            assert len(records) == 1
            assert records[0]["document_id"] == "selected"
            assert records[0]["title"] == title and records[0]["source"] == source
            assert not records[0]["title_truncated"] and not records[0]["source_truncated"]
            root = ExplorerDOM(browser.text).root
            assert values(root, source_class) == [source]
            # The catalog's existing empty-title navigation label is not stored metadata.
            assert values(root, title_class) == (
                [] if surface == "catalog" and not title else [title]
            )
            if surface != "catalog":
                text_field = "excerpt" if surface == "search" else "text"
                css_class = "search-excerpt" if surface == "search" else "passage-text"
                assert records[0][text_field] == TEXT and values(root, css_class) == [TEXT]
            assert "EXCLUDED" not in response.text + browser.text
            assert not any(guard.call_count for guard in guards.values())
        assert api.path.read_bytes() == before
        current = (response.content, browser.content)
        if original is not None:
            assert current == original
        original = current


@pytest.mark.parametrize("surface", SURFACES)
@pytest.mark.parametrize("field", ["title", "source"])
@pytest.mark.parametrize(
    "value", ["NULL", "X''", "X'73796e746865746963'", "0", "CAST(X'80' AS TEXT)"]
)
def test_empty_label_fix_does_not_accept_corrupt_labels(
    tmp_path: Path, surface: str, field: str, value: str
) -> None:
    path = tmp_path / "corpus.sqlite3"
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(
            "CREATE TABLE documents (document_id TEXT PRIMARY KEY, title, text TEXT, source, "
            "metadata TEXT)"
        )
        connection.execute(
            "CREATE TABLE chunks (chunk_id TEXT PRIMARY KEY, document_id TEXT, title, text TEXT, "
            "source, metadata TEXT)"
        )
    seed(path, "", "")
    table = "documents" if surface == "catalog" else "chunks"
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(f"UPDATE {table} SET {field} = {value}")  # noqa: S608
    before = path.read_bytes()
    error = DocumentCatalogError if surface == "catalog" else DocumentChunksError
    with pytest.raises(error):
        if surface == "catalog":
            SQLiteDocumentCatalog(path).list_documents(limit=1)
        elif surface == "chunks":
            SQLiteDocumentChunks(path).list_chunks("d0", limit=1)
        elif surface == "context":
            SQLiteSourceContext(path).read("d0", "d0:c1")
        else:
            SQLiteLiteralSearch(path).search(
                LiteralSearchRequest(query="passage", document_ids=["d0"], limit=1)
            )
    assert path.read_bytes() == before


@pytest.mark.parametrize("value", [None, "", 0, False, bytearray(), b"\x80"])
def test_prefix_stays_strict_for_nonbytes_and_malformed_text(value: object) -> None:
    with pytest.raises(DocumentCatalogError):
        _prefix(value, "UTF-8", 300)
