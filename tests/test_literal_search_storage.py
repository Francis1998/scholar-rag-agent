"""Real-SQLite literal semantics, bounded projections, cursors, and read-only snapshots."""

import base64
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from pydantic import ValidationError

from retrieval.models import Chunk, Document
from storage.document_chunks import DocumentChunksError
from storage.document_store import SQLiteDocumentStore
from storage.literal_search import (
    MAX_RESPONSE_BYTES,
    MAX_TEXT_BYTES,
    LiteralSearchError,
    LiteralSearchPage,
    LiteralSearchRequest,
    SQLiteLiteralSearch,
)
from storage.paper_collections import CollectionError, SQLitePaperCollections


def seed(
    path: Path,
    document_id: str = "paper",
    chunk_ids: tuple[str, ...] = ("c", "a", "b"),
    *,
    text: str = "Before exact phrase and after.",
    title: str = "Synthetic title",
    source: str = "synthetic:search",
) -> None:
    SQLiteDocumentStore(path).add_documents(
        [Document(document_id=document_id, title=title, source=source, text="PRIVATE_BODY")],
        [
            Chunk(
                document_id=document_id,
                chunk_id=identifier,
                text=text,
                title=title,
                source=source,
                metadata={"private": "PRIVATE_METADATA"},
            )
            for identifier in chunk_ids
        ],
    )


def search(path: Path, **options: object) -> LiteralSearchPage:
    return SQLiteLiteralSearch(path).search(
        LiteralSearchRequest.model_validate({"query": "exact phrase", **options})
    )


@pytest.mark.parametrize(
    "query",
    ["exact phrase", "cafe\u0301", "\u03b2\U0001f52c\u7814", "%_\"'\\", "' OR 1=1 --", " phrase "],
)
def test_exact_unicode_first_occurrence_and_literal_metacharacters(
    tmp_path: Path, query: str
) -> None:
    path = tmp_path / "corpus.sqlite3"
    prefix = "\u7814\U0001f52c" * 700
    text = prefix + query + " second occurrence: " + query + "z" * 1000
    seed(path, chunk_ids=("a",), text=text)
    match = search(path, query=query).matches[0]
    assert match.match_start == text.index(query) == len(prefix)
    assert match.match_end == len(prefix) + len(query)
    assert text[match.match_start : match.match_end] == query
    assert match.excerpt == text[match.excerpt_start : match.excerpt_end]
    assert match.excerpt_start == match.match_start - 120
    assert len(match.excerpt) == 800
    assert match.excerpt_truncated_before and match.excerpt_truncated_after
    assert match.text_characters == len(text)


@pytest.mark.parametrize("query", ["Exact phrase", "exact", "cafe\u0301", "%", "phrase"])
def test_case_normalization_and_tokens_are_not_silently_expanded(
    tmp_path: Path, query: str
) -> None:
    path = tmp_path / "corpus.sqlite3"
    seed(path, chunk_ids=("a",), text="EXACT PHRASE caf\u00e9")
    result = search(path, query=query)
    assert result.matches == [] and result.next_cursor is None


@pytest.mark.parametrize("tail", ["", "x" * 788, "x" * 789])
def test_exact_excerpt_boundary_and_first_character_matches(tmp_path: Path, tail: str) -> None:
    path = tmp_path / "corpus.sqlite3"
    text = "exact phrase" + tail
    seed(path, chunk_ids=("a",), text=text)
    match = search(path).matches[0]
    assert match.match_start == match.excerpt_start == 0
    assert match.match_end == 12
    assert match.excerpt == text[:800]
    assert match.excerpt_end == min(800, len(text))
    assert match.excerpt_truncated_before is False
    assert match.excerpt_truncated_after == (len(text) > 800)


def test_scope_precedes_limit_and_unknown_ids_never_broaden_it(tmp_path: Path) -> None:
    path = tmp_path / "corpus.sqlite3"
    seed(path, "a-excluded", tuple(f"a-{i:03}" for i in range(60)))
    seed(path, "z-selected", ("z-c", "z-a", "z-b"))
    selected = search(path, document_ids=[" z-selected ", "missing", "z-selected"], limit=1)
    assert selected.document_ids == ("missing", "z-selected")
    assert [(match.document_id, match.chunk_id) for match in selected.matches] == [
        ("z-selected", "z-a")
    ]
    assert selected.next_cursor
    second = search(
        path,
        document_ids=["z-selected", "missing"],
        cursor=selected.next_cursor,
        limit=50,
    )
    assert [match.chunk_id for match in second.matches] == ["z-b", "z-c"]
    assert second.next_cursor is None
    empty = search(path, document_ids=["missing"])
    assert empty.matches == [] and empty.document_ids == ("missing",)


def test_pagination_is_exclusive_deterministic_and_not_a_cross_request_snapshot(
    tmp_path: Path,
) -> None:
    path = tmp_path / "corpus.sqlite3"
    seed(path, "a-paper", ("same-c", "same-a", "same-b"))
    seed(path, "z-paper", ("other-a",))
    first = search(path, limit=2)
    assert [match.chunk_id for match in first.matches] == ["same-a", "same-b"]
    second = search(path, limit=2, cursor=first.next_cursor)
    assert [(m.document_id, m.chunk_id) for m in second.matches] == [
        ("a-paper", "same-c"),
        ("z-paper", "other-a"),
    ]
    assert second.next_cursor is None
    assert search(path, limit=2, cursor=first.next_cursor) == second
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("DELETE FROM chunks WHERE chunk_id = 'same-b'")
    seed(path, "a-paper", ("same-aa", "same-bb"))
    continued = search(path, cursor=first.next_cursor)
    assert [match.chunk_id for match in continued.matches] == ["same-bb", "same-c", "other-a"]
    assert "same-aa" not in [match.chunk_id for match in continued.matches]
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("DELETE FROM chunks")
    assert search(path, cursor=first.next_cursor).matches == []


@pytest.mark.parametrize(
    "change",
    [
        {"query": "Exact phrase"},
        {"query": "exact phrase "},
        {"document_ids": ["paper"]},
        {"document_ids": ["unknown"]},
    ],
)
def test_cursor_rejects_changed_query_or_scope(tmp_path: Path, change: dict[str, object]) -> None:
    path = tmp_path / "corpus.sqlite3"
    seed(path)
    cursor = search(path, limit=1).next_cursor
    with pytest.raises(LiteralSearchError) as error:
        search(path, cursor=cursor, **change)
    assert error.value.code == "invalid_search_cursor" and error.value.status_code == 422


@pytest.mark.parametrize(
    "change",
    [
        {"version": 2},
        {"version": True},
        {"version": 1.0},
        {"chunk_id": ""},
        {"chunk_id": "\ud800"},
        {"document_id": " spaced "},
        {"fingerprint": "not-a-fingerprint"},
        {"extra": "private"},
    ],
)
def test_cursor_rejects_malformed_version_identity_and_noncanonical_encoding(
    tmp_path: Path, change: dict[str, object]
) -> None:
    path = tmp_path / "corpus.sqlite3"
    seed(path)
    cursor = search(path, limit=1).next_cursor
    assert cursor is not None
    decoded = json.loads(base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)))
    decoded.update(change)
    forged = base64.urlsafe_b64encode(json.dumps(decoded).encode()).decode().rstrip("=")
    for value in (forged, cursor + "=", "%%%%", "e30"):
        with pytest.raises(LiteralSearchError, match="next_cursor"):
            search(path, cursor=value)


def test_collection_scope_revision_missing_members_and_missing_storage(tmp_path: Path) -> None:
    path = tmp_path / "corpus.sqlite3"
    seed(path, "excluded", tuple(f"first-{i:03}" for i in range(60)))
    seed(path, "selected", ("z-a", "z-b", "z-c"))
    store = SQLitePaperCollections(path)
    collection = store.create(name="Synthetic selection", document_ids=["selected"])
    first = search(path, collection_id=collection.collection_id, limit=1)
    assert first.document_ids == ("selected",) and first.collection_revision == 1
    assert [match.chunk_id for match in first.matches] == ["z-a"]
    assert first.collection_id == collection.collection_id
    with pytest.raises(LiteralSearchError):
        search(path, document_ids=["selected"], cursor=first.next_cursor)
    store.replace(
        collection.collection_id,
        name="Renamed selection",
        document_ids=["selected"],
        expected_revision=1,
    )
    with pytest.raises(LiteralSearchError):
        search(path, collection_id=collection.collection_id, cursor=first.next_cursor)
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("DELETE FROM documents WHERE document_id = 'selected'")
    with pytest.raises(CollectionError) as error:
        search(path, collection_id=collection.collection_id)
    assert error.value.code == "collection_documents_missing"
    with pytest.raises(CollectionError) as error:
        search(path, collection_id="col_" + "0" * 32)
    assert error.value.code == "collection_not_found"


def test_collection_and_passages_share_one_read_only_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "corpus.sqlite3"
    seed(path, "selected", ("a",))
    seed(path, "excluded", ("b",), text="excluded")
    store = SQLitePaperCollections(path)
    collection = store.create(name="Selection", document_ids=["selected"])
    original_connect = sqlite3.connect
    with closing(original_connect(path)) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
    original = SQLitePaperCollections._missing_documents

    def change_after_membership(
        connection: sqlite3.Connection, identifiers: tuple[str, ...]
    ) -> tuple[str, ...]:
        with closing(original_connect(path)) as writer, writer:
            writer.execute("UPDATE chunks SET text = 'changed' WHERE chunk_id = 'a'")
            writer.execute(
                "UPDATE paper_collections SET document_ids = '[\"excluded\"]', revision = 2"
            )
        return original(connection, identifiers)

    connections: list[str] = []
    statements: list[str] = []

    def connect(database: str, *, uri: bool, timeout: float) -> sqlite3.Connection:
        connections.append(database)
        assert uri is True and database.endswith("?mode=ro")
        connection = original_connect(database, uri=uri, timeout=timeout)
        connection.set_trace_callback(statements.append)
        return connection

    monkeypatch.setattr(
        SQLitePaperCollections, "_missing_documents", staticmethod(change_after_membership)
    )
    monkeypatch.setattr(sqlite3, "connect", connect)
    result = search(path, collection_id=collection.collection_id)
    assert result.collection_revision == 1 and result.document_ids == ("selected",)
    assert result.matches[0].excerpt == "Before exact phrase and after."
    assert len(connections) == 1
    assert statements[0] == "BEGIN"
    assert all(
        statement.lstrip().split()[0].upper() in {"BEGIN", "PRAGMA", "SELECT", "WITH"}
        for statement in statements
    )


@pytest.mark.parametrize(
    "change",
    [
        {"query": ""},
        {"query": " \t\n"},
        {"query": 42},
        {"query": b"exact phrase"},
        {"query": "\ud800"},
        {"query": "x" * 201},
        {"document_ids": None},
        {"document_ids": []},
        {"document_ids": "paper"},
        {"document_ids": {"paper"}},
        {"document_ids": ["paper"] * 101},
        {"document_ids": ["\ud800"]},
        {"document_ids": [42]},
        {"collection_id": None},
        {"collection_id": ""},
        {"collection_id": "col_" + "0" * 32, "document_ids": ["paper"]},
        {"cursor": None},
        {"cursor": ""},
        {"cursor": "\ud800"},
        {"cursor": "a" * 4097},
        {"limit": True},
        {"limit": "1"},
        {"limit": 1.0},
        {"limit": 0},
        {"limit": 51},
        {"extra": "not accepted"},
    ],
)
def test_python_inputs_are_strict_even_after_model_copy(
    tmp_path: Path, change: dict[str, object]
) -> None:
    with pytest.raises(ValidationError):
        LiteralSearchRequest.model_validate({"query": "exact phrase", **change})
    forged = LiteralSearchRequest(query="exact phrase").model_copy(update=change)
    with pytest.raises(ValidationError):
        SQLiteLiteralSearch(tmp_path / "must-not-exist.sqlite3").search(forged)
    assert not (tmp_path / "must-not-exist.sqlite3").exists()


@pytest.mark.parametrize("encoding", ["UTF-8", "UTF-16le", "UTF-16be"])
def test_standalone_restart_uri_escaping_unicode_ids_and_encoded_offsets(
    tmp_path: Path, encoding: str
) -> None:
    path = tmp_path / "corpus #?% \u7814.sqlite3"
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(f"PRAGMA encoding = '{encoding}'")
        connection.execute("CREATE TABLE marker (value TEXT)")
    document_id = "\U0001f52c" * 128
    text = "\u03b2\U0001f52c\u7814" * 400 + "exact phrase" + "z" * 1000
    seed(path, document_id, ("\u7814" * 255 + "a", "\u7814" * 255 + "b"), text=text)
    before = path.read_bytes()
    first = search(path, limit=1)
    assert first.matches[0].document_id == document_id
    assert first.matches[0].match_start == 1200
    assert first.matches[0].text_characters == len(text)
    assert first.next_cursor and len(first.next_cursor) <= 4096
    assert search(path, limit=1) == first
    second = search(path, cursor=first.next_cursor)
    assert second.matches[0].chunk_id == "\u7814" * 255 + "b"
    assert second.next_cursor is None and path.read_bytes() == before
    assert set(tmp_path.iterdir()) == {path}


@pytest.mark.parametrize("kind", ["missing", "not-sqlite", "schema"])
def test_missing_or_invalid_storage_is_sanitized_not_an_empty_success(
    tmp_path: Path, kind: str
) -> None:
    path = tmp_path / "PRIVATE_PATH.sqlite3"
    if kind == "not-sqlite":
        path.write_bytes(b"PRIVATE_STORED_CONTENT")
    elif kind == "schema":
        with closing(sqlite3.connect(path)) as connection:
            connection.execute("CREATE TABLE unrelated (value TEXT)")
    before = path.read_bytes() if path.exists() else None
    with pytest.raises(LiteralSearchError) as error:
        search(path)
    assert error.value.code == "search_storage_unavailable"
    assert error.value.status_code == 503 and "PRIVATE" not in str(error.value)
    assert (path.read_bytes() if path.exists() else None) == before


@pytest.mark.parametrize(
    "assignment",
    [
        "chunk_id = NULL",
        "chunk_id = ''",
        "chunk_id = zeroblob(10)",
        "chunk_id = '" + "x" * 257 + "'",
        "document_id = 'orphan'",
        "document_id = zeroblob(10)",
        "title = zeroblob(10)",
        "source = zeroblob(10)",
        "title = CAST(X'80' AS TEXT)",
        "source = CAST(X'80' AS TEXT)",
        "text = zeroblob(10)",
        "text = CAST(X'80' AS TEXT) || 'exact phrase'",
        "text = 'exact phrase' || char(0) || 'suffix'",
    ],
)
def test_invalid_matched_rows_and_lookahead_fail_without_payload(
    tmp_path: Path, assignment: str
) -> None:
    path = tmp_path / "corpus.sqlite3"
    seed(path, chunk_ids=("a", "b"))
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(f"UPDATE chunks SET {assignment} WHERE chunk_id = 'b'")  # noqa: S608
    with pytest.raises(DocumentChunksError, match="invalid or unsupported"):
        search(path, limit=1)


@pytest.mark.parametrize(
    "raw",
    [b"\x80 unrelated", b"\x80" + b"z" * 1000 + b"exact phrase"],
    ids=["nonmatching-corrupt-text", "corruption-before-late-match"],
)
def test_malformed_searched_unicode_cannot_be_empty_success_or_claim_valid_offsets(
    tmp_path: Path, raw: bytes
) -> None:
    path = tmp_path / "corpus.sqlite3"
    seed(path, chunk_ids=("a",))
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("UPDATE chunks SET text = CAST(? AS TEXT)", (raw,))
    with pytest.raises(DocumentChunksError, match="invalid or unsupported"):
        search(path)


def test_unicode_validation_is_scoped_and_never_receives_an_oversized_passage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import storage.literal_search as module

    path = tmp_path / "corpus.sqlite3"
    seed(path, "excluded", ("a",), text="x" * (MAX_TEXT_BYTES + 1))
    seed(path, "selected", ("b",), text="exact phrase")
    original = module._valid_text
    observed = []

    def validate(value: bytes, *, encoding: str) -> bool:
        assert len(value) <= MAX_TEXT_BYTES
        observed.append(len(value))
        return original(value, encoding=encoding)

    monkeypatch.setattr(module, "_valid_text", validate)
    result = search(path, document_ids=["selected"])
    assert result.matches[0].document_id == "selected"
    assert observed and set(observed) == {len(b"exact phrase")}
    with pytest.raises(LiteralSearchError, match="read limit"):
        search(path, document_ids=["excluded"])


def test_oversized_text_and_response_fail_explicitly_with_no_partial_page(tmp_path: Path) -> None:
    path = tmp_path / "corpus.sqlite3"
    seed(path, chunk_ids=("a",), text="z" * (MAX_TEXT_BYTES + 1))
    with pytest.raises(LiteralSearchError) as error:
        search(path)
    assert error.value.code == "search_text_read_limit" and error.value.status_code == 413
    seed(
        path,
        chunk_ids=tuple(f"c-{index:02}" for index in range(50)),
        text="exact phrase" + "\U0001f52c" * 1000,
        title="\U0001f52c" * 301,
        source="\U0001f52c" * 513,
    )
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("DELETE FROM chunks WHERE chunk_id = 'a'")
    with pytest.raises(LiteralSearchError) as error:
        search(path, limit=50)
    assert error.value.code == "search_response_too_large" and error.value.status_code == 413
    result = search(path, limit=1)
    assert len(result.to_json().encode()) <= MAX_RESPONSE_BYTES
    match = result.matches[0]
    assert len(match.title) == 300 and match.title_truncated
    assert len(match.source) == 512 and match.source_truncated
    assert len(match.excerpt) == 800


def test_default_maximum_and_empty_pages_are_bounded(tmp_path: Path) -> None:
    path = tmp_path / "corpus.sqlite3"
    seed(path, chunk_ids=tuple(f"c-{index:02}" for index in range(51)))
    assert len(search(path).matches) == 20
    first = search(path, limit=50)
    assert len(first.matches) == 50
    last = search(path, limit=50, cursor=first.next_cursor)
    assert len(last.matches) == 1 and last.next_cursor is None
    seed(path, "empty", ())
    assert search(path, document_ids=["empty"]).matches == []


def test_read_deadline_is_an_explicit_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import storage.literal_search as module

    path = tmp_path / "corpus.sqlite3"
    seed(path, chunk_ids=tuple(f"c-{i:03}" for i in range(100)))
    monkeypatch.setattr(module, "SEARCH_TIMEOUT_SECONDS", -1)
    with pytest.raises(LiteralSearchError) as error:
        search(path, query="absent")
    assert error.value.code == "search_timeout" and error.value.status_code == 504
