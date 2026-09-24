"""Standalone selection validation, transaction isolation, and cleanup contracts."""

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from pydantic import ValidationError

from retrieval.models import Document
from storage.document_store import SQLiteDocumentStore
from storage.paper_collections import MAX_REVISION, CollectionError, SQLitePaperCollections


@pytest.fixture
def store(tmp_path: Path) -> SQLitePaperCollections:
    path = tmp_path / "corpus.sqlite3"
    SQLiteDocumentStore(path).add_documents(
        [
            Document(document_id=identifier, title="Synthetic", text="GraphRAG", source="synthetic")
            for identifier in ("a", "b")
        ],
        [],
    )
    return SQLitePaperCollections(path)


def test_python_immutable_normalization_and_restart(
    store: SQLitePaperCollections, tmp_path: Path
) -> None:
    supplied = [" a ", "b", "a"]
    collection = store.create(name="  Methods  ", document_ids=supplied)
    assert supplied == [" a ", "b", "a"]
    supplied.clear()
    assert collection.name == "Methods" and collection.document_ids == ("a", "b")
    with pytest.raises(ValidationError, match="frozen"):
        collection.name = "Mutated"
    snapshot = store.resolve(collection.collection_id)
    replaced = store.replace(
        collection.collection_id, name="Changed", document_ids=["b"], expected_revision=1
    )
    restarted = SQLitePaperCollections(tmp_path / "corpus.sqlite3")
    assert restarted.get(collection.collection_id) == replaced
    assert snapshot == ("a", "b")
    assert restarted.resolve(collection.collection_id) == ("b",)
    store.delete(collection.collection_id, expected_revision=2)
    assert snapshot == ("a", "b")


@pytest.mark.parametrize("limit", [0, 101, True, 1.5, "2"])
def test_python_pages_require_strict_bounded_integer(
    store: SQLitePaperCollections, limit: object
) -> None:
    with pytest.raises(ValidationError):
        store.list_collections(limit=limit)  # type: ignore[arg-type]


@pytest.mark.parametrize("ids", [None, [], set("a"), {"a": 1}, ["a"] * 101])
def test_python_invalid_membership_cannot_mean_unscoped(
    store: SQLitePaperCollections, ids: object
) -> None:
    with pytest.raises(ValidationError):
        store.create(name="Invalid", document_ids=ids)  # type: ignore[arg-type]
    assert store.list_collections().collections == []


@pytest.mark.parametrize("operation", ["create", "replace"])
def test_document_validation_and_write_share_a_reserved_transaction(
    store: SQLitePaperCollections, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    saved = store.create(name="First", document_ids=["a"])
    entered, release = threading.Event(), threading.Event()
    original = store._validate_documents

    def hold(connection: sqlite3.Connection, ids: tuple[str, ...]) -> None:
        original(connection, ids)
        entered.set()
        if not release.wait(timeout=5):
            raise AssertionError("Test did not release the reserved transaction.")

    monkeypatch.setattr(store, "_validate_documents", hold)
    with ThreadPoolExecutor(max_workers=1) as pool:
        if operation == "create":
            pending = pool.submit(store.create, name="New", document_ids=["b"])
        else:
            pending = pool.submit(
                store.replace,
                saved.collection_id,
                name="Updated",
                document_ids=["b"],
                expected_revision=1,
            )
        try:
            assert entered.wait(timeout=3)
            with (
                sqlite3.connect(tmp_path / "corpus.sqlite3", timeout=0) as other,
                pytest.raises(sqlite3.OperationalError, match="locked"),
            ):
                other.execute("DELETE FROM documents WHERE document_id = 'b'")
        finally:
            release.set()
        result = pending.result(timeout=3)
    assert store.resolve(result.collection_id) == ("b",)


def test_resolution_checks_members_in_one_read_snapshot(
    store: SQLitePaperCollections, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved = store.create(name="Snapshot", document_ids=["a"])
    path = tmp_path / "corpus.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
    original = store._missing_documents

    def change_after_metadata_read(
        connection: sqlite3.Connection, ids: tuple[str, ...]
    ) -> tuple[str, ...]:
        with sqlite3.connect(path) as writer:
            writer.execute("DELETE FROM documents WHERE document_id = 'a'")
            writer.execute("UPDATE paper_collections SET document_ids = '[\"b\"]', revision = 2")
        return original(connection, ids)

    monkeypatch.setattr(store, "_missing_documents", change_after_metadata_read)
    assert store.resolve(saved.collection_id) == ("a",)
    assert store.get(saved.collection_id).document_ids == ("b",)


def test_schema_initialization_is_additive_and_does_not_backfill(tmp_path: Path) -> None:
    path = tmp_path / "corpus.sqlite3"
    SQLiteDocumentStore(path).add_documents(
        [Document(document_id="a", title="Original", text="private-body", source="private-source")],
        [],
    )
    with sqlite3.connect(path) as connection:
        before = connection.execute("SELECT * FROM documents").fetchall()
    store = SQLitePaperCollections(path)
    assert store.list_collections().collections == []
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT * FROM documents").fetchall() == before
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert tables == {"documents", "chunks", "paper_collections"}
    before_bytes = path.read_bytes()
    SQLitePaperCollections(path)
    assert path.read_bytes() == before_bytes


def test_list_validates_corrupt_lookahead_and_delete_can_remove_it(
    store: SQLitePaperCollections, tmp_path: Path
) -> None:
    records = sorted(
        [store.create(name=name, document_ids=["a"]) for name in ("A", "B")],
        key=lambda record: record.collection_id,
    )
    with sqlite3.connect(tmp_path / "corpus.sqlite3") as connection:
        connection.execute(
            "UPDATE paper_collections SET document_ids = '[]' WHERE collection_id = ?",
            (records[1].collection_id,),
        )
    with pytest.raises(CollectionError) as error:
        store.list_collections(limit=1)
    assert error.value.code == "invalid_collection_record"
    store.delete(records[1].collection_id, expected_revision=1)
    assert len(store.list_collections().collections) == 1


def test_revision_exhaustion_is_explicit_and_deletion_still_works(
    store: SQLitePaperCollections, tmp_path: Path
) -> None:
    saved = store.create(name="Maximum", document_ids=["a"])
    with sqlite3.connect(tmp_path / "corpus.sqlite3") as connection:
        connection.execute("UPDATE paper_collections SET revision = ?", (MAX_REVISION,))
    with pytest.raises(CollectionError) as error:
        store.replace(
            saved.collection_id,
            name="Changed",
            document_ids=["b"],
            expected_revision=MAX_REVISION,
        )
    assert error.value.code == "collection_revision_exhausted"
    assert store.get(saved.collection_id).document_ids == ("a",)
    store.delete(saved.collection_id, expected_revision=MAX_REVISION)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("name", " not-normalized "),
        ("name", "private\x00sentinel"),
        ("name", b"private-sentinel"),
        ("name", "x" * 121),
        ("revision", "invalid"),
        ("document_ids", b"private-sentinel"),
    ],
)
def test_invalid_saved_metadata_has_safe_diagnostics(
    store: SQLitePaperCollections, tmp_path: Path, column: str, value: object
) -> None:
    saved = store.create(name="Valid", document_ids=["a"])
    statements = {
        "name": "UPDATE paper_collections SET name = ?",
        "revision": "UPDATE paper_collections SET revision = ?",
        "document_ids": "UPDATE paper_collections SET document_ids = ?",
    }
    with sqlite3.connect(tmp_path / "corpus.sqlite3") as connection:
        connection.execute(statements[column], (value,))
    with pytest.raises(CollectionError) as error:
        store.get(saved.collection_id)
    assert error.value.code == "invalid_collection_record"
    assert "sentinel" not in str(error.value)


def test_missing_database_is_not_recreated_by_operations(
    store: SQLitePaperCollections, tmp_path: Path
) -> None:
    path = tmp_path / "corpus.sqlite3"
    path.rename(tmp_path / "moved.sqlite3")
    with pytest.raises(CollectionError) as error:
        store.list_collections()
    assert error.value.code == "collection_storage_error"
    assert not path.exists()


def test_maximum_astral_membership_round_trips_and_can_be_replaced(tmp_path: Path) -> None:
    path = tmp_path / "astral.sqlite3"
    identifiers = [f"{index:02d}" + "\U0001f9ea" * 126 for index in range(100)]
    assert all(len(identifier) == 128 for identifier in identifiers)
    SQLiteDocumentStore(path).add_documents(
        [
            Document(document_id=identifier, title="Synthetic", text="GraphRAG", source="synthetic")
            for identifier in identifiers
        ],
        [],
    )
    store = SQLitePaperCollections(path)
    saved = store.create(name="Maximum Unicode", document_ids=identifiers)
    assert store.get(saved.collection_id) == saved
    assert store.list_collections().collections[0].document_count == 100
    assert store.resolve(saved.collection_id) == tuple(identifiers)
    restarted = SQLitePaperCollections(path)
    assert restarted.get(saved.collection_id) == saved
    revised = restarted.replace(
        saved.collection_id,
        name="Reversed Unicode",
        document_ids=list(reversed(identifiers)),
        expected_revision=1,
    )
    assert restarted.get(saved.collection_id) == revised
    assert restarted.resolve(saved.collection_id) == tuple(reversed(identifiers))
    assert SQLitePaperCollections(path).get(saved.collection_id) == revised


def test_embedded_nul_identity_is_not_a_truncated_pagination_alias(
    store: SQLitePaperCollections, tmp_path: Path
) -> None:
    saved = store.create(name="Corrupt identity", document_ids=["a"])
    with sqlite3.connect(tmp_path / "corpus.sqlite3") as connection:
        connection.execute(
            "UPDATE paper_collections SET collection_id = ?",
            (saved.collection_id + "\x00suffix",),
        )
    with pytest.raises(CollectionError) as error:
        store.list_collections(limit=1)
    assert error.value.code == "invalid_collection_record"
    with pytest.raises(CollectionError) as missing:
        store.get(saved.collection_id)
    assert missing.value.code == "collection_not_found"
