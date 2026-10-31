"""Native deltas reserve the complete old selection, not only surviving members."""

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

import pytest

from storage import paper_collections
from storage.paper_collections import MAX_REVISION, CollectionError, SQLitePaperCollections
from tests.test_paper_collections_store import store as store


def test_removed_member_is_validated_under_the_same_write_reservation(
    store: SQLitePaperCollections, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved = store.create(name="Before", document_ids=["a", "b"])
    changes = paper_collections.CollectionEdit(
        name="After", expected_revision=1, remove_document_ids=("a",)
    )
    entered, release = threading.Event(), threading.Event()
    original = store._missing_documents

    def hold(connection: sqlite3.Connection, ids: tuple[str, ...]) -> tuple[str, ...]:
        missing = original(connection, ids)
        if ids == ("a", "b"):
            entered.set()
            if not release.wait(5):
                raise AssertionError("Test did not release the reserved edit.")
        return missing

    monkeypatch.setattr(store, "_missing_documents", hold)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(store.edit, saved.collection_id, changes=changes)
        try:
            assert entered.wait(3)
            with (
                closing(sqlite3.connect(tmp_path / "corpus.sqlite3", timeout=0)) as writer,
                writer,
                pytest.raises(sqlite3.OperationalError, match="locked"),
            ):
                writer.execute("DELETE FROM documents WHERE document_id = 'a'")
        finally:
            release.set()
        result = pending.result(3)
    assert result.document_ids == ("b",) and result.revision == 2


def test_preview_is_read_only_and_exhaustion_and_corruption_fail_closed(
    store: SQLitePaperCollections, tmp_path: Path
) -> None:
    saved = store.create(name="Before", document_ids=["a"])
    path = tmp_path / "corpus.sqlite3"
    changes = paper_collections.CollectionEdit(
        name="After", expected_revision=1, add_document_ids=("b",)
    )
    before = path.read_bytes()
    current, candidate = store.preview_edit(saved.collection_id, changes=changes)
    assert current == saved and candidate.document_ids == ("a", "b")
    assert candidate.name == "After"
    assert path.read_bytes() == before
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("UPDATE paper_collections SET revision = ?", (MAX_REVISION,))
    exhausted = changes.model_copy(update={"expected_revision": MAX_REVISION})
    for operation in (store.edit, store.preview_edit):
        with pytest.raises(CollectionError) as error:
            operation(saved.collection_id, changes=exhausted)
        assert error.value.code == "collection_revision_exhausted"
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("UPDATE paper_collections SET document_ids = '[]', revision = 1")
    with pytest.raises(CollectionError) as error:
        store.edit(saved.collection_id, changes=changes)
    assert error.value.code == "invalid_collection_record"
