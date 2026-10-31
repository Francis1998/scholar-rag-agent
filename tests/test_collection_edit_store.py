"""Native deltas reserve the complete old selection, not only surviving members."""

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

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


@pytest.mark.parametrize("operation", ["preview_edit", "edit"])
@pytest.mark.parametrize("factory", ["copy", "construct"])
@pytest.mark.parametrize(
    "update",
    [
        pytest.param({"remove_document_ids": ("a", "a")}, id="duplicate-removals"),
        pytest.param({"add_document_ids": ("b", "b")}, id="duplicate-additions"),
        pytest.param({"add_document_ids": ("a",), "remove_document_ids": ("a",)}, id="overlap"),
        pytest.param({"add_document_ids": {"b"}}, id="unordered-additions"),
        pytest.param({"remove_document_ids": {"a"}}, id="unordered-removals"),
        pytest.param({"add_document_ids": "b"}, id="string-not-sequence"),
        pytest.param({"remove_document_ids": (7,)}, id="nonstring-id"),
        pytest.param({"add_document_ids": (" b",)}, id="nonexact-id"),
        pytest.param({"add_document_ids": ("\ud800",)}, id="invalid-unicode"),
        pytest.param({"add_document_ids": ("b" * 129,)}, id="oversized-id"),
        pytest.param(
            {"add_document_ids": tuple(f"paper-{index}" for index in range(101))},
            id="too-many-additions",
        ),
        pytest.param(
            {"remove_document_ids": tuple(f"paper-{index}" for index in range(101))},
            id="too-many-removals",
        ),
        pytest.param({"name": 7}, id="nonstring-name"),
        pytest.param({"name": "x" * 121}, id="oversized-name"),
        pytest.param({"name": "invalid\0name"}, id="control-name"),
        pytest.param({"expected_revision": True}, id="boolean-revision"),
        pytest.param({"expected_revision": "1"}, id="string-revision"),
        pytest.param({"expected_revision": 0}, id="zero-revision"),
        pytest.param({"expected_revision": MAX_REVISION + 1}, id="oversized-revision"),
    ],
)
def test_python_edit_revalidates_bypassed_models_before_opening_storage(
    store: SQLitePaperCollections,
    tmp_path: Path,
    operation: str,
    factory: str,
    update: dict[str, object],
) -> None:
    saved = store.create(name="Before", document_ids=["a"])
    valid = paper_collections.CollectionEdit(
        name="After", expected_revision=1, add_document_ids=("b",)
    )
    changes = (
        valid.model_copy(update=update)
        if factory == "copy"
        else paper_collections.CollectionEdit.model_construct(**{**valid.model_dump(), **update})
    )
    path = tmp_path / "corpus.sqlite3"
    before = path.read_bytes()
    with patch.object(store, "_connection", wraps=store._connection) as connection:
        with pytest.raises(ValidationError):
            getattr(store, operation)(saved.collection_id, changes=changes)
        connection.assert_not_called()
    assert path.read_bytes() == before
    assert store.get(saved.collection_id) == saved


@pytest.mark.parametrize("factory", ["copy", "construct"])
def test_valid_detached_copied_and_constructed_edits_still_preview_and_save(
    store: SQLitePaperCollections, tmp_path: Path, factory: str
) -> None:
    saved = store.create(name="Before", document_ids=["a"])
    valid = paper_collections.CollectionEdit(name="After", expected_revision=1)
    added, removed = ["b"], ["a"]
    update: dict[str, object] = {
        "name": "  After  ",
        "add_document_ids": added,
        "remove_document_ids": removed,
    }
    changes = (
        valid.model_copy(update=update)
        if factory == "copy"
        else paper_collections.CollectionEdit.model_construct(**{**valid.model_dump(), **update})
    )
    path = tmp_path / "corpus.sqlite3"
    before = path.read_bytes()
    current, proposed = store.preview_edit(saved.collection_id, changes=changes)
    assert current == saved and proposed.name == "After" and proposed.document_ids == ("b",)
    assert path.read_bytes() == before
    result = store.edit(saved.collection_id, changes=changes)
    added.clear()
    removed.clear()
    assert changes.name == "  After  " and valid.name == "After"
    assert result.name == "After" and result.document_ids == ("b",) and result.revision == 2
    assert SQLitePaperCollections(path).get(saved.collection_id) == result
