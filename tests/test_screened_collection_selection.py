"""Resolve only current human inclusions, without repairing or widening broken selections."""

import json
import sqlite3
from pathlib import Path

import pytest
from pydantic import ValidationError

from storage.paper_collections import CollectionError, PaperCollection
from storage.paper_screening import ScreeningQueue, ScreeningSubmission
from tests.test_paper_screening_storage import make_store, submission


def test_selection_uses_current_labels_in_collection_order_and_detaches(tmp_path: Path) -> None:
    store, collection_id, ids = make_store(tmp_path, count=6)
    store.submit(collection_id, ids[5], submission())
    store.collections.replace(
        collection_id, name="Revised review", document_ids=ids[::-1], expected_revision=1
    )
    for identifier, decision in zip(
        ids[:4], ("include", "exclude", "include", "unsure"), strict=True
    ):
        store.submit(
            collection_id,
            identifier,
            ScreeningSubmission(
                collection_revision=2,
                expected_decision_revision=0,
                decision=decision,
                reason="Human screening of a synthetic note.",
            ),
        )
    for revision in (1, 2):
        store.submit(
            collection_id,
            ids[2],
            submission(revision=revision).model_copy(update={"collection_revision": 2}),
        )
    queue = store.list_queue(collection_id, collection_revision=2)
    assert queue.counts.model_dump() == {
        "include": 2,
        "exclude": 1,
        "unsure": 1,
        "unscreened": 1,
        "stale": 1,
    }
    selected = store.select_included(collection_id, collection_revision=2)
    assert isinstance(selected, tuple)
    assert selected == (ids[2], ids[0])
    assert selected == queue.included_document_ids
    # The decision revision is independent of the collection revision.
    store.submit(
        collection_id,
        ids[2],
        submission("exclude", revision=3).model_copy(update={"collection_revision": 2}),
    )
    assert store.select_included(collection_id, collection_revision=2) == (ids[0],)
    store.collections.delete(collection_id, expected_revision=2)
    assert selected == (ids[2], ids[0])


def test_selection_is_complete_not_a_queue_page_or_serialized_review_export(tmp_path: Path) -> None:
    store, collection_id, ids = make_store(tmp_path, count=100)
    for identifier in ids:
        store.submit(
            collection_id, identifier, submission().model_copy(update={"reason": "\x01" * 1000})
        )
    with pytest.raises(CollectionError, match="byte limit"):
        store.list_queue(collection_id, collection_revision=1, limit=100)
    assert store.select_included(collection_id, collection_revision=1) == tuple(ids)


@pytest.mark.parametrize("state", ["unscreened", "exclude", "unsure", "stale"])
def test_no_current_includes_is_an_explicit_conflict(tmp_path: Path, state: str) -> None:
    store, collection_id, ids = make_store(tmp_path)
    revision = 1
    if state != "unscreened":
        store.submit(collection_id, ids[0], submission("include" if state == "stale" else state))
    if state == "stale":
        store.collections.replace(
            collection_id, name="New criteria", document_ids=ids, expected_revision=1
        )
        revision = 2
    before = (tmp_path / "store.sqlite3").read_bytes()
    with pytest.raises(CollectionError) as error:
        store.select_included(collection_id, collection_revision=revision)
    assert (error.value.code, error.value.status_code) == ("screened_collection_empty", 409)
    assert (tmp_path / "store.sqlite3").read_bytes() == before


@pytest.mark.parametrize("revision", [None, True, False, 0, -1, 1.0, "1", 2**63, [], {}])
def test_python_selection_requires_a_strict_revision(tmp_path: Path, revision: object) -> None:
    store, collection_id, ids = make_store(tmp_path)
    store.submit(collection_id, ids[0], submission())
    with pytest.raises(ValidationError):
        store.select_included(collection_id, collection_revision=revision)  # type: ignore[arg-type]


@pytest.mark.parametrize("identifier", [None, True, 1, "", [], {}, "not-a-collection"])
def test_python_selection_requires_an_exact_collection_id(
    tmp_path: Path, identifier: object
) -> None:
    store, _, _ = make_store(tmp_path)
    with pytest.raises(ValidationError):
        store.select_included(identifier, collection_revision=1)  # type: ignore[arg-type]


def test_stale_revision_unknown_collection_and_storage_failure(tmp_path: Path) -> None:
    store, collection_id, ids = make_store(tmp_path)
    store.submit(collection_id, ids[0], submission())
    store.collections.replace(collection_id, name="Changed", document_ids=ids, expected_revision=1)
    with pytest.raises(CollectionError) as stale:
        store.select_included(collection_id, collection_revision=1)
    assert stale.value.code == "collection_revision_conflict"
    with pytest.raises(CollectionError) as missing:
        store.select_included("col_" + "0" * 32, collection_revision=1)
    assert (missing.value.code, missing.value.status_code) == ("collection_not_found", 404)
    with sqlite3.connect(tmp_path / "store.sqlite3") as connection:
        connection.execute("DROP TABLE paper_screening_decisions")
    with pytest.raises(CollectionError) as unavailable:
        store.select_included(collection_id, collection_revision=2)
    assert unavailable.value.status_code == 503


@pytest.mark.parametrize("member", [0, 1, 2])
def test_missing_included_excluded_or_unscreened_member_fails(tmp_path: Path, member: int) -> None:
    store, collection_id, ids = make_store(tmp_path, count=3)
    store.submit(collection_id, ids[0], submission())
    store.submit(collection_id, ids[1], submission("exclude"))
    with sqlite3.connect(tmp_path / "store.sqlite3") as connection:
        connection.execute("DELETE FROM documents WHERE document_id = ?", (ids[member],))
    with pytest.raises(CollectionError) as error:
        store.select_included(collection_id, collection_revision=1)
    assert error.value.code == "collection_documents_missing"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("decision", "automatic"),
        ("reason", ""),
        ("reason", "\ud800"),
        ("revision", True),
        ("collection_revision", 2),
        ("collection_id", "col_" + "0" * 32),
        ("document_id", "foreign"),
        ("updated_at", "not-a-time"),
    ],
)
def test_malformed_excluded_review_invalidates_entire_selection(
    tmp_path: Path, field: str, value: object
) -> None:
    store, collection_id, ids = make_store(tmp_path)
    store.submit(collection_id, ids[0], submission())
    excluded = store.submit(collection_id, ids[1], submission("exclude"))
    record = {**excluded.model_dump(mode="json"), field: value}
    with sqlite3.connect(tmp_path / "store.sqlite3") as connection:
        connection.execute(
            "UPDATE paper_screening_decisions SET record = ? WHERE document_id = ?",
            (json.dumps(record), ids[1]),
        )
    with pytest.raises(CollectionError) as error:
        store.select_included(collection_id, collection_revision=1)
    assert error.value.code == "invalid_screening_record"


@pytest.mark.parametrize("record", ["not-json", b"blob", "\x00", "x" * 16385])
def test_invalid_review_envelope_is_not_skipped(tmp_path: Path, record: object) -> None:
    store, collection_id, ids = make_store(tmp_path)
    store.submit(collection_id, ids[0], submission())
    with sqlite3.connect(tmp_path / "store.sqlite3") as connection:
        connection.execute("UPDATE paper_screening_decisions SET record = ?", (record,))
    with pytest.raises(CollectionError) as error:
        store.select_included(collection_id, collection_revision=1)
    assert error.value.code == "invalid_screening_record"


@pytest.mark.parametrize("membership", ["[]", '["missing"]', '["paper-0", "paper-0"]', "not-json"])
def test_broken_collection_cannot_produce_an_inclusion(tmp_path: Path, membership: str) -> None:
    store, collection_id, ids = make_store(tmp_path)
    store.submit(collection_id, ids[0], submission())
    with sqlite3.connect(tmp_path / "store.sqlite3") as connection:
        connection.execute("UPDATE paper_collections SET document_ids = ?", (membership,))
    with pytest.raises(CollectionError) as error:
        store.select_included(collection_id, collection_revision=1)
    assert error.value.status_code == 409


def test_membership_and_reviews_share_one_read_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, collection_id, ids = make_store(tmp_path)
    store.submit(collection_id, ids[0], submission())
    with sqlite3.connect(tmp_path / "store.sqlite3") as connection:
        assert connection.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
    original = store._snapshot

    def concurrent_edit(
        connection: sqlite3.Connection, collection: PaperCollection
    ) -> ScreeningQueue:
        store.submit(collection_id, ids[0], submission("exclude", revision=1))
        store.collections.replace(
            collection_id, name="Changed", document_ids=[ids[1]], expected_revision=1
        )
        return original(connection, collection)

    with monkeypatch.context() as patch:
        patch.setattr(store, "_snapshot", concurrent_edit)
        selected = store.select_included(collection_id, collection_revision=1)
    assert selected == (ids[0],)
    with pytest.raises(CollectionError) as changed:
        store.select_included(collection_id, collection_revision=1)
    assert changed.value.code == "collection_revision_conflict"
    with pytest.raises(CollectionError) as empty:
        store.select_included(collection_id, collection_revision=2)
    assert empty.value.code == "screened_collection_empty"
