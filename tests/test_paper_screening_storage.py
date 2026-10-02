"""Screening metadata bounds, corruption detection and real transaction concurrency."""

import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from pydantic import ValidationError

from retrieval.models import Document
from storage import paper_screening
from storage.document_store import SQLiteDocumentStore
from storage.paper_collections import MAX_REVISION, CollectionError, PaperCollection
from storage.paper_screening import (
    ScreeningReview,
    ScreeningSubmission,
    SQLitePaperScreening,
)


def make_store(tmp_path: Path, count: int = 2) -> tuple[SQLitePaperScreening, str, list[str]]:
    path = tmp_path / "store.sqlite3"
    ids = [f"paper-{number}" for number in range(count)]
    SQLiteDocumentStore(path).add_documents(
        [Document(document_id=i, title=i, text="Synthetic only.", source="synthetic") for i in ids],
        [],
    )
    store = SQLitePaperScreening(path)
    saved = store.collections.create(name="Synthetic review", document_ids=ids)
    return store, saved.collection_id, ids


def submission(decision: str = "include", revision: int = 0) -> ScreeningSubmission:
    return ScreeningSubmission.model_validate(
        {
            "collection_revision": 1,
            "expected_decision_revision": revision,
            "decision": decision,
            "reason": "Human opinion about a synthetic note.",
        }
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", None),
        ("schema_version", "2.0"),
        ("revision", True),
        ("revision", 0),
        ("collection_revision", 2),
        ("document_id", "foreign"),
        ("collection_id", "col_" + "0" * 32),
        ("reason", "\ud800"),
        ("updated_at", "not-a-time"),
    ],
)
def test_invalid_saved_review_never_becomes_a_selection(
    tmp_path: Path, field: str, value: object
) -> None:
    store, collection_id, ids = make_store(tmp_path)
    saved = store.submit(collection_id, ids[0], submission())
    record = saved.model_dump(mode="json")
    if value is None:
        record.pop(field)
    else:
        record[field] = value
    with sqlite3.connect(tmp_path / "store.sqlite3") as connection:
        connection.execute("UPDATE paper_screening_decisions SET record = ?", (json.dumps(record),))
    with pytest.raises(CollectionError) as read_error:
        store.list_queue(collection_id, collection_revision=1)
    assert read_error.value.code == "invalid_screening_record"
    with pytest.raises(CollectionError) as write_error:
        store.submit(collection_id, ids[1], submission())
    assert write_error.value.code == "invalid_screening_record"


@pytest.mark.parametrize(
    "record",
    ["{invalid-json", "x" * (paper_screening.MAX_RECORD_CHARACTERS + 1), b"blob", "\x00"],
)
def test_invalid_saved_envelope_is_bounded(tmp_path: Path, record: object) -> None:
    store, collection_id, ids = make_store(tmp_path)
    store.submit(collection_id, ids[0], submission())
    with sqlite3.connect(tmp_path / "store.sqlite3") as connection:
        connection.execute("UPDATE paper_screening_decisions SET record = ?", (record,))
    with pytest.raises(CollectionError) as error:
        store.list_queue(collection_id, collection_revision=1)
    assert error.value.code == "invalid_screening_record"


def test_concurrent_decisions_cannot_lose_an_update(tmp_path: Path) -> None:
    store, collection_id, ids = make_store(tmp_path)
    start = threading.Barrier(2)

    def write(decision: str) -> str:
        start.wait(timeout=5)
        try:
            store.submit(collection_id, ids[0], submission(decision))
        except CollectionError as exc:
            return exc.code
        return "saved"

    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(write, ["include", "exclude"]))
    assert sorted(results) == ["saved", "screening_revision_conflict"]
    result = store.list_queue(collection_id, collection_revision=1)
    assert result.items[0].review is not None
    assert result.items[0].review.revision == 1


def test_membership_is_locked_through_the_decision_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, collection_id, ids = make_store(tmp_path)
    original = store._reviews

    def read(
        connection: sqlite3.Connection, collection: PaperCollection
    ) -> dict[str, ScreeningReview]:
        with (
            sqlite3.connect(tmp_path / "store.sqlite3", timeout=0) as other,
            pytest.raises(sqlite3.OperationalError, match="locked"),
        ):
            other.execute("DELETE FROM documents WHERE document_id = ?", (ids[0],))
        return original(connection, collection)

    monkeypatch.setattr(store, "_reviews", read)
    assert store.submit(collection_id, ids[0], submission()).revision == 1


@pytest.mark.parametrize("limit", [0, 101, True, 1.5, "1"])
def test_direct_python_limits_do_not_coerce(tmp_path: Path, limit: object) -> None:
    store, collection_id, _ = make_store(tmp_path)
    with pytest.raises(ValidationError):
        store.list_queue(collection_id, collection_revision=1, limit=limit)  # type: ignore[arg-type]


def test_maximum_collection_and_real_response_bound(tmp_path: Path) -> None:
    store, collection_id, ids = make_store(tmp_path, count=100)
    payload = submission().model_copy(update={"reason": "\x01" * 1000})
    for identifier in ids:
        store.submit(collection_id, identifier, payload)
    with pytest.raises(CollectionError) as error:
        store.list_queue(collection_id, collection_revision=1, limit=100)
    assert error.value.code == "screening_response_too_large"
    page = store.list_queue(collection_id, collection_revision=1, limit=20)
    assert len(page.items) == 20
    assert page.included_document_ids == tuple(ids)
    assert page.counts.include == 100
    assert len(page.model_dump_json().encode("utf-8")) <= paper_screening.MAX_RESPONSE_BYTES


def test_exhausted_revision_cannot_overflow_sqlite(tmp_path: Path) -> None:
    store, collection_id, ids = make_store(tmp_path)
    saved = store.submit(collection_id, ids[0], submission())
    with sqlite3.connect(tmp_path / "store.sqlite3") as connection:
        connection.execute(
            "UPDATE paper_screening_decisions SET record = ?",
            (saved.model_copy(update={"revision": MAX_REVISION}).model_dump_json(),),
        )
    with pytest.raises(CollectionError) as error:
        store.submit(collection_id, ids[0], submission(revision=MAX_REVISION))
    assert error.value.code == "screening_revision_exhausted"


def test_direct_unvalidated_submission_is_checked_again(tmp_path: Path) -> None:
    store, collection_id, ids = make_store(tmp_path)
    with pytest.raises(ValidationError):
        store.submit(collection_id, ids[0], submission().model_copy(update={"reason": ""}))
    assert store.list_queue(collection_id, collection_revision=1).counts.unscreened == 2
