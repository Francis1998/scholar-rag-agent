"""Append-only human decisions preserve the existing latest-only screening contract."""

import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock

import pytest

from storage import paper_screening
from storage.paper_collections import MAX_REVISION, CollectionError, PaperCollection
from storage.paper_screening import ScreeningReview, SQLitePaperScreening
from tests.test_paper_screening_storage import make_store, submission


def test_successful_submissions_append_exact_records(tmp_path: Path) -> None:
    store, collection_id, ids = make_store(tmp_path)
    saved = [
        store.submit(collection_id, ids[0], submission(decision, revision))
        for revision, decision in enumerate(("include", "exclude", "unsure"))
    ]
    with closing(sqlite3.connect(tmp_path / "store.sqlite3")) as connection:
        assert connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'paper_screening_history'"
        ).fetchone(), "Successful submissions need a separate additive history table."
        records = connection.execute(
            "SELECT record FROM paper_screening_history ORDER BY revision"
        ).fetchall()
    assert [row[0] for row in records] == [review.model_dump_json() for review in saved]
    page = store.list_history(collection_id, ids[0], collection_revision=1)
    assert page.items == list(reversed(saved))
    assert page.latest_review == saved[-1]
    assert page.current_status == "unsure"
    assert page.first_retained_revision == 1
    assert page.first_retained_at == saved[0].updated_at
    assert page.unretained_decision_revisions == 0
    assert page.next_cursor is None
    assert store.list_queue(collection_id, collection_revision=1).items[0].review == saved[-1]


@pytest.mark.parametrize("existing", [False, True])
def test_failed_history_insert_rolls_back_latest_decision(tmp_path: Path, existing: bool) -> None:
    store, collection_id, ids = make_store(tmp_path)
    if existing:
        store.submit(collection_id, ids[0], submission())
    before = store.list_queue(collection_id, collection_revision=1)
    with closing(sqlite3.connect(tmp_path / "store.sqlite3")) as connection, connection:
        connection.execute(
            "CREATE TRIGGER reject_history BEFORE INSERT ON paper_screening_history "
            "BEGIN SELECT RAISE(ABORT, 'PRIVATE_FAILURE'); END"
        )
    with pytest.raises(CollectionError) as error:
        store.submit(collection_id, ids[0], submission("exclude", int(existing)))
    assert error.value.status_code == 503
    assert "PRIVATE_" not in str(error.value)
    assert store.list_queue(collection_id, collection_revision=1) == before
    assert len(store.list_history(collection_id, ids[0], collection_revision=1).items) == int(
        existing
    )


def test_rejected_retry_does_not_append_but_new_revision_does(tmp_path: Path) -> None:
    store, collection_id, ids = make_store(tmp_path)
    store.submit(collection_id, ids[0], submission())
    before = store.list_history(collection_id, ids[0], collection_revision=1)
    for payload in (submission(), submission("exclude")):
        with pytest.raises(CollectionError, match="Decision changed"):
            store.submit(collection_id, ids[0], payload)
    assert store.list_history(collection_id, ids[0], collection_revision=1) == before
    saved = store.submit(collection_id, ids[0], submission(revision=1))
    assert saved.revision == 2
    assert len(store.list_history(collection_id, ids[0], collection_revision=1).items) == 2


def test_legacy_latest_is_not_backfilled_and_boundary_survives_restart(tmp_path: Path) -> None:
    store, collection_id, ids = make_store(tmp_path)
    legacy = store.submit(collection_id, ids[0], submission())
    with closing(sqlite3.connect(tmp_path / "store.sqlite3")) as connection, connection:
        connection.execute("DROP TABLE paper_screening_history")
    store = SQLitePaperScreening(tmp_path / "store.sqlite3")
    page = store.list_history(collection_id, ids[0], collection_revision=1)
    assert page.items == []
    assert page.latest_review == legacy
    assert page.first_retained_revision is None and page.first_retained_at is None
    assert page.unretained_decision_revisions == 1
    assert page.current_status == "include"
    saved = store.submit(collection_id, ids[0], submission("exclude", 1))
    restarted = SQLitePaperScreening(tmp_path / "store.sqlite3")
    page = restarted.list_history(collection_id, ids[0], collection_revision=1)
    assert page.items == [saved]
    assert page.first_retained_revision == 2
    assert page.first_retained_at == saved.updated_at
    assert page.unretained_decision_revisions == 1
    assert page.latest_review == saved
    empty = restarted.list_history(collection_id, ids[1], collection_revision=1)
    assert empty.current_status == "unscreened"
    assert empty.latest_review is None and empty.items == []
    assert empty.unretained_decision_revisions == 0


def test_collection_changes_removal_readdition_and_deletion(tmp_path: Path) -> None:
    store, collection_id, ids = make_store(tmp_path)
    saved = store.submit(collection_id, ids[0], submission())
    store.collections.replace(
        collection_id, name="Removed", document_ids=ids[1:], expected_revision=1
    )
    with pytest.raises(CollectionError) as removed:
        store.list_history(collection_id, ids[0], collection_revision=2)
    assert removed.value.code == "screening_document_not_member"
    with pytest.raises(CollectionError) as stale:
        store.list_history(collection_id, ids[0], collection_revision=1)
    assert stale.value.code == "collection_revision_conflict"
    with pytest.raises(CollectionError):
        store.submit(collection_id, ids[0], submission("exclude", 1))
    store.collections.replace(collection_id, name="Readded", document_ids=ids, expected_revision=2)
    page = store.list_history(collection_id, ids[0], collection_revision=3)
    assert page.items == [saved] and page.current_status == "stale"
    assert store.list_queue(collection_id, collection_revision=3).included_document_ids == ()
    store.submit(
        collection_id,
        ids[0],
        submission("unsure", 1).model_copy(update={"collection_revision": 3}),
    )
    page = store.list_history(collection_id, ids[0], collection_revision=3)
    assert [record.collection_revision for record in page.items] == [3, 1]
    assert page.current_status == "unsure"
    store.collections.delete(collection_id, expected_revision=3)
    with pytest.raises(CollectionError) as deleted:
        store.list_history(collection_id, ids[0], collection_revision=3)
    assert deleted.value.status_code == 404
    with closing(sqlite3.connect(tmp_path / "store.sqlite3")) as connection:
        assert connection.execute("SELECT count(*) FROM paper_screening_history").fetchone()[0] == 2


def test_revision_order_pagination_is_stable_across_new_decisions_and_clock_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, collection_id, ids = make_store(tmp_path)
    timestamp = datetime(2026, 10, 15, tzinfo=UTC)
    monkeypatch.setattr(paper_screening, "datetime", Mock(now=Mock(return_value=timestamp)))
    for revision in range(5):
        store.submit(collection_id, ids[0], submission(revision=revision))
    first = store.list_history(collection_id, ids[0], collection_revision=1, limit=2)
    assert [item.revision for item in first.items] == [5, 4]
    assert first.next_cursor == 4
    store.submit(collection_id, ids[0], submission("exclude", 5))
    second = store.list_history(
        collection_id, ids[0], collection_revision=1, limit=2, cursor=first.next_cursor
    )
    third = store.list_history(
        collection_id, ids[0], collection_revision=1, limit=2, cursor=second.next_cursor
    )
    assert [item.revision for item in second.items + third.items] == [3, 2, 1]
    assert third.next_cursor is None and second.latest_review is not None
    assert second.latest_review.revision == 6
    assert {item.updated_at for item in first.items + second.items + third.items} == {timestamp}
    assert store.list_history(collection_id, ids[0], collection_revision=1, cursor=1).items == []
    for cursor in (7, MAX_REVISION):
        with pytest.raises(CollectionError) as error:
            store.list_history(collection_id, ids[0], collection_revision=1, cursor=cursor)
        assert error.value.code == "screening_history_cursor_invalid"
    with pytest.raises(CollectionError):
        store.list_history(collection_id, ids[1], collection_revision=1, cursor=1)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("limit", 0),
        ("limit", 101),
        ("limit", True),
        ("limit", 1.0),
        ("limit", "1"),
        ("cursor", 0),
        ("cursor", MAX_REVISION + 1),
        ("cursor", False),
        ("cursor", "1"),
        ("collection_revision", 0),
        ("collection_revision", True),
        ("collection_revision", 1.0),
        ("collection_revision", "1"),
        ("collection_id", "PRIVATE_ID"),
        ("document_id", " PRIVATE_ID"),
        ("document_id", ""),
        ("document_id", "\ud800"),
        ("document_id", "x" * 129),
    ],
)
def test_python_history_validates_without_echoing_private_inputs(
    tmp_path: Path, field: str, value: object
) -> None:
    store, collection_id, ids = make_store(tmp_path)
    options = {
        "collection_id": collection_id,
        "document_id": ids[0],
        "collection_revision": 1,
        field: value,
    }
    with pytest.raises(CollectionError) as error:
        store.list_history(**options)  # type: ignore[arg-type]
    assert error.value.code == "invalid_screening_history_request"
    assert error.value.status_code == 422 and "PRIVATE_" not in str(error.value)


def test_history_read_uses_one_read_only_snapshot_during_concurrent_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, collection_id, ids = make_store(tmp_path)
    saved = store.submit(collection_id, ids[0], submission())
    other = SQLitePaperScreening(tmp_path / "store.sqlite3")
    with closing(sqlite3.connect(tmp_path / "store.sqlite3")) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
    original = store._reviews

    def read(
        connection: sqlite3.Connection, collection: PaperCollection
    ) -> dict[str, ScreeningReview]:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            connection.execute("DELETE FROM paper_screening_history")
        other.collections.replace(
            collection_id, name="Concurrent revision", document_ids=ids, expected_revision=1
        )
        other.submit(
            collection_id,
            ids[0],
            submission("exclude", 1).model_copy(update={"collection_revision": 2}),
        )
        return original(connection, collection)

    monkeypatch.setattr(store, "_reviews", read)
    result = store.list_history(collection_id, ids[0], collection_revision=1)
    assert result.items == [saved] and result.latest_review == saved
    assert result.collection_revision == 1 and result.current_status == "include"
    assert (
        other.list_history(collection_id, ids[0], collection_revision=2).current_status == "exclude"
    )


def test_concurrent_submissions_append_only_the_winner(tmp_path: Path) -> None:
    store, collection_id, ids = make_store(tmp_path)
    barrier = threading.Barrier(2)

    def write(decision: str) -> str:
        barrier.wait(timeout=5)
        try:
            store.submit(collection_id, ids[0], submission(decision))
            return "saved"
        except CollectionError as exc:
            return exc.code

    with ThreadPoolExecutor(max_workers=2) as workers:
        assert sorted(workers.map(write, ("include", "exclude"))) == [
            "saved",
            "screening_revision_conflict",
        ]
    assert len(store.list_history(collection_id, ids[0], collection_revision=1).items) == 1


def test_history_response_byte_boundary_and_real_maximum_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, collection_id, ids = make_store(tmp_path)
    reason = "\U0001f52c" * 1000
    store.submit(collection_id, ids[0], submission().model_copy(update={"reason": reason}))
    page = store.list_history(collection_id, ids[0], collection_revision=1)
    content = page.model_dump_json()
    assert len(content.encode()) > len(content) and page.items[0].reason == reason
    with monkeypatch.context() as patch:
        patch.setattr(paper_screening, "MAX_HISTORY_RESPONSE_BYTES", len(content.encode()))
        assert store.list_history(collection_id, ids[0], collection_revision=1) == page
        patch.setattr(paper_screening, "MAX_HISTORY_RESPONSE_BYTES", len(content.encode()) - 1)
        with pytest.raises(CollectionError) as error:
            store.list_history(collection_id, ids[0], collection_revision=1)
        assert error.value.code == "screening_history_response_too_large"
    for revision in range(1, 101):
        store.submit(
            collection_id,
            ids[0],
            submission(revision=revision).model_copy(update={"reason": "\x01" * 1000}),
        )
    with pytest.raises(CollectionError) as error:
        store.list_history(collection_id, ids[0], collection_revision=1, limit=100)
    assert error.value.status_code == 413
    page = store.list_history(collection_id, ids[0], collection_revision=1, limit=20)
    assert len(page.items) == 20 and page.items[0].reason == "\x01" * 1000
    assert page.next_cursor == 82


@pytest.mark.parametrize(
    "record",
    [
        "PRIVATE_INVALID_JSON",
        "x" * (paper_screening.MAX_RECORD_CHARACTERS + 1),
        b"PRIVATE_BLOB",
        "\0PRIVATE_NUL",
    ],
)
def test_history_corrupt_envelopes_fail_privately(tmp_path: Path, record: object) -> None:
    store, collection_id, ids = make_store(tmp_path)
    store.submit(collection_id, ids[0], submission())
    with closing(sqlite3.connect(tmp_path / "store.sqlite3")) as connection, connection:
        connection.execute("UPDATE paper_screening_history SET record = ?", (record,))
    with pytest.raises(CollectionError) as error:
        store.list_history(collection_id, ids[0], collection_revision=1)
    assert error.value.code == "invalid_screening_history_record"
    assert "PRIVATE_" not in str(error.value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("document_id", "PRIVATE_FOREIGN"),
        ("collection_id", "col_" + "0" * 32),
        ("collection_revision", 2),
        ("revision", 7),
        ("reason", "PRIVATE_CHANGED"),
        ("updated_at", "PRIVATE_INVALID_TIME"),
    ],
)
def test_inconsistent_history_records_fail_closed(
    tmp_path: Path, field: str, value: object
) -> None:
    store, collection_id, ids = make_store(tmp_path)
    saved = store.submit(collection_id, ids[0], submission())
    with closing(sqlite3.connect(tmp_path / "store.sqlite3")) as connection, connection:
        connection.execute(
            "UPDATE paper_screening_history SET record = ?",
            (json.dumps({**saved.model_dump(mode="json"), field: value}),),
        )
    with pytest.raises(CollectionError) as error:
        store.list_history(collection_id, ids[0], collection_revision=1)
    assert error.value.code == "invalid_screening_history_record"


def test_gaps_and_unavailable_storage_are_not_empty_history(tmp_path: Path) -> None:
    store, collection_id, ids = make_store(tmp_path)
    for revision in range(3):
        store.submit(collection_id, ids[0], submission(revision=revision))
    with closing(sqlite3.connect(tmp_path / "store.sqlite3")) as connection, connection:
        connection.execute("DELETE FROM paper_screening_history WHERE revision = 2")
    with pytest.raises(CollectionError) as error:
        store.list_history(collection_id, ids[0], collection_revision=1)
    assert error.value.code == "invalid_screening_history_record"
    path = tmp_path / "store.sqlite3"
    path.rename(tmp_path / "offline.sqlite3")
    with pytest.raises(CollectionError) as unavailable:
        store.list_history(collection_id, ids[0], collection_revision=1)
    assert unavailable.value.code == "screening_storage_unavailable"
    assert not path.exists()
