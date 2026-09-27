"""Exercise review persistence failures and concurrency through the actual HTTP boundary."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from typing import Any
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from scripts.demo_evidence_export import offline_settings

from agent.review_models import AnswerReviewError, ReviewPageOptions, ReviewSubmission
from api.application import create_app
from storage.answer_reviews import SQLiteAnswerReviews
from tests.test_answer_reviews_api import ReviewAPI, raw_events, review_body
from tests.test_answer_reviews_api import review_api as review_api


@pytest.mark.parametrize("conflicting", [False, True])
def test_concurrent_retry_transactions_across_independent_apps(
    review_api: ReviewAPI, conflicting: bool
) -> None:
    api = review_api
    body = review_body(api)
    before = raw_events(api)
    restarted = create_app(offline_settings(api.database_path))
    barrier = Barrier(8)
    with TestClient(restarted) as second:

        def submit(index: int) -> tuple[int, dict[str, Any]]:
            client = api.client if index % 2 else second
            payload = (
                {**body, "comment": f"Concurrent opinion {index % 2}"} if conflicting else body
            )
            barrier.wait(timeout=10)
            response = client.post(api.path, json=payload)
            return response.status_code, response.json()

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(submit, range(8)))
    inserted = [record for status, record in results if status == 201]
    assert len(inserted) == 1
    if conflicting:
        assert [status for status, _ in results].count(409) == 4
        assert [status for status, _ in results].count(200) == 3
    else:
        assert [status for status, _ in results].count(200) == 7
    for status, record in results:
        if status in {200, 201}:
            assert record == inserted[0]
        else:
            assert status == 409
            assert record["detail"]["code"] == "review_id_conflict"
    assert api.client.get(api.path).json() == {"reviews": inserted, "next_cursor": None}
    assert raw_events(api) == before


@pytest.mark.parametrize(
    ("statement", "value"),
    [
        ("UPDATE answer_reviews SET schema_version = ?", "unsupported"),
        ("UPDATE answer_reviews SET review_id = ?", "invalid-uuid"),
        ("UPDATE answer_reviews SET decision = ?", "verified"),
        ("UPDATE answer_reviews SET comment = ?", " "),
        ("UPDATE answer_reviews SET comment = ?", "x" * 4001),
        ("UPDATE answer_reviews SET comment = ?", b"blob-comment"),
        ("UPDATE answer_reviews SET created_at = ?", "2026-09-22T12:00:00"),
        ("UPDATE answer_reviews SET created_at = ?", "12345"),
        ("UPDATE answer_reviews SET created_at = ?", "not-a-date"),
        ("UPDATE answer_reviews SET cited_chunk_ids = ?", "{private-invalid-json"),
        ("UPDATE answer_reviews SET cited_chunk_ids = ?", "{}"),
        ("UPDATE answer_reviews SET cited_chunk_ids = ?", "null"),
        ("UPDATE answer_reviews SET cited_chunk_ids = ?", '["outside-the-bundle"]'),
        ("UPDATE answer_reviews SET cited_chunk_ids = ?", '["duplicate","duplicate"]'),
        ("UPDATE answer_reviews SET cited_chunk_ids = ?", b"[]"),
        ("UPDATE answer_reviews SET cited_chunk_ids = ?", "[" * 2000 + "]" * 2000),
    ],
)
def test_corrupt_saved_reviews_are_not_returned_as_success(
    review_api: ReviewAPI, statement: str, value: str | bytes
) -> None:
    body = review_body(review_api)
    assert review_api.client.post(review_api.path, json=body).status_code == 201
    with sqlite3.connect(review_api.database_path) as connection:
        connection.execute("PRAGMA ignore_check_constraints = ON")
        connection.execute(statement, (value,))
    response = review_api.client.get(review_api.path)
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "invalid_review_record"
    assert "private-invalid-json" not in response.text
    assert "blob-comment" not in response.text
    assert str(review_api.database_path) not in response.text


def test_lookahead_row_corruption_is_explicit(review_api: ReviewAPI) -> None:
    first = review_body(review_api)
    assert review_api.client.post(review_api.path, json=first).status_code == 201
    assert review_api.client.post(review_api.path, json=review_body(review_api)).status_code == 201
    with sqlite3.connect(review_api.database_path) as connection:
        connection.execute(
            "UPDATE answer_reviews SET created_at = 'invalid' WHERE review_id = ?",
            (first["review_id"],),
        )
    response = review_api.client.get(review_api.path, params={"limit": 1})
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "invalid_review_record"


def test_corrupt_existing_retry_does_not_append_a_replacement(review_api: ReviewAPI) -> None:
    body = review_body(review_api)
    assert review_api.client.post(review_api.path, json=body).status_code == 201
    with sqlite3.connect(review_api.database_path) as connection:
        connection.execute("UPDATE answer_reviews SET created_at = 'invalid'")
    response = review_api.client.post(review_api.path, json=body)
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "invalid_review_record"
    with sqlite3.connect(review_api.database_path) as connection:
        assert connection.execute("SELECT count(*) FROM answer_reviews").fetchone()[0] == 1


def test_missing_review_table_returns_sanitized_storage_errors(review_api: ReviewAPI) -> None:
    before = raw_events(review_api)
    with sqlite3.connect(review_api.database_path) as connection:
        connection.execute("DROP TABLE answer_reviews")
    for response in (
        review_api.client.get(review_api.path),
        review_api.client.post(review_api.path, json=review_body(review_api)),
    ):
        assert response.status_code == 503
        assert response.json()["detail"]["code"] == "review_storage_unavailable"
        assert "no such table" not in response.text
        assert str(review_api.database_path) not in response.text
        assert response.headers["cache-control"] == "no-store"
    assert raw_events(review_api) == before
    assert review_api.client.get(f"/runs/{review_api.run_id}/export").status_code == 200


@pytest.mark.parametrize("method", ["GET", "POST"])
def test_evidence_read_storage_failure_is_explicit_and_sanitized(
    review_api: ReviewAPI, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    def broken_read(run_id: str) -> list[dict[str, Any]]:
        raise sqlite3.OperationalError("synthetic-private-database-path-or-error")

    monkeypatch.setattr(review_api.container.event_log, "list_events", broken_read)
    response = review_api.client.request(method, review_api.path, json=review_body(review_api))
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "review_storage_unavailable"
    assert "synthetic-private" not in response.text


def test_failed_insert_rolls_back_and_retry_can_succeed(review_api: ReviewAPI) -> None:
    api = review_api
    body = review_body(api)
    before = raw_events(api)
    with sqlite3.connect(api.database_path) as connection:
        connection.execute(
            """
            CREATE TRIGGER fail_review_insert AFTER INSERT ON answer_reviews
            BEGIN SELECT RAISE(ABORT, 'synthetic-private-disk-error'); END
            """
        )
    response = api.client.post(api.path, json=body)
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "review_storage_unavailable"
    assert "synthetic-private" not in response.text
    assert api.client.get(api.path).json() == {"reviews": [], "next_cursor": None}
    with sqlite3.connect(api.database_path) as connection:
        connection.execute("DROP TRIGGER fail_review_insert")
    retry = api.client.post(api.path, json=body)
    assert retry.status_code == 201
    assert api.client.get(api.path).json()["reviews"] == [retry.json()]
    assert raw_events(api) == before


def test_invalid_inserted_record_rolls_back_before_commit(review_api: ReviewAPI) -> None:
    with sqlite3.connect(review_api.database_path) as connection:
        connection.execute(
            """
            CREATE TRIGGER corrupt_inserted_review AFTER INSERT ON answer_reviews
            BEGIN
                UPDATE answer_reviews SET created_at = 'corrupt' WHERE id = NEW.id;
            END
            """
        )
    response = review_api.client.post(review_api.path, json=review_body(review_api))
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "invalid_review_record"
    assert review_api.client.get(review_api.path).json() == {"reviews": [], "next_cursor": None}


def test_write_lock_is_bounded_and_does_not_consume_review_id(review_api: ReviewAPI) -> None:
    body = review_body(review_api)
    with sqlite3.connect(review_api.database_path) as blocker:
        blocker.execute("BEGIN IMMEDIATE")
        response = review_api.client.post(review_api.path, json=body)
        assert response.status_code == 503
        assert response.json()["detail"]["code"] == "review_storage_unavailable"
        blocker.rollback()
    assert review_api.client.get(review_api.path).json() == {"reviews": [], "next_cursor": None}
    assert review_api.client.post(review_api.path, json=body).status_code == 201


def test_requests_do_not_silently_recreate_a_missing_review_database(tmp_path: Path) -> None:
    path = tmp_path / "removed.sqlite3"
    store = SQLiteAnswerReviews(path)
    path.unlink()
    with pytest.raises(AnswerReviewError) as caught:
        store.list_reviews("run", frozen_chunk_ids=frozenset())
    assert caught.value.code == "review_storage_unavailable"
    assert not path.exists()
    with pytest.raises(AnswerReviewError) as caught:
        store.append(
            "run",
            ReviewSubmission(review_id=uuid4(), decision="accepted", comment="Opinion."),
            frozen_chunk_ids=frozenset(),
        )
    assert caught.value.status_code == 503
    assert not path.exists()


def test_initialization_failure_is_not_hidden(tmp_path: Path) -> None:
    with pytest.raises(AnswerReviewError) as caught:
        SQLiteAnswerReviews(tmp_path)
    assert caught.value.status_code == 503
    assert caught.value.code == "review_storage_unavailable"


def test_storage_uses_bound_parameters_and_accepts_exact_id_limits(tmp_path: Path) -> None:
    store = SQLiteAnswerReviews(tmp_path / "bound.sqlite3")
    run_id = "run'; DROP TABLE answer_reviews; --"
    chunk_ids = tuple(f"{index:03d}" + "x" * 253 for index in range(100))
    submission = ReviewSubmission(
        review_id=uuid4(),
        decision="accepted",
        comment="An opinion'); DROP TABLE answer_reviews; --",
        cited_chunk_ids=chunk_ids,
    )
    saved, created = store.append(run_id, submission, frozen_chunk_ids=frozenset(chunk_ids))
    assert created
    assert saved.comment == submission.comment
    assert saved.cited_chunk_ids == chunk_ids
    assert store.list_reviews(run_id, frozen_chunk_ids=frozenset(chunk_ids)).reviews == [saved]
    assert store.list_reviews("' OR 1=1 --", frozen_chunk_ids=frozenset()).reviews == []


@pytest.mark.parametrize("limit", [0, 101, True, 1.5, "2"])
def test_direct_pagination_callers_cannot_bypass_bounds(limit: object) -> None:
    with pytest.raises(ValidationError):
        ReviewPageOptions.model_validate({"limit": limit})
