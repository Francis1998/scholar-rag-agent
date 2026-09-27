"""Append-only SQLite reviews, stored independently of the saved agent event log."""

import json
import sqlite3
from contextlib import closing
from pathlib import Path

from pydantic import ValidationError

from agent.review_models import (
    DEFAULT_REVIEW_LIMIT,
    AnswerReviewError,
    ReviewHistoryPage,
    ReviewPageOptions,
    ReviewSubmission,
    SavedAnswerReview,
    invalid_review_record,
    review_storage_unavailable,
)

_SELECT_REVIEW = """
SELECT id AS sequence, schema_version, run_id, review_id, created_at,
       decision, comment, cited_chunk_ids
FROM answer_reviews
"""


def _read_record(row: sqlite3.Row) -> SavedAnswerReview:
    record = dict(row)
    encoded_ids = record["cited_chunk_ids"]
    if not isinstance(encoded_ids, str):
        raise invalid_review_record()
    try:
        record["cited_chunk_ids"] = json.loads(encoded_ids)
        review = SavedAnswerReview.model_validate(record)
    except (ValidationError, json.JSONDecodeError, RecursionError) as exc:
        raise invalid_review_record() from exc
    if record["review_id"] != str(review.review_id):
        raise invalid_review_record()
    return review


def ensure_saved_references(review: SavedAnswerReview, frozen_chunk_ids: frozenset[str]) -> None:
    """Out-of-bundle references in persisted rows are corruption, not a new request error."""
    if not frozen_chunk_ids.issuperset(review.cited_chunk_ids):
        raise invalid_review_record()


class SQLiteAnswerReviews:
    """Serialize retries and appends with short transactions and a run-scoped unique key."""

    def __init__(self, database_path: Path | str) -> None:
        path = Path(database_path).resolve()
        self._database_uri = path.as_uri() + "?mode=rw"
        try:
            with closing(sqlite3.connect(path)) as connection, connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS answer_reviews (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        schema_version TEXT NOT NULL CHECK (schema_version = '1.0'),
                        run_id TEXT NOT NULL CHECK (length(run_id) BETWEEN 1 AND 256),
                        review_id TEXT NOT NULL CHECK (length(review_id) = 36),
                        created_at TEXT NOT NULL,
                        decision TEXT NOT NULL
                            CHECK (decision IN ('accepted', 'needs_revision', 'rejected')),
                        comment TEXT NOT NULL CHECK (length(comment) BETWEEN 1 AND 4000),
                        cited_chunk_ids TEXT NOT NULL
                            CHECK (json_valid(cited_chunk_ids)
                                   AND json_type(cited_chunk_ids) = 'array'
                                   AND json_array_length(cited_chunk_ids) <= 100),
                        UNIQUE (run_id, review_id)
                    )
                    """
                )
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS idx_answer_reviews_run_sequence "
                    "ON answer_reviews(run_id, id)"
                )
        except sqlite3.Error as exc:
            raise review_storage_unavailable() from exc

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_uri, uri=True, timeout=5)
        connection.row_factory = sqlite3.Row
        return connection

    def append(
        self,
        run_id: str,
        submission: ReviewSubmission,
        *,
        frozen_chunk_ids: frozenset[str],
    ) -> tuple[SavedAnswerReview, bool]:
        """Return the original identical retry or atomically insert exactly one new review."""
        try:
            with closing(self._connect()) as connection, connection:
                connection.execute("BEGIN IMMEDIATE")
                previous = connection.execute(
                    _SELECT_REVIEW + " WHERE run_id = ? AND review_id = ?",
                    (run_id, str(submission.review_id)),
                ).fetchone()
                if previous is not None:
                    saved = _read_record(previous)
                    ensure_saved_references(saved, frozen_chunk_ids)
                    if (saved.decision, saved.comment, saved.cited_chunk_ids) != (
                        submission.decision,
                        submission.comment,
                        submission.cited_chunk_ids,
                    ):
                        raise AnswerReviewError(
                            "review_id_conflict",
                            "This review ID already exists for this run with different content.",
                            409,
                        )
                    return saved, False
                if not frozen_chunk_ids.issuperset(submission.cited_chunk_ids):
                    raise AnswerReviewError(
                        "invalid_review_references",
                        "Cited chunk IDs must belong to this run's frozen evidence bundle.",
                        422,
                    )
                inserted = connection.execute(
                    """
                    INSERT INTO answer_reviews (
                        schema_version, run_id, review_id, created_at,
                        decision, comment, cited_chunk_ids
                    ) VALUES ('1.0', ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), ?, ?, ?)
                    """,
                    (
                        run_id,
                        str(submission.review_id),
                        submission.decision,
                        submission.comment,
                        json.dumps(submission.cited_chunk_ids, separators=(",", ":")),
                    ),
                )
                row = connection.execute(
                    _SELECT_REVIEW + " WHERE id = ?", (inserted.lastrowid,)
                ).fetchone()
                if row is None:
                    raise invalid_review_record()
                saved = _read_record(row)
                ensure_saved_references(saved, frozen_chunk_ids)
                return saved, True
        except sqlite3.Error as exc:
            raise review_storage_unavailable() from exc

    def list_reviews(
        self,
        run_id: str,
        *,
        limit: int = DEFAULT_REVIEW_LIMIT,
        cursor: int | None = None,
        frozen_chunk_ids: frozenset[str],
    ) -> ReviewHistoryPage:
        """Validate the lookahead too, so corrupt rows cannot masquerade as a next page."""
        options = ReviewPageOptions(limit=limit, cursor=cursor)
        try:
            with closing(self._connect()) as connection:
                rows = connection.execute(
                    _SELECT_REVIEW
                    + " WHERE run_id = ? AND (? IS NULL OR id < ?) ORDER BY id DESC LIMIT ?",
                    (run_id, options.cursor, options.cursor, options.limit + 1),
                ).fetchall()
        except sqlite3.Error as exc:
            raise review_storage_unavailable() from exc
        reviews = []
        for row in rows:
            review = _read_record(row)
            ensure_saved_references(review, frozen_chunk_ids)
            reviews.append(review)
        page = reviews[: options.limit]
        return ReviewHistoryPage(
            reviews=page,
            next_cursor=page[-1].sequence if len(reviews) > options.limit else None,
        )
