"""Bounded, revisioned human screening of collection members, not scientific validation."""

import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    TypeAdapter,
    ValidationError,
)

from retrieval.scope import MAX_DOCUMENT_IDS
from storage.document_catalog import Identity
from storage.paper_collections import (
    MAX_REVISION,
    CollectionError,
    CollectionId,
    PaperCollection,
    Revision,
    SQLitePaperCollections,
)

MAX_REASON_CHARACTERS = 1000
MAX_RECORD_CHARACTERS = 16384
MAX_RESPONSE_BYTES = 524288
Decision = Literal["include", "exclude", "unsure"]
ScreeningStatus = Literal["unscreened", "include", "exclude", "unsure", "stale"]


def _readable_text(value: str) -> str:
    if not value.strip() or "\x00" in value:
        raise ValueError("Text must be nonblank and cannot contain NUL.")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("Text must contain valid Unicode.") from exc
    return value


Reason = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=MAX_REASON_CHARACTERS),
    AfterValidator(_readable_text),
]
_IDENTITY = TypeAdapter(Identity)


class ScreeningSubmission(BaseModel):
    """Zero is the initial decision revision, never an unconditional overwrite."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    collection_revision: Revision
    expected_decision_revision: int = Field(strict=True, ge=0, le=MAX_REVISION)
    decision: Decision
    reason: Reason


class ScreeningReview(BaseModel):
    """Latest human opinion only; older decisions are replaced, not an audit history."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_version: Literal["1.0"]
    collection_id: CollectionId
    document_id: Identity
    collection_revision: Revision
    revision: Revision
    decision: Decision
    reason: Reason
    updated_at: AwareDatetime


class ScreeningItem(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: Identity
    status: ScreeningStatus
    review: ScreeningReview | None


class ScreeningCounts(BaseModel):
    unscreened: int = Field(ge=0, le=MAX_DOCUMENT_IDS)
    include: int = Field(ge=0, le=MAX_DOCUMENT_IDS)
    exclude: int = Field(ge=0, le=MAX_DOCUMENT_IDS)
    unsure: int = Field(ge=0, le=MAX_DOCUMENT_IDS)
    stale: int = Field(ge=0, le=MAX_DOCUMENT_IDS)


class ScreeningQueue(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    collection_id: CollectionId
    collection_revision: Revision
    total_documents: int = Field(ge=1, le=MAX_DOCUMENT_IDS)
    counts: ScreeningCounts
    items: list[ScreeningItem] = Field(max_length=MAX_DOCUMENT_IDS)
    next_cursor: Identity | None
    included_document_ids: tuple[Identity, ...] = Field(max_length=MAX_DOCUMENT_IDS)


class _QueueOptions(BaseModel):
    collection_revision: Revision
    limit: int = Field(default=20, strict=True, ge=1, le=MAX_DOCUMENT_IDS)
    cursor: Identity | None = None
    status: ScreeningStatus | None = None


def _invalid_record() -> CollectionError:
    return CollectionError(
        "invalid_screening_record", "Saved screening metadata is invalid or inconsistent.", 409
    )


class SQLitePaperScreening:
    """Own each connection; reserve collection and review revisions before any write."""

    def __init__(self, database_path: Path | str) -> None:
        self.collections = SQLitePaperCollections(database_path)
        try:
            with closing(sqlite3.connect(database_path)) as connection, connection:
                connection.execute(
                    """CREATE TABLE IF NOT EXISTS paper_screening_decisions (
                        collection_id TEXT NOT NULL,
                        document_id TEXT NOT NULL,
                        record TEXT NOT NULL,
                        PRIMARY KEY (collection_id, document_id)
                    )"""
                )
        except sqlite3.Error as exc:
            raise CollectionError(
                "screening_storage_unavailable", "Screening storage is unavailable.", 503
            ) from exc

    @staticmethod
    def _reviews(
        connection: sqlite3.Connection, collection: PaperCollection
    ) -> dict[str, ScreeningReview]:
        rows = connection.execute(
            """SELECT document_id, substr(record, 1, ?) AS record,
                      typeof(record) != 'text' OR length(record) > ?
                          OR instr(record, char(0)) > 0 AS invalid
               FROM paper_screening_decisions
               WHERE collection_id = ?
                 AND document_id IN (SELECT value FROM json_each(?))
               LIMIT ?""",
            (
                MAX_RECORD_CHARACTERS + 1,
                MAX_RECORD_CHARACTERS,
                collection.collection_id,
                json.dumps(collection.document_ids),
                MAX_DOCUMENT_IDS + 1,
            ),
        ).fetchall()
        reviews: dict[str, ScreeningReview] = {}
        for row in rows:
            if row["invalid"]:
                raise _invalid_record()
            try:
                review = ScreeningReview.model_validate_json(row["record"])
            except ValidationError as exc:
                raise _invalid_record() from exc
            if (
                review.collection_id != collection.collection_id
                or review.document_id != row["document_id"]
                or review.collection_revision > collection.revision
                or review.document_id in reviews
            ):
                raise _invalid_record()
            reviews[review.document_id] = review
        return reviews

    def submit(
        self, collection_id: str, document_id: str, submission: ScreeningSubmission
    ) -> ScreeningReview:
        identifier = _IDENTITY.validate_python(document_id)
        # Revalidate even model_construct/copy inputs at the direct Python boundary.
        payload = ScreeningSubmission.model_validate(submission.model_dump())
        with self.collections.resolved_collection(
            collection_id, expected_revision=payload.collection_revision, write=True
        ) as (connection, collection):
            if identifier not in collection.document_ids:
                raise CollectionError(
                    "screening_document_not_member", "Document is not a collection member.", 422
                )
            previous = self._reviews(connection, collection).get(identifier)
            revision = previous.revision if previous is not None else 0
            if revision != payload.expected_decision_revision:
                raise CollectionError(
                    "screening_revision_conflict",
                    "Decision changed; read the screening queue before updating.",
                    409,
                )
            if revision == MAX_REVISION:
                raise CollectionError(
                    "screening_revision_exhausted", "Decision revision cannot increase.", 409
                )
            review = ScreeningReview(
                schema_version="1.0",
                collection_id=collection.collection_id,
                document_id=identifier,
                collection_revision=collection.revision,
                revision=revision + 1,
                decision=payload.decision,
                reason=payload.reason,
                updated_at=datetime.now(UTC),
            )
            record = review.model_dump_json()
            if len(record) > MAX_RECORD_CHARACTERS:
                raise _invalid_record()
            connection.execute(
                """INSERT INTO paper_screening_decisions (collection_id, document_id, record)
                   VALUES (?, ?, ?)
                   ON CONFLICT (collection_id, document_id)
                   DO UPDATE SET record = excluded.record""",
                (collection.collection_id, identifier, record),
            )
            return review

    def list_queue(
        self,
        collection_id: str,
        *,
        collection_revision: int,
        limit: int = 20,
        cursor: str | None = None,
        status: ScreeningStatus | None = None,
    ) -> ScreeningQueue:
        options = _QueueOptions(
            collection_revision=collection_revision, limit=limit, cursor=cursor, status=status
        )
        with self.collections.resolved_collection(
            collection_id, expected_revision=options.collection_revision
        ) as (connection, collection):
            reviews = self._reviews(connection, collection)
            if options.cursor is not None and options.cursor not in collection.document_ids:
                raise CollectionError(
                    "screening_cursor_invalid", "Cursor must identify a collection member.", 422
                )
            counts: dict[str, int] = {
                "unscreened": 0,
                "include": 0,
                "exclude": 0,
                "unsure": 0,
                "stale": 0,
            }
            items: list[ScreeningItem] = []
            included = []
            after_cursor = options.cursor is None
            for identifier in collection.document_ids:
                review = reviews.get(identifier)
                item_status: ScreeningStatus = "unscreened"
                if review is not None:
                    item_status = (
                        review.decision
                        if review.collection_revision == collection.revision
                        else "stale"
                    )
                counts[item_status] += 1
                if item_status == "include":
                    included.append(identifier)
                if after_cursor and (options.status is None or item_status == options.status):
                    try:
                        items.append(
                            ScreeningItem(document_id=identifier, status=item_status, review=review)
                        )
                    except ValidationError as exc:
                        raise _invalid_record() from exc
                if identifier == options.cursor:
                    after_cursor = True
            try:
                result = ScreeningQueue(
                    collection_id=collection.collection_id,
                    collection_revision=collection.revision,
                    total_documents=len(collection.document_ids),
                    counts=ScreeningCounts.model_validate(counts),
                    items=items[: options.limit],
                    next_cursor=items[options.limit - 1].document_id
                    if len(items) > options.limit
                    else None,
                    included_document_ids=tuple(included),
                )
            except ValidationError as exc:
                raise _invalid_record() from exc
            if len(result.model_dump_json().encode("utf-8")) > MAX_RESPONSE_BYTES:
                raise CollectionError(
                    "screening_response_too_large",
                    "Screening response exceeds its byte limit; request a smaller page.",
                    413,
                )
            return result
