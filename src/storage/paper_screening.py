"""Bounded, revisioned human screening of collection members, not scientific validation."""

import csv
import io
import json
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
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
from storage.document_catalog import (
    MAX_SOURCE_CHARACTERS,
    MAX_TITLE_CHARACTERS,
    DocumentCatalogError,
    Identity,
    _prefix,
)
from storage.paper_collections import (
    MAX_REVISION,
    CollectionError,
    CollectionId,
    CollectionName,
    PaperCollection,
    Revision,
    SQLitePaperCollections,
)

MAX_REASON_CHARACTERS = 1000
MAX_RECORD_CHARACTERS = 16384
MAX_RESPONSE_BYTES = 524288
MAX_EXPORT_BYTES = 524288
Decision = Literal["include", "exclude", "unsure"]
ScreeningStatus = Literal["unscreened", "include", "exclude", "unsure", "stale"]
ScreeningExportFormat = Literal["json", "csv"]


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


class ScreeningExportItem(ScreeningItem):
    """Current catalog labels beside the latest recorded human opinion."""

    title: str = Field(max_length=MAX_TITLE_CHARACTERS)
    title_truncated: bool
    source: str = Field(max_length=MAX_SOURCE_CHARACTERS)
    source_truncated: bool


class ScreeningExport(BaseModel):
    """One complete read snapshot, not frozen paper contents or review history."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    collection_id: CollectionId
    collection_name: CollectionName
    collection_revision: Revision
    total_documents: int = Field(ge=1, le=MAX_DOCUMENT_IDS)
    counts: ScreeningCounts
    items: list[ScreeningExportItem] = Field(min_length=1, max_length=MAX_DOCUMENT_IDS)
    included_document_ids: tuple[Identity, ...] = Field(max_length=MAX_DOCUMENT_IDS)


@dataclass(frozen=True)
class ScreeningDownload:
    """Already serialized and byte-checked; API and Python save identical bytes."""

    content: bytes
    media_type: str
    filename: str


class _QueueOptions(BaseModel):
    collection_revision: Revision
    limit: int = Field(default=20, strict=True, ge=1, le=MAX_DOCUMENT_IDS)
    cursor: Identity | None = None
    status: ScreeningStatus | None = None


class _ExportOptions(BaseModel):
    collection_revision: Revision
    format: ScreeningExportFormat = "json"


def _csv_cell(value: str | int | bool | None) -> str | int:
    # Uniform prefixes are reversible and protect even whitespace/control-led formulas.
    if isinstance(value, str):
        return "'" + value
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _csv_export(result: ScreeningExport) -> bytes:
    output = io.StringIO(newline="")
    fields = (
        "schema_version",
        "csv_text_encoding",
        "collection_id",
        "collection_name",
        "collection_revision",
        "total_documents",
        "count_unscreened",
        "count_include",
        "count_exclude",
        "count_unsure",
        "count_stale",
        "document_id",
        "title",
        "title_truncated",
        "source",
        "source_truncated",
        "status",
        "included_document_id",
        "review_schema_version",
        "review_collection_revision",
        "review_revision",
        "review_decision",
        "review_reason",
        "review_updated_at",
    )
    writer = csv.DictWriter(output, fieldnames=fields, quoting=csv.QUOTE_ALL, lineterminator="\r\n")
    writer.writeheader()
    for item in result.items:
        review = item.review
        row: dict[str, str | int | bool | None] = {
            "schema_version": result.schema_version,
            "csv_text_encoding": "apostrophe-prefix-v1",
            "collection_id": result.collection_id,
            "collection_name": result.collection_name,
            "collection_revision": result.collection_revision,
            "total_documents": result.total_documents,
            **{f"count_{status}": count for status, count in result.counts.model_dump().items()},
            "document_id": item.document_id,
            "title": item.title,
            "title_truncated": item.title_truncated,
            "source": item.source,
            "source_truncated": item.source_truncated,
            "status": item.status,
            "included_document_id": item.document_id if item.status == "include" else None,
            "review_schema_version": review.schema_version if review else None,
            "review_collection_revision": review.collection_revision if review else None,
            "review_revision": review.revision if review else None,
            "review_decision": review.decision if review else None,
            "review_reason": review.reason if review else None,
            "review_updated_at": review.updated_at.isoformat() if review else None,
        }
        writer.writerow({field: _csv_cell(value) for field, value in row.items()})
    return output.getvalue().encode("utf-8")


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

    def _snapshot(
        self, connection: sqlite3.Connection, collection: PaperCollection
    ) -> ScreeningQueue:
        reviews = self._reviews(connection, collection)
        counts: dict[str, int] = {
            "unscreened": 0,
            "include": 0,
            "exclude": 0,
            "unsure": 0,
            "stale": 0,
        }
        items = []
        included = []
        for identifier in collection.document_ids:
            review = reviews.get(identifier)
            status: ScreeningStatus = "unscreened"
            if review is not None:
                status = (
                    review.decision
                    if review.collection_revision == collection.revision
                    else "stale"
                )
            counts[status] += 1
            if status == "include":
                included.append(identifier)
            items.append(ScreeningItem(document_id=identifier, status=status, review=review))
        try:
            return ScreeningQueue(
                collection_id=collection.collection_id,
                collection_revision=collection.revision,
                total_documents=len(collection.document_ids),
                counts=ScreeningCounts.model_validate(counts),
                items=items,
                next_cursor=None,
                included_document_ids=tuple(included),
            )
        except ValidationError as exc:
            raise _invalid_record() from exc

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
            snapshot = self._snapshot(connection, collection)
            if options.cursor is not None and options.cursor not in collection.document_ids:
                raise CollectionError(
                    "screening_cursor_invalid", "Cursor must identify a collection member.", 422
                )
            start = (
                0 if options.cursor is None else collection.document_ids.index(options.cursor) + 1
            )
            items = [
                item
                for item in snapshot.items[start:]
                if options.status is None or item.status == options.status
            ]
            result = snapshot.model_copy(
                update={
                    "items": items[: options.limit],
                    "next_cursor": items[options.limit - 1].document_id
                    if len(items) > options.limit
                    else None,
                }
            )
            if len(result.model_dump_json().encode("utf-8")) > MAX_RESPONSE_BYTES:
                raise CollectionError(
                    "screening_response_too_large",
                    "Screening response exceeds its byte limit; request a smaller page.",
                    413,
                )
            return result

    def export_results(
        self,
        collection_id: str,
        *,
        collection_revision: int,
        format: ScreeningExportFormat = "json",
    ) -> ScreeningDownload:
        """Download every current member in one read transaction, without partial success."""
        options = _ExportOptions(collection_revision=collection_revision, format=format)
        with self.collections.resolved_collection(
            collection_id, expected_revision=options.collection_revision
        ) as (connection, collection):
            snapshot = self._snapshot(connection, collection)
            encoding = connection.execute("PRAGMA encoding").fetchone()[0]
            # SQLite substr on an empty BLOB can return NULL; empty text labels are valid.
            rows = connection.execute(
                """SELECT document_id,
                          CASE WHEN title = '' THEN X''
                              ELSE substr(CAST(title AS BLOB), 1, ?) END AS title,
                          CASE WHEN source = '' THEN X''
                              ELSE substr(CAST(source AS BLOB), 1, ?) END AS source,
                          typeof(title) != 'text' OR typeof(source) != 'text'
                              OR instr(title, char(0)) > 0
                              OR instr(source, char(0)) > 0 AS invalid
                   FROM documents
                   WHERE document_id IN (SELECT value FROM json_each(?))
                   LIMIT ?""",
                (
                    4 * (MAX_TITLE_CHARACTERS + 1),
                    4 * (MAX_SOURCE_CHARACTERS + 1),
                    json.dumps(collection.document_ids),
                    MAX_DOCUMENT_IDS + 1,
                ),
            ).fetchall()
            labels = {row["document_id"]: row for row in rows}
            try:
                if (
                    len(labels) != len(rows)
                    or set(labels) != set(collection.document_ids)
                    or any(row["invalid"] for row in rows)
                ):
                    raise DocumentCatalogError()
                items = []
                for item in snapshot.items:
                    row = labels[item.document_id]
                    title = _prefix(row["title"], encoding, MAX_TITLE_CHARACTERS)
                    source = _prefix(row["source"], encoding, MAX_SOURCE_CHARACTERS)
                    items.append(
                        ScreeningExportItem(
                            **item.model_dump(),
                            title=title[:MAX_TITLE_CHARACTERS],
                            title_truncated=len(title) > MAX_TITLE_CHARACTERS,
                            source=source[:MAX_SOURCE_CHARACTERS],
                            source_truncated=len(source) > MAX_SOURCE_CHARACTERS,
                        )
                    )
            except (DocumentCatalogError, ValidationError) as exc:
                error = DocumentCatalogError()
                raise CollectionError(error.code, str(error), 409) from exc
            result = ScreeningExport(
                collection_id=collection.collection_id,
                collection_name=collection.name,
                collection_revision=collection.revision,
                total_documents=snapshot.total_documents,
                counts=snapshot.counts,
                items=items,
                included_document_ids=snapshot.included_document_ids,
            )
        content = (
            (result.model_dump_json(indent=2) + "\n").encode("utf-8")
            if options.format == "json"
            else _csv_export(result)
        )
        if len(content) > MAX_EXPORT_BYTES:
            raise CollectionError(
                "screening_export_too_large",
                "Complete screening export exceeds the download byte limit; no partial export.",
                413,
            )
        slug = re.sub(r"[^A-Za-z0-9]+", "-", collection.name).strip("-")[:40] or "collection"
        return ScreeningDownload(
            content=content,
            media_type="application/json"
            if options.format == "json"
            else "text/csv; charset=utf-8",
            filename=f"screening-{slug}-{collection.collection_id}-r{collection.revision}.{options.format}",
        )
