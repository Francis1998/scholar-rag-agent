"""Immutable human notes on exact spans of completed, frozen run evidence."""

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from agent.evidence import SHA256, CaptureLimits, EvidenceSource
from storage.document_chunks import ChunkIdentity
from storage.evidence_export import BoundedRunEvents, EvidenceExporter, EvidenceExportError
from storage.run_history import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE, EventID

MAX_NOTE_CHARACTERS = 1000
MAX_QUOTE_CHARACTERS = 1000
MAX_RECORD_TEXT_BYTES = 16384
MAX_RESPONSE_BYTES = 262144


def _unicode(value: str) -> str:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("Text must contain valid Unicode.") from exc
    return value


def _note(value: str) -> str:
    if not value.strip() or "\x00" in value:
        raise ValueError("A human note must be nonblank and cannot contain NUL.")
    return _unicode(value)


# Imported chunk/document IDs are opaque: do not trim, case-fold, or ban punctuation.
AnnotationIdentity = Annotated[ChunkIdentity, AfterValidator(_unicode)]
SpanOffset = Annotated[int, Field(strict=True, ge=0, le=CaptureLimits().max_context_bytes)]
_IDENTITY = TypeAdapter(AnnotationIdentity)


class AnnotationSubmission(BaseModel):
    """Client UUID plus exact payload is a run-scoped retry identity, not an author identity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    annotation_id: UUID
    document_id: AnnotationIdentity
    chunk_id: AnnotationIdentity
    source_text_sha256: Annotated[SHA256, Field(strict=True)]
    start: SpanOffset
    end: SpanOffset
    quote: Annotated[
        str,
        Field(strict=True, min_length=1, max_length=MAX_QUOTE_CHARACTERS),
        AfterValidator(_unicode),
    ]
    note: Annotated[
        str,
        Field(strict=True, min_length=1, max_length=MAX_NOTE_CHARACTERS),
        AfterValidator(_note),
    ]

    @model_validator(mode="after")
    def ordered_span(self) -> Self:
        if self.start >= self.end:
            raise ValueError("A half-open span must have start < end.")
        return self


class SavedEvidenceAnnotation(AnnotationSubmission):
    """Versioned provenance plus a human opinion; not entailment or a signed audit record."""

    schema_version: Literal["1.0"]
    sequence: EventID
    run_id: AnnotationIdentity
    created_at: AwareDatetime

    @field_validator("created_at", mode="before")
    @classmethod
    def utc_timestamp(cls, value: object) -> object:
        if (
            not isinstance(value, str)
            or len(value) > 40
            or "T" not in value
            or not value.endswith("Z")
        ):
            raise ValueError("Saved annotation timestamps must be UTC date-time text.")
        return value


class AnnotationPage(BaseModel):
    """Newest-first annotations, with an exclusive sequence cursor for older records."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    annotations: list[SavedEvidenceAnnotation] = Field(max_length=MAX_PAGE_SIZE)
    next_cursor: EventID | None


class _PageOptions(BaseModel):
    limit: int = Field(default=DEFAULT_PAGE_SIZE, strict=True, ge=1, le=MAX_PAGE_SIZE)
    cursor: EventID | None = None


class AnnotationError(EvidenceExportError):
    """Sanitized failure shared by direct Python callers and the HTTP boundary."""


def _invalid_request() -> AnnotationError:
    return AnnotationError(
        "invalid_annotation_request", "Annotation fields, run ID, or pagination are invalid.", 422
    )


def _invalid_record() -> AnnotationError:
    return AnnotationError(
        "invalid_annotation_record",
        "Saved annotations are invalid, unsupported, or inconsistent with frozen evidence.",
    )


def _unavailable() -> AnnotationError:
    return AnnotationError(
        "annotation_storage_unavailable",
        "Saved annotation or evidence storage is unavailable.",
        503,
    )


def _bounded_response(record: BaseModel) -> None:
    if len(record.model_dump_json().encode("utf-8")) > MAX_RESPONSE_BYTES:
        raise AnnotationError(
            "annotation_response_too_large",
            "Annotations exceed the 262144-byte UTF-8 response limit; request a smaller page.",
            413,
        )


_SIZES_SQL = """
SELECT typeof(sequence) != 'integer' OR typeof(start) != 'integer'
           OR typeof(end) != 'integer' OR typeof(schema_version) != 'text'
           OR typeof(run_id) != 'text' OR typeof(annotation_id) != 'text'
           OR typeof(document_id) != 'text' OR typeof(chunk_id) != 'text'
           OR typeof(source_text_sha256) != 'text' OR typeof(quote) != 'text'
           OR typeof(note) != 'text' OR typeof(created_at) != 'text' AS invalid_record,
       length(CAST(schema_version AS BLOB)) + length(CAST(run_id AS BLOB))
           + length(CAST(annotation_id AS BLOB)) + length(CAST(document_id AS BLOB))
           + length(CAST(chunk_id AS BLOB)) + length(CAST(source_text_sha256 AS BLOB))
           + length(CAST(quote AS BLOB)) + length(CAST(note AS BLOB))
           + length(CAST(created_at AS BLOB)) AS text_bytes
FROM evidence_annotations
"""
_RECORDS_SQL = """
SELECT sequence, start, end, CAST(schema_version AS BLOB) AS schema_version,
       CAST(run_id AS BLOB) AS run_id, CAST(annotation_id AS BLOB) AS annotation_id,
       CAST(document_id AS BLOB) AS document_id, CAST(chunk_id AS BLOB) AS chunk_id,
       CAST(source_text_sha256 AS BLOB) AS source_text_sha256,
       CAST(quote AS BLOB) AS quote, CAST(note AS BLOB) AS note,
       CAST(created_at AS BLOB) AS created_at
FROM evidence_annotations
"""


def _anchor(
    annotation: AnnotationSubmission, sources: dict[str, EvidenceSource], *, saved: bool
) -> None:
    source = sources.get(annotation.chunk_id)
    if source is None:
        if saved:
            raise _invalid_record()
        raise AnnotationError(
            "annotation_source_not_found", "The chunk is not in this run's frozen evidence.", 404
        )
    text = source.chunk.text
    if (
        annotation.document_id != source.chunk.document_id
        or annotation.source_text_sha256 != source.text_sha256
        or annotation.end > len(text)
        or annotation.quote != text[annotation.start : annotation.end]
    ):
        if saved:
            raise _invalid_record()
        raise AnnotationError(
            "invalid_annotation_selector",
            "Document, digest, offsets, and quote must exactly match the frozen source text.",
            422,
        )


class SQLiteEvidenceAnnotations:
    """Initialize only an additive table in an existing database; never initialize run events."""

    def __init__(self, database_path: Path | str) -> None:
        self._database_uri = Path(database_path).resolve().as_uri()
        try:
            with closing(self._connect(write=True)) as connection, connection:
                connection.execute(
                    """CREATE TABLE IF NOT EXISTS evidence_annotations (
                        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                        schema_version TEXT NOT NULL,
                        run_id TEXT NOT NULL,
                        annotation_id TEXT NOT NULL,
                        document_id TEXT NOT NULL,
                        chunk_id TEXT NOT NULL,
                        source_text_sha256 TEXT NOT NULL,
                        start INTEGER NOT NULL,
                        end INTEGER NOT NULL,
                        quote TEXT NOT NULL,
                        note TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        UNIQUE (run_id, annotation_id)
                    )"""
                )
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS idx_evidence_annotations_run_sequence "
                    "ON evidence_annotations(run_id, sequence)"
                )
        except sqlite3.Error as exc:
            raise _unavailable() from exc

    def _connect(self, *, write: bool = False) -> sqlite3.Connection:
        mode = "rw" if write else "ro"
        connection = sqlite3.connect(self._database_uri + f"?mode={mode}", uri=True, timeout=5)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _sources(
        connection: sqlite3.Connection, encoding: str, run_id: str
    ) -> dict[str, EvidenceSource]:
        try:
            bundle = EvidenceExporter(BoundedRunEvents(connection, encoding)).export(run_id)
            for source in bundle.snapshot.sources:
                _IDENTITY.validate_python(source.chunk.chunk_id)
                _IDENTITY.validate_python(source.chunk.document_id)
        except (ValidationError, UnicodeError, RecursionError) as exc:
            raise AnnotationError(
                "invalid_run_record",
                "Saved evidence is inconsistent, invalid, or an unsupported version.",
            ) from exc
        return {source.chunk.chunk_id: source for source in bundle.snapshot.sources}

    @staticmethod
    def _read(
        connection: sqlite3.Connection,
        encoding: str,
        run_id: str,
        sources: dict[str, EvidenceSource],
        *,
        annotation_id: UUID | None = None,
        cursor: int | None = None,
        fetch_limit: int,
    ) -> list[SavedEvidenceAnnotation]:
        parameters = {
            "run_id": run_id,
            "annotation_id": str(annotation_id) if annotation_id is not None else None,
            "cursor": cursor,
            "fetch_limit": fetch_limit,
        }
        selector = " WHERE run_id = :run_id"
        if annotation_id is not None:
            selector += " AND annotation_id = :annotation_id"
        if cursor is not None:
            selector += " AND sequence < :cursor"
        selector += " ORDER BY sequence DESC LIMIT :fetch_limit"
        sizes = connection.execute(_SIZES_SQL + selector, parameters).fetchall()
        factor = 1 if encoding == "UTF-8" else 2
        if any(row["invalid_record"] for row in sizes):
            raise _invalid_record()
        if any(row["text_bytes"] > factor * MAX_RECORD_TEXT_BYTES for row in sizes):
            raise _invalid_record()
        rows = connection.execute(_RECORDS_SQL + selector, parameters).fetchall()
        result = []
        for row in rows:
            try:
                record = {
                    key: value.decode(encoding) if isinstance(value, bytes) else value
                    for key, value in dict(row).items()
                }
                if (
                    sum(
                        len(value.encode("utf-8"))
                        for value in record.values()
                        if isinstance(value, str)
                    )
                    > MAX_RECORD_TEXT_BYTES
                ):
                    raise _invalid_record()
                annotation = SavedEvidenceAnnotation.model_validate(record)
            except (ValidationError, UnicodeError) as exc:
                raise _invalid_record() from exc
            if annotation.run_id != run_id or record["annotation_id"] != str(
                annotation.annotation_id
            ):
                raise _invalid_record()
            _anchor(annotation, sources, saved=True)
            result.append(annotation)
        return result

    def create(
        self, run_id: str, submission: AnnotationSubmission
    ) -> tuple[SavedEvidenceAnnotation, bool]:
        """Atomically replay identical UUID/content or append once; return (record, created)."""
        try:
            run_id = _IDENTITY.validate_python(run_id)
            payload = AnnotationSubmission.model_validate(submission.model_dump(warnings=False))
        except ValidationError as exc:
            raise _invalid_request() from exc
        try:
            with closing(self._connect(write=True)) as connection, connection:
                connection.execute("BEGIN IMMEDIATE")
                encoding: str = connection.execute("PRAGMA encoding").fetchone()[0]
                sources = self._sources(connection, encoding, run_id)
                previous = self._read(
                    connection,
                    encoding,
                    run_id,
                    sources,
                    annotation_id=payload.annotation_id,
                    fetch_limit=1,
                )
                if previous:
                    saved = previous[0]
                    if any(
                        getattr(saved, field) != getattr(payload, field)
                        for field in AnnotationSubmission.model_fields
                    ):
                        raise AnnotationError(
                            "annotation_id_conflict",
                            "This annotation ID already exists for this run "
                            "with different content.",
                        )
                    _bounded_response(saved)
                    return saved, False
                _anchor(payload, sources, saved=False)
                connection.execute(
                    """INSERT INTO evidence_annotations (
                        schema_version, run_id, annotation_id, document_id, chunk_id,
                        source_text_sha256, start, end, quote, note, created_at
                    ) VALUES (
                        '1.0', :run_id, :annotation_id, :document_id, :chunk_id,
                        :source_text_sha256, :start, :end, :quote, :note,
                        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                    )""",
                    {**payload.model_dump(mode="json"), "run_id": run_id},
                )
                inserted = self._read(
                    connection,
                    encoding,
                    run_id,
                    sources,
                    annotation_id=payload.annotation_id,
                    fetch_limit=1,
                )
                if len(inserted) != 1:
                    raise _invalid_record()
                saved = inserted[0]
                _bounded_response(saved)
                return saved, True
        except sqlite3.Error as exc:
            raise _unavailable() from exc

    def list_annotations(
        self, run_id: str, *, limit: int = DEFAULT_PAGE_SIZE, cursor: int | None = None
    ) -> AnnotationPage:
        """Read evidence and a bounded page (including lookahead) in one read-only snapshot."""
        try:
            run_id = _IDENTITY.validate_python(run_id)
            options = _PageOptions(limit=limit, cursor=cursor)
        except ValidationError as exc:
            raise _invalid_request() from exc
        try:
            with closing(self._connect()) as connection, connection:
                connection.execute("BEGIN")
                encoding: str = connection.execute("PRAGMA encoding").fetchone()[0]
                sources = self._sources(connection, encoding, run_id)
                records = self._read(
                    connection,
                    encoding,
                    run_id,
                    sources,
                    cursor=options.cursor,
                    fetch_limit=options.limit + 1,
                )
                page = AnnotationPage(
                    annotations=records[: options.limit],
                    next_cursor=records[options.limit - 1].sequence
                    if len(records) > options.limit
                    else None,
                )
                _bounded_response(page)
                return page
        except sqlite3.Error as exc:
            raise _unavailable() from exc
