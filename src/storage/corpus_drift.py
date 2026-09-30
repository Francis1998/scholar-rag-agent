"""Bounded exact inspection of frozen evidence against one persisted corpus snapshot."""

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from agent.evidence import SHA256, CaptureLimits, EvidenceBundle, text_digest
from retrieval.models import Chunk
from storage.evidence_export import MAX_EVENT_BYTES as MAX_EVENT_BYTES
from storage.evidence_export import MAX_RUN_BYTES as MAX_RUN_BYTES
from storage.evidence_export import MAX_RUN_EVENTS as MAX_RUN_EVENTS
from storage.evidence_export import BoundedRunEvents, EvidenceExporter, EvidenceExportError

MAX_CURRENT_BYTES = CaptureLimits().max_snapshot_bytes
MAX_CURRENT_TEXT_BYTES = CaptureLimits().max_context_bytes
MAX_ID_CHARACTERS = 256
DriftIdentity = Annotated[str, Field(strict=True, min_length=1, max_length=MAX_ID_CHARACTERS)]
ChangedField = Literal["text", "title", "source", "metadata"]
MissingReason = Literal["document_missing", "chunk_missing", "chunk_reassigned"]
_IDENTITY = TypeAdapter(DriftIdentity)
_FIELDS: tuple[ChangedField, ...] = ("text", "title", "source", "metadata")


class ChunkDigests(BaseModel):
    """Exact UTF-8 field hashes; metadata uses canonical JSON, not raw stored spacing."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text_sha256: SHA256
    title_sha256: SHA256
    source_sha256: SHA256
    metadata_sha256: SHA256

    @classmethod
    def from_chunk(cls, chunk: Chunk) -> "ChunkDigests":
        return cls(
            text_sha256=text_digest(chunk.text),
            title_sha256=text_digest(chunk.title),
            source_sha256=text_digest(chunk.source),
            metadata_sha256=text_digest(
                json.dumps(
                    chunk.metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
            ),
        )


class SourceDrift(BaseModel):
    """One frozen rank, never a new retrieval score or scientific validity judgment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rank: int = Field(ge=1, le=50)
    chunk_id: DriftIdentity
    document_id: DriftIdentity
    status: Literal["unchanged", "changed", "missing"]
    changed_fields: list[ChangedField] = Field(max_length=4)
    missing_reason: MissingReason | None
    frozen: ChunkDigests
    current: ChunkDigests | None


class DriftCounts(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    total: int = Field(ge=0, le=50)
    unchanged: int = Field(ge=0, le=50)
    changed: int = Field(ge=0, le=50)
    missing: int = Field(ge=0, le=50)


class CorpusDriftReport(BaseModel):
    """A deterministic, read-only report for one validated completed run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    run_id: DriftIdentity
    context_sha256: SHA256
    counts: DriftCounts
    has_drift: bool
    sources: list[SourceDrift] = Field(max_length=50)
    notices: list[str]


class CorpusDriftError(EvidenceExportError):
    """An explicit bounded-read failure, safe for API responses and code-only logging."""


def _limit_exceeded() -> CorpusDriftError:
    return CorpusDriftError(
        "drift_limit_exceeded",
        "Saved events or selected current chunks exceed the corpus-drift read limits.",
    )


def _invalid_run() -> CorpusDriftError:
    return CorpusDriftError(
        "invalid_run_record", "Saved evidence is inconsistent, invalid, or an unsupported version."
    )


def _invalid_corpus() -> CorpusDriftError:
    return CorpusDriftError(
        "invalid_corpus_record", "A selected current chunk record is invalid or unsupported."
    )


_CHUNK_SIZES_SQL = """
SELECT typeof(chunk_id) != 'text' OR typeof(document_id) != 'text'
           OR typeof(title) != 'text' OR typeof(text) != 'text'
           OR typeof(source) != 'text' OR typeof(metadata) != 'text' AS invalid_record,
       length(CAST(text AS BLOB)) AS text_bytes,
       length(CAST(chunk_id AS BLOB)) + length(CAST(document_id AS BLOB))
           + length(CAST(title AS BLOB)) + length(CAST(text AS BLOB))
           + length(CAST(source AS BLOB)) + length(CAST(metadata AS BLOB)) AS record_bytes
FROM chunks WHERE chunk_id = ? LIMIT 2
"""
_CHUNK_SQL = """
SELECT CAST(chunk_id AS BLOB) AS chunk_id, CAST(document_id AS BLOB) AS document_id,
       CAST(title AS BLOB) AS title, CAST(text AS BLOB) AS text,
       CAST(source AS BLOB) AS source, CAST(metadata AS BLOB) AS metadata
FROM chunks WHERE chunk_id = ? LIMIT 1
"""


def _decode(value: bytes, encoding: str) -> str:
    return value.decode(encoding)


def _metadata_pairs(pairs: list[tuple[str, object]]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for key, value in pairs:
        if key in metadata or not isinstance(value, str):
            raise _invalid_corpus()
        metadata[key] = value
    return metadata


class SQLiteCorpusDrift:
    """Read an existing database, with no container, settings, retriever, or model dependency."""

    def __init__(self, database_path: Path | str) -> None:
        self._database_uri = Path(database_path).resolve().as_uri() + "?mode=ro"

    def report(self, run_id: str) -> CorpusDriftReport:
        """Validate frozen evidence, then compare only its identities in one read transaction."""
        try:
            _IDENTITY.validate_python(run_id)
            run_id.encode("utf-8")
        except (ValidationError, UnicodeError) as exc:
            raise CorpusDriftError(
                "invalid_drift_request", "The run ID must contain 1 to 256 valid characters.", 422
            ) from exc
        try:
            with closing(sqlite3.connect(self._database_uri, uri=True, timeout=5)) as connection:
                connection.row_factory = sqlite3.Row
                connection.execute("BEGIN")
                encoding: str = connection.execute("PRAGMA encoding").fetchone()[0]
                bundle = self._bundle(connection, encoding, run_id)
                sources = self._compare(connection, encoding, bundle)
        except sqlite3.Error as exc:
            raise CorpusDriftError(
                "drift_storage_unavailable", "Saved evidence or corpus storage is unavailable.", 503
            ) from exc
        counts = DriftCounts(
            total=len(sources),
            unchanged=sum(source.status == "unchanged" for source in sources),
            changed=sum(source.status == "changed" for source in sources),
            missing=sum(source.status == "missing" for source in sources),
        )
        return CorpusDriftReport(
            run_id=run_id,
            context_sha256=bundle.snapshot.context_sha256,
            counts=counts,
            has_drift=bool(counts.changed or counts.missing),
            sources=sources,
            notices=[
                "Exact retrieved-chunk comparison only, not whole-paper equality, "
                "scientific validity, entailment, or an instruction to regenerate.",
                "Document rows are checked for identity presence only. New/unretrieved chunks, "
                "document bodies, retrieval indexes, scores, and external sources are not checked.",
                "Review before sharing: identifiers and digests can be sensitive and are not "
                "signatures. No query, answer, passage, title, source label, or metadata "
                "is returned.",
                "One SQLite read snapshot per request; later requests may observe corpus changes. "
                "No events, saved evidence, or corpus data are written.",
            ],
        )

    @staticmethod
    def _bundle(connection: sqlite3.Connection, encoding: str, run_id: str) -> EvidenceBundle:
        try:
            bundle = EvidenceExporter(BoundedRunEvents(connection, encoding)).export(run_id)
            for source in bundle.snapshot.sources:
                _IDENTITY.validate_python(source.chunk.chunk_id)
                _IDENTITY.validate_python(source.chunk.document_id)
                ChunkDigests.from_chunk(source.chunk)
            return bundle
        except EvidenceExportError as exc:
            if exc.code == "evidence_read_limit_exceeded":
                raise _limit_exceeded() from exc
            if exc.code == "invalid_run_record":
                raise _invalid_run() from exc
            raise
        except (ValidationError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
            raise _invalid_run() from exc

    @staticmethod
    def _current(
        connection: sqlite3.Connection,
        encoding: str,
        chunk_id: str,
        remaining_bytes: int,
        remaining_text_bytes: int,
    ) -> tuple[Chunk | None, int, int]:
        sizes = connection.execute(_CHUNK_SIZES_SQL, (chunk_id,)).fetchall()
        if not sizes:
            return None, 0, 0
        if len(sizes) != 1 or sizes[0]["invalid_record"]:
            raise _invalid_corpus()
        factor = 1 if encoding == "UTF-8" else 2
        if (
            sizes[0]["record_bytes"] > factor * remaining_bytes
            or sizes[0]["text_bytes"] > factor * remaining_text_bytes
        ):
            raise _limit_exceeded()
        row = connection.execute(_CHUNK_SQL, (chunk_id,)).fetchone()
        try:
            record = {
                field: _decode(row[field], encoding)
                for field in ("chunk_id", "document_id", "title", "text", "source", "metadata")
            }
            record_bytes = sum(len(value.encode("utf-8")) for value in record.values())
            text_bytes = len(record["text"].encode("utf-8"))
            if record_bytes > remaining_bytes or text_bytes > remaining_text_bytes:
                raise _limit_exceeded()
            metadata = json.loads(record["metadata"], object_pairs_hook=_metadata_pairs)
            chunk = Chunk.model_validate({**record, "metadata": metadata}, strict=True)
            _IDENTITY.validate_python(chunk.document_id)
            ChunkDigests.from_chunk(chunk)
        except (ValidationError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
            raise _invalid_corpus() from exc
        return chunk, record_bytes, text_bytes

    def _compare(
        self, connection: sqlite3.Connection, encoding: str, bundle: EvidenceBundle
    ) -> list[SourceDrift]:
        sources = []
        remaining_bytes: int = MAX_CURRENT_BYTES
        remaining_text_bytes: int = MAX_CURRENT_TEXT_BYTES
        for source in bundle.snapshot.sources:
            chunk = source.chunk
            current, record_bytes, text_bytes = self._current(
                connection, encoding, chunk.chunk_id, remaining_bytes, remaining_text_bytes
            )
            remaining_bytes -= record_bytes
            remaining_text_bytes -= text_bytes
            exists = connection.execute(
                "SELECT 1 FROM documents WHERE document_id = ? LIMIT 2", (chunk.document_id,)
            ).fetchall()
            if len(exists) > 1:
                raise _invalid_corpus()
            reason: MissingReason | None = None
            if not exists:
                reason = "document_missing"
            elif current is None:
                reason = "chunk_missing"
            elif current.document_id != chunk.document_id:
                reason = "chunk_reassigned"
            frozen = ChunkDigests.from_chunk(chunk)
            latest = (
                ChunkDigests.from_chunk(current) if current is not None and not reason else None
            )
            changed = [
                field
                for field in _FIELDS
                if latest is not None
                and getattr(frozen, f"{field}_sha256") != getattr(latest, f"{field}_sha256")
            ]
            sources.append(
                SourceDrift(
                    rank=source.rank,
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    status="missing" if reason else "changed" if changed else "unchanged",
                    changed_fields=changed,
                    missing_reason=reason,
                    frozen=frozen,
                    current=latest,
                )
            )
        return sources
