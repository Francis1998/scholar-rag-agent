"""Complete, bounded human-review downloads over one read-only SQLite snapshot."""

import json
import sqlite3
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from agent.evidence import EvidenceBundle, text_digest
from agent.markdown import literal_block
from agent.review_models import AnswerReviewError, SavedAnswerReview, invalid_review_record
from storage.answer_reviews import _read_record, ensure_saved_references
from storage.evidence_annotations import _RECORDS_SQL as _ANNOTATION_RECORDS
from storage.evidence_annotations import _SIZES_SQL as _ANNOTATION_SIZES
from storage.evidence_annotations import (
    MAX_RECORD_TEXT_BYTES,
    SavedEvidenceAnnotation,
    read_annotation_record,
)
from storage.evidence_export import (
    BoundedRunEvents,
    EvidenceExporter,
    EvidenceExportError,
    SQLiteEvidenceReader,
)
from storage.run_history import Identity

MAX_NOTEBOOK_REVIEWS = 1000
MAX_NOTEBOOK_ANNOTATIONS = 1000
MAX_REVIEW_RECORD_BYTES = 262144
MAX_HISTORY_BYTES = 4 * 1024 * 1024
MAX_EXPORT_BYTES = 8 * 1024 * 1024
NotebookFormat = Literal["json", "markdown"]
_IDENTITY = TypeAdapter(Identity)
_WARNINGS = (
    "Human opinions are not scientific validation. Review decisions do not change model "
    "grounding, which records token overlap, not truth or entailment.",
    "UUIDs identify records, not authenticated reviewers. Different opinions coexist; "
    "no consensus or latest verdict is inferred.",
    "All retained reviews and annotations for this run are read in a single read-only SQLite "
    "snapshot. Later appends require a new export. Sequences are original and newest-first "
    "within each history, not a shared cross-history clock.",
    "Review before sharing: original queries, full captured passages, source metadata, "
    "generated answers and human comments may be sensitive. This is not a signed audit record.",
)
_REVIEW_SIZES = """
SELECT typeof(id) != 'integer' OR typeof(schema_version) != 'text'
           OR typeof(run_id) != 'text' OR typeof(review_id) != 'text'
           OR typeof(created_at) != 'text' OR typeof(decision) != 'text'
           OR typeof(comment) != 'text' OR typeof(cited_chunk_ids) != 'text' AS invalid_record,
       length(CAST(schema_version AS BLOB)) + length(CAST(run_id AS BLOB))
           + length(CAST(review_id AS BLOB)) + length(CAST(created_at AS BLOB))
           + length(CAST(decision AS BLOB)) + length(CAST(comment AS BLOB))
           + length(CAST(cited_chunk_ids AS BLOB)) AS text_bytes
FROM answer_reviews
"""
_REVIEW_RECORDS = """
SELECT id AS sequence, CAST(schema_version AS BLOB) AS schema_version,
       CAST(run_id AS BLOB) AS run_id, CAST(review_id AS BLOB) AS review_id,
       CAST(created_at AS BLOB) AS created_at, CAST(decision AS BLOB) AS decision,
       CAST(comment AS BLOB) AS comment, CAST(cited_chunk_ids AS BLOB) AS cited_chunk_ids
FROM answer_reviews
"""


class NotebookExportError(EvidenceExportError):
    """Sanitized notebook failure, shared by Python and HTTP."""


def _too_large() -> NotebookExportError:
    return NotebookExportError(
        "notebook_export_too_large",
        "The complete notebook exceeds a fixed record or UTF-8 byte limit; no partial export.",
        413,
    )


def _invalid(kind: Literal["review", "annotation", "run"]) -> NotebookExportError:
    return NotebookExportError(
        f"invalid_{kind}_record",
        "Saved notebook records are invalid, inconsistent, or unsupported.",
    )


def _bounded(content: str) -> str:
    if len(content.encode("utf-8")) > MAX_EXPORT_BYTES:
        raise _too_large()
    return content


def _json(value: BaseModel) -> str:
    return json.dumps(
        value.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    )


class ReviewNotebook(BaseModel):
    """Frozen evidence and separate complete opinions; no inferred author or combined verdict."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    consistency: Literal["single_sqlite_read_transaction"] = "single_sqlite_read_transaction"
    history_order: Literal["sequence_desc"] = "sequence_desc"
    evidence: EvidenceBundle
    reviews: list[SavedAnswerReview] = Field(max_length=MAX_NOTEBOOK_REVIEWS)
    annotations: list[SavedEvidenceAnnotation] = Field(max_length=MAX_NOTEBOOK_ANNOTATIONS)
    warnings: tuple[str, ...] = _WARNINGS

    def to_json(self) -> str:
        """Complete UTF-8 JSON, preserving the existing nested evidence/opinion schemas."""
        return _bounded(_json(self) + "\n")

    def to_markdown(self) -> str:
        """All dynamic data stays in literal fences, including IDs, metadata and opinions."""
        sections = [
            "# Portable human-review notebook\n",
            "Version 1.0. Human opinions, not scientific validation.\n",
            literal_block("\n".join(self.warnings)),
            "## Original query\n",
            literal_block(self.evidence.query),
            "## Frozen answer (model output, not a human verdict)\n",
            literal_block(self.evidence.answer.answer),
            "## Exact frozen passages\n",
        ]
        for source in self.evidence.snapshot.sources:
            sections.extend(
                [f"### Evidence rank {source.rank}\n", literal_block(source.chunk.text)]
            )
        sections.extend(
            [
                "## Complete frozen evidence and provenance\n",
                "The existing evidence bundle is preserved in full below: query, plan, answer, "
                "claims, citations, exact request, sources with metadata/ranks/digests, generation "
                "provenance and allowlisted event trace. Unknown event payloads remain omitted "
                "under the evidence bundle's privacy contract, not reconstructed.\n",
                literal_block(_json(self.evidence), "json"),
                "## Complete answer-review history\n",
                "Newest original sequence first. UUIDs identify records, not authenticated people. "
                "Accepted, needs_revision and rejected remain separate recorded opinions.\n",
            ]
        )
        if not self.reviews:
            sections.append("No answer reviews recorded in this snapshot.\n")
        for review in self.reviews:
            sections.extend(
                [
                    f"### Review sequence {review.sequence}\n",
                    literal_block(_json(review), "json"),
                    literal_block(review.comment),
                ]
            )
        sections.extend(
            [
                "## Complete exact-quote annotation history\n",
                "Offsets are half-open Unicode code-point indices into the captured source, "
                "not UTF-8 bytes, normalized text or a search for the first matching quote.\n",
            ]
        )
        if not self.annotations:
            sections.append("No evidence annotations recorded in this snapshot.\n")
        for annotation in self.annotations:
            sections.extend(
                [
                    f"### Annotation sequence {annotation.sequence}\n",
                    literal_block(_json(annotation), "json"),
                    literal_block(f"Exact span: [{annotation.start}, {annotation.end})"),
                    literal_block(annotation.quote),
                    literal_block(annotation.note),
                ]
            )
        return _bounded("\n".join(sections))

    def filename(self, format: NotebookFormat) -> str:
        extension = "json" if format == "json" else "md"
        return f"review-notebook-{text_digest(self.evidence.run_id)[:16]}.{extension}"


def _history(
    connection: sqlite3.Connection,
    encoding: str,
    run_id: str,
    kind: Literal["review", "annotation"],
    remaining_bytes: int,
) -> tuple[list[dict[str, object]], int]:
    """Preflight bounded row metadata before materializing any untrusted opinion text."""
    if kind == "review":
        sizes_sql, records_sql = _REVIEW_SIZES, _REVIEW_RECORDS
        maximum, record_bytes, sequence = MAX_NOTEBOOK_REVIEWS, MAX_REVIEW_RECORD_BYTES, "id"
    else:
        sizes_sql, records_sql = _ANNOTATION_SIZES, _ANNOTATION_RECORDS
        maximum, record_bytes, sequence = (
            MAX_NOTEBOOK_ANNOTATIONS,
            MAX_RECORD_TEXT_BYTES,
            "sequence",
        )
    selector = f" WHERE run_id = ? ORDER BY {sequence} DESC LIMIT ?"
    sizes = connection.execute(sizes_sql + selector, (run_id, maximum + 1)).fetchall()
    if len(sizes) > maximum:
        raise _too_large()
    if any(row["invalid_record"] for row in sizes):
        raise _invalid(kind)
    factor = 1 if encoding == "UTF-8" else 2
    if (
        any(row["text_bytes"] > factor * record_bytes for row in sizes)
        or sum(row["text_bytes"] for row in sizes) > factor * remaining_bytes
    ):
        raise _too_large()
    rows = connection.execute(records_sql + selector, (run_id, maximum)).fetchall()
    records: list[dict[str, object]] = []
    total = 0
    for row in rows:
        try:
            record = {
                key: value.decode(encoding) if isinstance(value, bytes) else value
                for key, value in dict(row).items()
            }
            size = sum(
                len(value.encode("utf-8")) for value in record.values() if isinstance(value, str)
            )
        except UnicodeError as exc:
            raise _invalid(kind) from exc
        total += size
        if size > record_bytes or total > remaining_bytes:
            raise _too_large()
        records.append(record)
    return records, total


class SQLiteReviewNotebook:
    """Read existing tables only; construction performs no I/O or schema initialization."""

    def __init__(self, database_path: Path | str) -> None:
        self._reader = SQLiteEvidenceReader(database_path)

    def export(self, run_id: str) -> ReviewNotebook:
        """Require the entire run's histories and both representations to fit, or fail."""
        try:
            _IDENTITY.validate_python(run_id)
            run_id.encode("utf-8")
        except (ValidationError, UnicodeError) as exc:
            raise NotebookExportError(
                "invalid_notebook_export_request",
                "Use an exact run ID containing 1 to 256 valid Unicode characters.",
                422,
            ) from exc
        try:
            with self._reader.snapshot() as (connection, encoding):
                bundle = EvidenceExporter(BoundedRunEvents(connection, encoding)).export(run_id)
                review_rows, used = _history(
                    connection, encoding, run_id, "review", MAX_HISTORY_BYTES
                )
                annotation_rows, _ = _history(
                    connection, encoding, run_id, "annotation", MAX_HISTORY_BYTES - used
                )
                sources = {source.chunk.chunk_id: source for source in bundle.snapshot.sources}
                frozen_ids = frozenset(sources)
                reviews = [_read_record(row) for row in review_rows]
                for review in reviews:
                    if review.run_id != run_id:
                        raise invalid_review_record()
                    ensure_saved_references(review, frozen_ids)
                annotations = [
                    read_annotation_record(row, run_id, sources) for row in annotation_rows
                ]
            result = ReviewNotebook(evidence=bundle, reviews=reviews, annotations=annotations)
            result.to_json()
            result.to_markdown()
            return result
        except AnswerReviewError as exc:
            raise NotebookExportError(exc.code, str(exc), exc.status_code) from exc
        except EvidenceExportError as exc:
            if exc.code == "evidence_read_limit_exceeded":
                raise _too_large() from exc
            raise
        except sqlite3.Error as exc:
            raise NotebookExportError(
                "notebook_storage_unavailable", "Saved notebook storage is unavailable.", 503
            ) from exc
        except (ValidationError, UnicodeError, ValueError, RecursionError) as exc:
            raise _invalid("run") from exc
