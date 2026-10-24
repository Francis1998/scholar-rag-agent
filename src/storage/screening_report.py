"""Portable human progress reports derived from one complete screening export."""

import hashlib
import json
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from agent.markdown import literal_block
from retrieval.scope import MAX_DOCUMENT_IDS
from storage import paper_screening
from storage.document_catalog import Identity
from storage.paper_collections import CollectionError, CollectionId, Revision
from storage.paper_screening import (
    Reason,
    ScreeningDownload,
    ScreeningExport,
    SQLitePaperScreening,
)

MAX_REPORT_BYTES = 1_048_576
ScreeningReportFormat = Literal["json", "markdown"]
_RESOLVED = ("include", "exclude")
_NEEDS_REVIEW = ("unscreened", "unsure", "stale")
_NOTES = (
    "Resolved means a current include or exclude decision, not scientific validation, "
    "quality assessment or final acceptance.",
    "Still needing human review means unscreened, unsure or stale. Earlier include/exclude "
    "opinions remain provenance, never active decisions.",
    "This is not PRISMA: there are no identified-study totals, deduplication, inferred stages, "
    "recall estimates or stopping recommendations. Every current member remains visible.",
    "Reasons are grouped by exact Unicode strings, preserving case, whitespace and newlines. "
    "Groups follow first current exclusion in saved membership order; member IDs keep that order.",
    "One complete export supplies all counts and reasons. A collection revision binds its "
    "name and membership, not later decision edits. Corpus contents are not frozen. "
    "Separate downloads may differ at the same revision; the snapshot SHA-256 identifies "
    "the exported values, not a signed audit record.",
    "Review before sharing: names, labels, IDs and human reasons may be sensitive. "
    "No authenticated reviewer identity or consensus is inferred.",
)


def _invalid_snapshot() -> CollectionError:
    return CollectionError(
        "invalid_screening_snapshot",
        "The complete screening snapshot is invalid or inconsistent; no report was produced.",
        409,
    )


def _validated_snapshot(snapshot: ScreeningExport) -> tuple[ScreeningExport, bytes]:
    try:
        original = snapshot.model_dump()
        result = ScreeningExport.model_validate(original, strict=True)
        if result.model_dump() != original:
            raise ValueError("Snapshot values must not be normalized.")
        result.collection_name.encode("utf-8")
        counts = dict.fromkeys((*_RESOLVED, *_NEEDS_REVIEW), 0)
        identifiers = []
        included = []
        for item in result.items:
            item.document_id.encode("utf-8")
            for label in (item.title, item.source):
                label.encode("utf-8")
                if "\0" in label:
                    raise ValueError("Labels must not contain NUL.")
            identifiers.append(item.document_id)
            review = item.review
            expected = "unscreened"
            if review is not None:
                if (
                    review.collection_id != result.collection_id
                    or review.document_id != item.document_id
                    or review.collection_revision > result.collection_revision
                ):
                    raise ValueError("Review provenance must match its member and collection.")
                expected = (
                    review.decision
                    if review.collection_revision == result.collection_revision
                    else "stale"
                )
            if item.status != expected:
                raise ValueError("Current status does not match the recorded review.")
            counts[item.status] += 1
            if item.status == "include":
                included.append(item.document_id)
        if (
            len(identifiers) != len(set(identifiers))
            or len(identifiers) != result.total_documents
            or counts != result.counts.model_dump()
            or tuple(included) != result.included_document_ids
        ):
            raise ValueError("Snapshot membership, counts and include selection must agree.")
        content = (result.model_dump_json(indent=2) + "\n").encode("utf-8")
    except (ValidationError, ValueError, TypeError, UnicodeError) as exc:
        raise _invalid_snapshot() from exc
    if len(content) > paper_screening.MAX_EXPORT_BYTES:
        raise CollectionError(
            "screening_export_too_large",
            "The complete screening snapshot exceeds its UTF-8 byte limit; no partial report.",
            413,
        )
    return result, content


def _summary(snapshot: ScreeningExport, content: bytes) -> dict[str, object]:
    resolved: list[str] = []
    needs_review: list[str] = []
    groups: dict[str, list[str]] = {}
    for item in snapshot.items:
        (resolved if item.status in _RESOLVED else needs_review).append(item.document_id)
        if item.status == "exclude" and item.review is not None:
            groups.setdefault(item.review.reason, []).append(item.document_id)
    return {
        "snapshot_sha256": hashlib.sha256(content).hexdigest(),
        "resolved_documents": len(resolved),
        "needs_review_documents": len(needs_review),
        "resolved_document_ids": tuple(resolved),
        "needs_review_document_ids": tuple(needs_review),
        "exclusion_reasons": tuple(
            {"reason": reason, "count": len(ids), "document_ids": tuple(ids)}
            for reason, ids in groups.items()
        ),
    }


def _bounded(content: str) -> str:
    if len(content.encode("utf-8")) > MAX_REPORT_BYTES:
        raise CollectionError(
            "screening_report_too_large",
            "The complete serialized report exceeds its UTF-8 byte limit; no partial report.",
            413,
        )
    return content


def _json(value: BaseModel) -> str:
    return json.dumps(value.model_dump(mode="json"), ensure_ascii=False, indent=2, allow_nan=False)


class ExclusionReasonGroup(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    reason: Reason
    count: int = Field(strict=True, ge=1, le=MAX_DOCUMENT_IDS)
    document_ids: tuple[Identity, ...] = Field(min_length=1, max_length=MAX_DOCUMENT_IDS)


class ScreeningReport(BaseModel):
    """The original export plus auditable partitions; no export-time clock or new opinions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    snapshot: ScreeningExport
    snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    resolved_statuses: tuple[Literal["include"], Literal["exclude"]] = ("include", "exclude")
    needs_review_statuses: tuple[Literal["unscreened"], Literal["unsure"], Literal["stale"]] = (
        "unscreened",
        "unsure",
        "stale",
    )
    reason_grouping: Literal["exact_string_first_occurrence"] = "exact_string_first_occurrence"
    resolved_documents: int = Field(strict=True, ge=0, le=MAX_DOCUMENT_IDS)
    needs_review_documents: int = Field(strict=True, ge=0, le=MAX_DOCUMENT_IDS)
    resolved_document_ids: tuple[Identity, ...] = Field(max_length=MAX_DOCUMENT_IDS)
    needs_review_document_ids: tuple[Identity, ...] = Field(max_length=MAX_DOCUMENT_IDS)
    exclusion_reasons: tuple[ExclusionReasonGroup, ...] = Field(max_length=MAX_DOCUMENT_IDS)

    @model_validator(mode="after")
    def consistent(self) -> Self:
        snapshot, content = _validated_snapshot(self.snapshot)
        expected = _summary(snapshot, content)
        if self.model_dump(include=set(expected)) != expected:
            raise ValueError(
                "Report totals, reason groups and digest must match the whole snapshot."
            )
        return self

    def _validated(self) -> Self:
        # Frozen models can still contain mutable nested lists or unchecked model_copy updates.
        try:
            return type(self).model_validate(self.model_dump(), strict=True)
        except (ValidationError, ValueError, TypeError) as exc:
            raise CollectionError(
                "invalid_screening_report",
                "Saved report fields are invalid or inconsistent; no download was produced.",
                409,
            ) from exc

    def to_json(self) -> str:
        return _bounded(_json(self._validated()) + "\n")

    def to_markdown(self) -> str:
        report = self._validated()
        snapshot = report.snapshot
        sections = [
            "# Human screening progress report\n",
            "Version 1.0. Saved human opinions, not automatic labels.\n",
            *[note + "\n" for note in _NOTES],
            "## Exact collection and snapshot\n",
            literal_block(
                json.dumps(
                    {
                        "collection_id": snapshot.collection_id,
                        "collection_name": snapshot.collection_name,
                        "collection_revision": snapshot.collection_revision,
                        "snapshot_sha256": report.snapshot_sha256,
                    },
                    ensure_ascii=True,
                    indent=2,
                ),
                "json",
            ),
            "## Current status of every member\n",
            "| Status | Members |\n| --- | ---: |\n"
            + "\n".join(
                f"| {status} | {count} |" for status, count in snapshot.counts.model_dump().items()
            )
            + f"\n| Total | {snapshot.total_documents} |\n",
            f"Resolved (current include + exclude): **{report.resolved_documents}**.\n",
            literal_block(
                json.dumps(report.resolved_document_ids, ensure_ascii=True, indent=2), "json"
            ),
            "Still needing human review (unscreened + unsure + stale): "
            f"**{report.needs_review_documents}**.\n",
            literal_block(
                json.dumps(report.needs_review_document_ids, ensure_ascii=True, indent=2), "json"
            ),
            "## Exclusion reasons: current excludes only\n",
        ]
        if not report.exclusion_reasons:
            sections.append("No current exclusions. Earlier stale opinions are not counted.\n")
        for index, group in enumerate(report.exclusion_reasons, 1):
            sections.extend(
                [
                    f"### Reason group {index}: {group.count} member(s)\n",
                    literal_block(group.reason),
                    literal_block(
                        json.dumps(group.document_ids, ensure_ascii=True, indent=2), "json"
                    ),
                ]
            )
        sections.extend(
            [
                "## Complete source snapshot for audit\n",
                "Exact IDs, statuses, original reasons/timestamps/revisions and title/source "
                "prefixes with their original truncation flags. JSON escapes control characters; "
                "no member, reason or opinion is dropped. No paper bodies or history are added.\n",
                literal_block(_json(snapshot), "json"),
            ]
        )
        return _bounded("\n".join(sections))


class _ReportOptions(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    collection_id: CollectionId
    collection_revision: Revision
    format: ScreeningReportFormat = "json"


def build_screening_report(snapshot: ScreeningExport) -> ScreeningReport:
    """Pure, strict derivation from a complete export; no files, models, network or database."""
    validated, content = _validated_snapshot(snapshot)
    return ScreeningReport.model_validate({"snapshot": validated, **_summary(validated, content)})


def export_screening_report(
    screening: SQLitePaperScreening,
    collection_id: str,
    *,
    collection_revision: int,
    format: ScreeningReportFormat = "json",
) -> ScreeningDownload:
    """Reuse the existing complete JSON export once; never combine separate queue/history reads."""
    options = _ReportOptions(
        collection_id=collection_id, collection_revision=collection_revision, format=format
    )
    download = screening.export_results(
        options.collection_id, collection_revision=options.collection_revision, format="json"
    )
    report = build_screening_report(ScreeningExport.model_validate_json(download.content))
    content = report.to_json() if options.format == "json" else report.to_markdown()
    extension = "json" if options.format == "json" else "md"
    return ScreeningDownload(
        content=content.encode("utf-8"),
        media_type="application/json"
        if options.format == "json"
        else "text/markdown; charset=utf-8",
        filename=f"screening-report-{options.collection_id}-r{options.collection_revision}.{extension}",
    )
