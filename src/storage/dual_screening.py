"""Independent multi-reviewer screening: pseudonymous labels, blinding and resolutions.

Existing unlabeled decisions are never moved, copied or rewritten. They are read in place
as the reserved reviewer ``default``. Named reviewers, conflict resolutions and protocol
changes use four additive tables created at startup; nothing existing is backfilled.
"""

import csv
import io
import json
import re
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from agent.markdown import literal_block
from retrieval.scope import MAX_DOCUMENT_IDS
from storage.csv_export import csv_cell
from storage.document_catalog import (
    MAX_SOURCE_CHARACTERS,
    MAX_TITLE_CHARACTERS,
    DocumentCatalogError,
    Identity,
    _prefix,
)
from storage.paper_collections import (
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    MAX_REVISION,
    CollectionError,
    CollectionId,
    CollectionName,
    PaperCollection,
    Revision,
)
from storage.paper_screening import (
    MAX_RECORD_CHARACTERS,
    Decision,
    Reason,
    ScreeningCounts,
    ScreeningDownload,
    ScreeningItem,
    ScreeningQueue,
    ScreeningReview,
    ScreeningStatus,
    SQLitePaperScreening,
)
from storage.screening_agreement import Z_95, AgreementStatistics, agreement_statistics
from storage.screening_priority import (
    _MEMBERS_SQL,
    MAX_FEATURE_TEXT_CHARACTERS,
    MAX_PRIORITY_RESPONSE_BYTES,
    PriorityMember,
    ScreeningPriority,
    build_priority,
)
from storage.source_context import ContextDocumentIdentity

DEFAULT_REVIEWER = "default"
MAX_NAMED_REVIEWERS = 8
MAX_REVIEWER_CHARACTERS = 32
MAX_REVIEWER_RESPONSE_BYTES = 524_288
MAX_TEAM_RESPONSE_BYTES = 1_048_576
MAX_DOWNLOAD_BYTES = 1_048_576
_CURRENT: tuple[ScreeningStatus, ...] = ("include", "exclude", "unsure")
_TABLES = (
    "paper_screening_reviewer_decisions",
    "paper_screening_resolutions",
    "paper_screening_protocols",
)


def _not_reserved(value: str) -> str:
    if value == DEFAULT_REVIEWER:
        raise ValueError("The label 'default' is reserved for existing unlabeled decisions.")
    return value


ReviewerLabel = Annotated[
    str,
    StringConstraints(
        strict=True,
        min_length=1,
        max_length=MAX_REVIEWER_CHARACTERS,
        pattern=r"^[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?$",
    ),
]
NamedReviewer = Annotated[ReviewerLabel, AfterValidator(_not_reserved)]
FinalDecisionValue = Literal["include", "exclude"]
ConflictType = Literal["decision", "exclusion_reason"]
FinalStatus = Literal["unscreened", "stale", "single_review", "agreed", "conflict", "resolved"]
ResolutionStatus = Literal["none", "current", "outdated"]
AgreementScale = Literal["include_exclude_unsure", "include_or_unsure_vs_exclude"]
TeamExportFormat = Literal["json", "csv"]
AgreementFormat = Literal["json", "markdown", "csv"]
_Count = Annotated[int, Field(strict=True, ge=0, le=MAX_DOCUMENT_IDS)]
_IDENTITY = TypeAdapter(Identity)
_NAMED = TypeAdapter(NamedReviewer)
_SCALES: dict[AgreementScale, tuple[tuple[str, ...], dict[str, str]]] = {
    "include_exclude_unsure": (
        ("include", "exclude", "unsure"),
        {"include": "include", "exclude": "exclude", "unsure": "unsure"},
    ),
    "include_or_unsure_vs_exclude": (
        ("include_or_unsure", "exclude"),
        {"include": "include_or_unsure", "unsure": "include_or_unsure", "exclude": "exclude"},
    ),
}


class ScreeningProtocol(BaseModel):
    """How independent reviews are compared; revision 0 is the unsaved blind default."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_version: Literal["1.0"] = "1.0"
    collection_id: CollectionId
    revision: int = Field(ge=0, le=MAX_REVISION)
    blind: bool
    exclusion_reasons_must_match: bool
    changed_by: NamedReviewer | None
    updated_at: AwareDatetime | None

    @model_validator(mode="after")
    def saved_fields(self) -> Self:
        unsaved = self.revision == 0
        if (
            unsaved != (self.changed_by is None)
            or unsaved != (self.updated_at is None)
            or (unsaved and (not self.blind or self.exclusion_reasons_must_match))
        ):
            raise ValueError("Only saved protocol revisions record a change; default is blind.")
        return self


class ProtocolChange(BaseModel):
    """A complete, revision-checked replacement of a collection's screening protocol."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    blind: bool = Field(strict=True)
    exclusion_reasons_must_match: bool = Field(strict=True)
    expected_protocol_revision: int = Field(strict=True, ge=0, le=MAX_REVISION)
    changed_by: NamedReviewer


class ReviewerReview(ScreeningReview):
    """One named reviewer's own decision; ``blinded`` records server-enforced blinding."""

    reviewer: NamedReviewer
    blinded: bool


class ReviewerSubmission(BaseModel):
    """Zero is a reviewer's first decision on a member, never an unconditional overwrite."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    reviewer: NamedReviewer
    collection_revision: Revision
    expected_decision_revision: int = Field(strict=True, ge=0, le=MAX_REVISION)
    decision: Decision
    reason: Reason


class ReviewerDecision(BaseModel):
    """A reviewer's latest decision as shown where blinding has lifted."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    reviewer: ReviewerLabel
    current: bool
    decision: Decision
    reason: Reason
    revision: Revision
    collection_revision: Revision
    updated_at: AwareDatetime
    blinded: bool | None


class BasisReference(BaseModel):
    """The exact reviewer decision revision a resolution is about."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    reviewer: ReviewerLabel
    revision: Revision


class BasisDecision(BasisReference):
    decision: Decision


def _sorted_unique(labels: Sequence[str]) -> bool:
    return list(labels) == sorted(set(labels))


class ScreeningResolution(BaseModel):
    """An append-only final decision for one disagreement; reviewer decisions stay intact."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_version: Literal["1.0"]
    collection_id: CollectionId
    document_id: Identity
    collection_revision: Revision
    revision: Revision
    resolver: NamedReviewer
    decision: FinalDecisionValue
    rationale: Reason
    conflict_type: ConflictType
    basis: tuple[BasisDecision, ...] = Field(min_length=2, max_length=MAX_NAMED_REVIEWERS + 1)
    updated_at: AwareDatetime

    @model_validator(mode="after")
    def ordered_basis(self) -> Self:
        if not _sorted_unique([entry.reviewer for entry in self.basis]):
            raise ValueError("Basis reviewers must be unique and sorted.")
        return self


class ResolutionSubmission(BaseModel):
    """Echo the conflict's exact basis and latest resolution revision you inspected."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    resolver: NamedReviewer
    collection_revision: Revision
    expected_resolution_revision: int = Field(strict=True, ge=0, le=MAX_REVISION)
    expected_basis: tuple[BasisReference, ...] = Field(
        min_length=2, max_length=MAX_NAMED_REVIEWERS + 1
    )
    decision: FinalDecisionValue
    rationale: Reason


class FinalDecision(BaseModel):
    """The team's current outcome for one member; a current resolution takes precedence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: FinalStatus
    decision: Decision | None
    source: Literal["agreement", "resolution"] | None
    exclusion_reason: Reason | None
    exclusion_reasons_differ: bool
    conflict_type: ConflictType | None
    resolution_status: ResolutionStatus
    resolution: ScreeningResolution | None

    @model_validator(mode="after")
    def consistent(self) -> Self:
        decided = self.status in ("agreed", "resolved")
        if (
            decided != (self.decision is not None)
            or decided != (self.source is not None)
            or (self.status == "resolved") != (self.source == "resolution")
            or (self.status == "resolved") != (self.resolution_status == "current")
            or (self.resolution_status == "none") != (self.resolution is None)
            or (self.status in ("conflict", "resolved")) != (self.conflict_type is not None)
            or (self.exclusion_reason is not None and self.decision != "exclude")
            or (self.exclusion_reasons_differ and self.exclusion_reason is not None)
            or (self.exclusion_reasons_differ and self.status != "agreed")
        ):
            raise ValueError("Final status, source and resolution must agree.")
        return self


class ReviewerItem(BaseModel):
    """A member from one reviewer's side: other reviewers are absent while blinded."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: Identity
    title: str = Field(max_length=MAX_TITLE_CHARACTERS)
    title_truncated: bool
    status: ScreeningStatus
    review: ReviewerReview | None
    others_hidden: bool
    other_decisions: tuple[ReviewerDecision, ...] | None
    final: FinalDecision | None

    @model_validator(mode="after")
    def hidden_means_absent(self) -> Self:
        if self.others_hidden != (self.other_decisions is None) or self.others_hidden != (
            self.final is None
        ):
            raise ValueError("Hidden members expose neither other decisions nor final status.")
        return self


class ReviewerQueue(BaseModel):
    """One reviewer's own statuses for every member; others only where blinding lifted."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    collection_id: CollectionId
    collection_revision: Revision
    reviewer: NamedReviewer
    blind: bool
    collection_blind: bool
    total_documents: int = Field(ge=1, le=MAX_DOCUMENT_IDS)
    counts: ScreeningCounts
    items: list[ReviewerItem] = Field(max_length=MAX_DOCUMENT_IDS)
    next_cursor: Identity | None


class ReviewerHistory(BaseModel):
    """One reviewer's own retained decisions for one member, newest first."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    collection_id: CollectionId
    collection_name: CollectionName
    collection_revision: Revision
    reviewer: NamedReviewer
    document_id: Identity
    current_status: ScreeningStatus
    latest_review: ReviewerReview | None
    items: list[ReviewerReview] = Field(max_length=MAX_PAGE_SIZE)
    next_cursor: Revision | None


class ReviewerMember(BaseModel):
    """One member for one reviewer, with that reviewer's recent own history, in one read."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    collection_id: CollectionId
    collection_name: CollectionName
    collection_revision: Revision
    reviewer: NamedReviewer
    blind: bool
    collection_blind: bool
    item: ReviewerItem
    recent_history: list[ReviewerReview] = Field(max_length=MAX_PAGE_SIZE)
    older_history: bool


class ReviewerPriority(BaseModel):
    """The existing ordering model trained on this reviewer's own current decisions only."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    reviewer: NamedReviewer
    training_scope: Literal["own_current_decisions"] = "own_current_decisions"
    priority: ScreeningPriority


class ReviewerProgress(BaseModel):
    """How many members a reviewer has decided; never which members or which decisions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    reviewer: ReviewerLabel
    labeled: bool
    current_decisions: _Count
    stale_decisions: _Count


class ConflictCounts(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    disagreements: _Count
    resolved: _Count
    unresolved: _Count
    decision: _Count
    exclusion_reason: _Count
    outdated_resolutions: _Count

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if (
            self.resolved + self.unresolved != self.disagreements
            or self.decision + self.exclusion_reason != self.disagreements
        ):
            raise ValueError("Conflict counts must partition every disagreement.")
        return self


class FinalCounts(BaseModel):
    """PRISMA-style counts for one screening stage over current members, not a flow diagram."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    total_documents: int = Field(strict=True, ge=1, le=MAX_DOCUMENT_IDS)
    unscreened: _Count
    stale: _Count
    single_review: _Count
    agreed_include: _Count
    agreed_exclude: _Count
    agreed_unsure: _Count
    conflict: _Count
    resolved_include: _Count
    resolved_exclude: _Count
    screened_at_least_once: _Count
    double_screened: _Count
    final_include: _Count
    final_exclude: _Count
    needs_review: _Count

    @model_validator(mode="after")
    def consistent(self) -> Self:
        double = (
            self.agreed_include
            + self.agreed_exclude
            + self.agreed_unsure
            + self.conflict
            + self.resolved_include
            + self.resolved_exclude
        )
        if (
            self.final_include != self.agreed_include + self.resolved_include
            or self.final_exclude != self.agreed_exclude + self.resolved_exclude
            or self.needs_review
            != self.unscreened
            + self.stale
            + self.single_review
            + self.agreed_unsure
            + self.conflict
            or self.double_screened != double
            or self.screened_at_least_once != double + self.single_review
            or self.total_documents != self.final_include + self.final_exclude + self.needs_review
        ):
            raise ValueError("Final counts must partition every current member.")
        return self


class FinalReasonGroup(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    reason: Reason
    count: int = Field(strict=True, ge=1, le=MAX_DOCUMENT_IDS)
    document_ids: tuple[Identity, ...] = Field(min_length=1, max_length=MAX_DOCUMENT_IDS)


class PairAgreement(BaseModel):
    """Two reviewers on one scale, over members both decided at the current revision."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    reviewer_a: ReviewerLabel
    reviewer_b: ReviewerLabel
    scale: AgreementScale
    statistics: AgreementStatistics
    unblinded_decisions: int = Field(strict=True, ge=0, le=2 * MAX_DOCUMENT_IDS)
    unknown_blinding_decisions: int = Field(strict=True, ge=0, le=2 * MAX_DOCUMENT_IDS)


class AgreementMethod(BaseModel):
    """Exact, versioned description of how agreement is computed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: Literal["cohen-kappa-fce1969-pabak-v1"] = "cohen-kappa-fce1969-pabak-v1"
    decisions: str = (
        "Each reviewer's latest decision recorded at the current collection revision. Stale "
        "decisions are excluded. Resolutions never change agreement statistics."
    )
    scales: str = (
        "include_exclude_unsure compares the three recorded categories (k = 3). "
        "include_or_unsure_vs_exclude counts unsure as include (k = 2), as Covidence does "
        "for title/abstract inter-rater reliability."
    )
    kappa: str = (
        "p_o = sum_i n_ii / n; p_e = sum_i p_i. * p_.i; kappa = (p_o - p_e) / (1 - p_e). "
        "Undefined (null, with a reason) when n = 0 or p_e = 1."
    )
    standard_error: str = (
        "Fleiss, Cohen and Everitt (1969) large-sample SE: Var = [sum_i p_ii (1 - (p_i. + "
        "p_.i)(1 - kappa))^2 + (1 - kappa)^2 sum_{i != j} p_ij (p_.i + p_j.)^2 - (kappa - "
        "p_e (1 - kappa))^2] / (n (1 - p_e)^2)."
    )
    confidence_interval: str = (
        f"Wald 95%: kappa -/+ {Z_95:.6f} * SE, clipped to [-1, 1]. Asymptotic; unreliable "
        "for few shared papers."
    )
    pabak: str = "PABAK = (k * p_o - 1) / (k - 1), with k the scale's category count."
    landis_koch: str = (
        "Landis and Koch (1977): < 0 poor; 0-0.20 slight; 0.21-0.40 fair; 0.41-0.60 "
        "moderate; 0.61-0.80 substantial; 0.81-1.00 almost perfect (upper bounds "
        "inclusive). The authors called these divisions arbitrary; they are not quality "
        "thresholds and depend on prevalence."
    )
    blinding: str = (
        "unblinded_decisions counts compared decisions recorded while blind mode was off "
        "or after that reviewer's earlier decision on the paper; unknown_blinding_decisions "
        "counts unlabeled default decisions. Both can inflate agreement."
    )


_AGREEMENT_NOTES = (
    "Pseudonymous reviewer labels are workflow names, not authenticated identities.",
    "Agreement uses independent reviewer decisions only. Resolutions decide final outcomes "
    "but never change kappa, PABAK or the confusion counts.",
    "Kappa depends on prevalence and marginal distributions; with skewed include rates a "
    "high observed agreement can give a low kappa. PABAK is shown beside it for that reason.",
    "Landis-Koch bands are conventional labels, not validated thresholds for screening.",
    "Counts cover current collection members at this revision: not a PRISMA flow diagram, "
    "deduplication, recall estimate or scientific validation.",
)
_SUMMARY_NOTES = (
    "The sections above describe the unlabeled default reviewer's own decisions, exactly "
    "as before. Team outcomes below take current resolutions first, then agreement of two "
    "or more independent current decisions.",
    *_AGREEMENT_NOTES,
)
_HIDDEN_NOTES = (
    "Blind mode is on: only how many members each reviewer has decided is shown. Decisions, "
    "conflicts, resolutions, final outcomes and agreement stay hidden until blind mode is "
    "turned off through the screening protocol.",
    "The sections above describe the unlabeled default reviewer's own decisions, exactly "
    "as before.",
)
_FINAL_NOTES = (
    "Final outcome precedence: a current resolution first (its reviewer decisions are "
    "unchanged), then agreement of two or more current independent decisions. One decision "
    "is single_review; disagreement without a current resolution is conflict.",
    "Reviewer decisions are listed exactly as recorded; resolutions never rewrite them.",
    *_AGREEMENT_NOTES[:1],
)


class AgreementReport(BaseModel):
    """Independent agreement, conflicts and final-outcome counts from one read snapshot."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    collection_id: CollectionId
    collection_name: CollectionName
    collection_revision: Revision
    protocol: ScreeningProtocol
    reviewers: tuple[ReviewerProgress, ...] = Field(max_length=MAX_NAMED_REVIEWERS + 1)
    pairs: tuple[PairAgreement, ...]
    conflicts: ConflictCounts
    unresolved_conflict_document_ids: tuple[Identity, ...] = Field(max_length=MAX_DOCUMENT_IDS)
    final_counts: FinalCounts
    final_exclusion_reasons: tuple[FinalReasonGroup, ...] = Field(max_length=MAX_DOCUMENT_IDS)
    final_excludes_without_single_reason: tuple[Identity, ...] = Field(max_length=MAX_DOCUMENT_IDS)
    method: AgreementMethod = AgreementMethod()
    notes: tuple[str, ...] = _AGREEMENT_NOTES


class ConflictItem(BaseModel):
    """A disagreement among current independent decisions, with what a resolution echoes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: Identity
    title: str = Field(max_length=MAX_TITLE_CHARACTERS)
    title_truncated: bool
    conflict_type: ConflictType
    decisions: tuple[ReviewerDecision, ...] = Field(
        min_length=2, max_length=MAX_NAMED_REVIEWERS + 1
    )
    expected_basis: tuple[BasisReference, ...] = Field(
        min_length=2, max_length=MAX_NAMED_REVIEWERS + 1
    )
    resolution_revision: int = Field(strict=True, ge=0, le=MAX_REVISION)
    final: FinalDecision


class ScreeningConflicts(BaseModel):
    """Every current disagreement in membership order; unblinded team view only."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    collection_id: CollectionId
    collection_name: CollectionName
    collection_revision: Revision
    protocol: ScreeningProtocol
    reviewers: tuple[ReviewerProgress, ...] = Field(max_length=MAX_NAMED_REVIEWERS + 1)
    total_documents: int = Field(ge=1, le=MAX_DOCUMENT_IDS)
    counts: ConflictCounts
    items: list[ConflictItem] = Field(max_length=MAX_DOCUMENT_IDS)
    next_cursor: Identity | None


class FinalDecisionItem(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: Identity
    title: str = Field(max_length=MAX_TITLE_CHARACTERS)
    title_truncated: bool
    source: str = Field(max_length=MAX_SOURCE_CHARACTERS)
    source_truncated: bool
    decisions: tuple[ReviewerDecision, ...] = Field(max_length=MAX_NAMED_REVIEWERS + 1)
    final: FinalDecision


class FinalDecisionExport(BaseModel):
    """Every current member's final outcome beside every reviewer's latest decision."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    collection_id: CollectionId
    collection_name: CollectionName
    collection_revision: Revision
    protocol: ScreeningProtocol
    reviewers: tuple[ReviewerProgress, ...] = Field(max_length=MAX_NAMED_REVIEWERS + 1)
    final_counts: FinalCounts
    items: list[FinalDecisionItem] = Field(min_length=1, max_length=MAX_DOCUMENT_IDS)
    notes: tuple[str, ...] = _FINAL_NOTES


class DualReviewSummary(BaseModel):
    """The team section of the progress report; decisions stay hidden in blind mode."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    protocol: ScreeningProtocol
    reviewers: tuple[ReviewerProgress, ...] = Field(max_length=MAX_NAMED_REVIEWERS + 1)
    hidden: bool
    final_counts: FinalCounts | None
    conflicts: ConflictCounts | None
    pairs: tuple[PairAgreement, ...] | None
    final_exclusion_reasons: tuple[FinalReasonGroup, ...] | None
    notes: tuple[str, ...]

    @model_validator(mode="after")
    def hidden_means_absent(self) -> Self:
        values = (self.final_counts, self.conflicts, self.pairs, self.final_exclusion_reasons)
        if self.hidden != self.protocol.blind or any(
            (value is None) != self.hidden for value in values
        ):
            raise ValueError("Blind summaries carry progress only; unblinded ones are complete.")
        return self


class TeamOverview(BaseModel):
    """Team progress for any protocol; conflicts and agreement only once unblinded."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    collection_id: CollectionId
    collection_name: CollectionName
    collection_revision: Revision
    summary: DualReviewSummary
    conflicts: tuple[ConflictItem, ...] | None = Field(max_length=MAX_DOCUMENT_IDS)

    @model_validator(mode="after")
    def hidden_means_absent(self) -> Self:
        if self.summary.hidden != (self.conflicts is None):
            raise ValueError("Blind overviews list no conflicts.")
        return self


def _storage_unavailable() -> CollectionError:
    return CollectionError(
        "screening_storage_unavailable", "Screening storage is unavailable.", 503
    )


def _invalid(kind: str) -> CollectionError:
    return CollectionError(
        f"invalid_{kind}_record",
        "Saved independent-screening data is invalid or inconsistent; nothing was repaired.",
        409,
    )


def _blind_mode_on() -> CollectionError:
    return CollectionError(
        "screening_blind_mode_on",
        "Blind mode is on for this collection: other reviewers' decisions, conflicts, "
        "resolutions and agreement stay hidden. Turn it off through the screening protocol.",
        409,
    )


def _not_member() -> CollectionError:
    return CollectionError(
        "screening_document_not_member", "Document is not a collection member.", 422
    )


def _status(review: ScreeningReview | None, collection_revision: int) -> ScreeningStatus:
    if review is None:
        return "unscreened"
    return review.decision if review.collection_revision == collection_revision else "stale"


def _decision(
    reviewer: str, review: ScreeningReview, blinded: bool | None, collection_revision: int
) -> ReviewerDecision:
    return ReviewerDecision(
        reviewer=reviewer,
        current=review.collection_revision == collection_revision,
        decision=review.decision,
        reason=review.reason,
        revision=review.revision,
        collection_revision=review.collection_revision,
        updated_at=review.updated_at,
        blinded=blinded,
    )


def _conflict_type(
    current: Sequence[ReviewerDecision], protocol: ScreeningProtocol
) -> ConflictType | None:
    if len(current) < 2:
        return None
    if len({entry.decision for entry in current}) > 1:
        return "decision"
    if (
        protocol.exclusion_reasons_must_match
        and current[0].decision == "exclude"
        and len({entry.reason for entry in current}) > 1
    ):
        return "exclusion_reason"
    return None


def _basis(current: Sequence[ReviewerDecision]) -> tuple[BasisDecision, ...]:
    return tuple(
        BasisDecision(reviewer=entry.reviewer, revision=entry.revision, decision=entry.decision)
        for entry in current
    )


@dataclass(frozen=True)
class _Snapshot:
    """Validated protocol, decisions and resolutions for current members, one transaction."""

    collection: PaperCollection
    protocol: ScreeningProtocol
    default: Mapping[str, ScreeningReview]
    named: Mapping[str, Mapping[str, ReviewerReview]]
    resolutions: Mapping[str, ScreeningResolution]

    def decisions(self, document_id: str) -> tuple[ReviewerDecision, ...]:
        revision = self.collection.revision
        entries = [
            _decision(reviewer, review, review.blinded, revision)
            for reviewer, review in self.named.get(document_id, {}).items()
        ]
        legacy = self.default.get(document_id)
        if legacy is not None:
            entries.append(_decision(DEFAULT_REVIEWER, legacy, None, revision))
        return tuple(sorted(entries, key=lambda entry: entry.reviewer))

    def final(self, document_id: str) -> FinalDecision:
        decisions = self.decisions(document_id)
        current = [entry for entry in decisions if entry.current]
        resolution = self.resolutions.get(document_id)
        resolution_status: ResolutionStatus = "none"
        if resolution is not None:
            resolution_status = (
                "current"
                if resolution.collection_revision == self.collection.revision
                and resolution.basis == _basis(current)
                else "outdated"
            )
        conflict = _conflict_type(current, self.protocol)
        reasons = {entry.reason for entry in current}
        values: dict[str, object] = {
            "decision": None,
            "source": None,
            "exclusion_reason": None,
            "exclusion_reasons_differ": False,
            "conflict_type": conflict,
            "resolution_status": resolution_status,
            "resolution": resolution,
        }
        if resolution is not None and resolution_status == "current":
            values.update(
                status="resolved",
                decision=resolution.decision,
                source="resolution",
                conflict_type=resolution.conflict_type,
                exclusion_reason=resolution.rationale if resolution.decision == "exclude" else None,
            )
        elif not current:
            values["status"] = "stale" if decisions else "unscreened"
        elif len(current) == 1:
            values["status"] = "single_review"
        elif conflict is not None:
            values["status"] = "conflict"
        else:
            agreed = current[0].decision
            values.update(status="agreed", decision=agreed, source="agreement")
            if agreed == "exclude":
                values.update(
                    exclusion_reason=current[0].reason if len(reasons) == 1 else None,
                    exclusion_reasons_differ=len(reasons) > 1,
                )
        return FinalDecision.model_validate(values)

    def progress(self) -> tuple[ReviewerProgress, ...]:
        current: dict[str, int] = {}
        stale: dict[str, int] = {}
        for document_id in self.collection.document_ids:
            for entry in self.decisions(document_id):
                bucket = current if entry.current else stale
                bucket[entry.reviewer] = bucket.get(entry.reviewer, 0) + 1
        return tuple(
            ReviewerProgress(
                reviewer=reviewer,
                labeled=reviewer != DEFAULT_REVIEWER,
                current_decisions=current.get(reviewer, 0),
                stale_decisions=stale.get(reviewer, 0),
            )
            for reviewer in sorted({*current, *stale})
        )


_PROTOCOL_SELECT = """
SELECT revision, substr(record, 1, :record_limit + 1) AS record,
       typeof(record) != 'text' OR typeof(revision) != 'integer'
           OR length(record) > :record_limit OR instr(record, char(0)) > 0 AS invalid
FROM paper_screening_protocols
WHERE collection_id = :collection_id
ORDER BY revision DESC LIMIT 1
"""
_NAMED_SELECT = """
SELECT document_id, reviewer, substr(record, 1, :record_limit + 1) AS record,
       typeof(record) != 'text' OR typeof(reviewer) != 'text'
           OR length(record) > :record_limit OR instr(record, char(0)) > 0 AS invalid
FROM paper_screening_reviewer_decisions
WHERE collection_id = :collection_id
  AND document_id IN (SELECT value FROM json_each(:document_ids))
LIMIT :row_limit
"""
_RESOLUTION_SELECT = """
SELECT document_id, revision, substr(record, 1, :record_limit + 1) AS record,
       typeof(record) != 'text' OR typeof(revision) != 'integer'
           OR length(record) > :record_limit OR instr(record, char(0)) > 0 AS invalid
FROM paper_screening_resolutions AS latest
WHERE collection_id = :collection_id
  AND document_id IN (SELECT value FROM json_each(:document_ids))
  AND revision = (
      SELECT MAX(revision) FROM paper_screening_resolutions AS earlier
      WHERE earlier.collection_id = latest.collection_id
        AND earlier.document_id = latest.document_id
  )
LIMIT :row_limit
"""
_HISTORY_SELECT = """
SELECT revision, substr(record, 1, :record_limit + 1) AS record,
       typeof(record) != 'text' OR typeof(revision) != 'integer'
           OR length(record) > :record_limit OR instr(record, char(0)) > 0 AS invalid
FROM paper_screening_reviewer_history
WHERE collection_id = :collection_id AND document_id = :document_id
  AND reviewer = :reviewer AND revision <= :last_revision
ORDER BY revision DESC LIMIT :fetch_limit
"""
_LABELS_SELECT = """
SELECT document_id,
       CASE WHEN title = '' THEN X''
           ELSE substr(CAST(title AS BLOB), 1, :title_bytes) END AS title,
       CASE WHEN source = '' THEN X''
           ELSE substr(CAST(source AS BLOB), 1, :source_bytes) END AS source,
       typeof(title) != 'text' OR typeof(source) != 'text'
           OR instr(title, char(0)) > 0 OR instr(source, char(0)) > 0 AS invalid
FROM documents
WHERE document_id IN (SELECT value FROM json_each(:document_ids))
LIMIT :row_limit
"""


def _protocol(connection: sqlite3.Connection, collection_id: str) -> ScreeningProtocol:
    row = connection.execute(
        _PROTOCOL_SELECT, {"collection_id": collection_id, "record_limit": MAX_RECORD_CHARACTERS}
    ).fetchone()
    if row is None:
        return ScreeningProtocol(
            collection_id=collection_id,
            revision=0,
            blind=True,
            exclusion_reasons_must_match=False,
            changed_by=None,
            updated_at=None,
        )
    if row["invalid"]:
        raise _invalid("screening_protocol")
    try:
        protocol = ScreeningProtocol.model_validate_json(row["record"])
    except ValidationError as exc:
        raise _invalid("screening_protocol") from exc
    if (
        protocol.collection_id != collection_id
        or protocol.revision != row["revision"]
        or protocol.revision < 1
    ):
        raise _invalid("screening_protocol")
    return protocol


def _named(
    connection: sqlite3.Connection, collection: PaperCollection
) -> dict[str, dict[str, ReviewerReview]]:
    rows = connection.execute(
        _NAMED_SELECT,
        {
            "collection_id": collection.collection_id,
            "document_ids": json.dumps(collection.document_ids),
            "record_limit": MAX_RECORD_CHARACTERS,
            "row_limit": MAX_DOCUMENT_IDS * MAX_NAMED_REVIEWERS + 1,
        },
    ).fetchall()
    if len(rows) > MAX_DOCUMENT_IDS * MAX_NAMED_REVIEWERS:
        raise _invalid("reviewer_screening")
    result: dict[str, dict[str, ReviewerReview]] = {}
    for row in rows:
        if row["invalid"]:
            raise _invalid("reviewer_screening")
        try:
            review = ReviewerReview.model_validate_json(row["record"])
        except ValidationError as exc:
            raise _invalid("reviewer_screening") from exc
        reviews = result.setdefault(review.document_id, {})
        if (
            review.collection_id != collection.collection_id
            or review.document_id != row["document_id"]
            or review.reviewer != row["reviewer"]
            or review.collection_revision > collection.revision
            or review.reviewer in reviews
        ):
            raise _invalid("reviewer_screening")
        reviews[review.reviewer] = review
    if len({label for reviews in result.values() for label in reviews}) > MAX_NAMED_REVIEWERS:
        raise _invalid("reviewer_screening")
    return result


def _resolutions(
    connection: sqlite3.Connection, collection: PaperCollection
) -> dict[str, ScreeningResolution]:
    rows = connection.execute(
        _RESOLUTION_SELECT,
        {
            "collection_id": collection.collection_id,
            "document_ids": json.dumps(collection.document_ids),
            "record_limit": MAX_RECORD_CHARACTERS,
            "row_limit": MAX_DOCUMENT_IDS + 1,
        },
    ).fetchall()
    result: dict[str, ScreeningResolution] = {}
    for row in rows:
        if row["invalid"]:
            raise _invalid("screening_resolution")
        try:
            resolution = ScreeningResolution.model_validate_json(row["record"])
        except ValidationError as exc:
            raise _invalid("screening_resolution") from exc
        if (
            resolution.collection_id != collection.collection_id
            or resolution.document_id != row["document_id"]
            or resolution.revision != row["revision"]
            or resolution.collection_revision > collection.revision
            or resolution.document_id in result
        ):
            raise _invalid("screening_resolution")
        result[resolution.document_id] = resolution
    return result


def _read(
    screening: SQLitePaperScreening, connection: sqlite3.Connection, collection: PaperCollection
) -> _Snapshot:
    return _Snapshot(
        collection=collection,
        protocol=_protocol(connection, collection.collection_id),
        default=screening._reviews(connection, collection),
        named=_named(connection, collection),
        resolutions=_resolutions(connection, collection),
    )


@dataclass(frozen=True)
class _Labels:
    title: str
    title_truncated: bool
    source: str
    source_truncated: bool


def _labels(connection: sqlite3.Connection, collection: PaperCollection) -> dict[str, _Labels]:
    encoding = connection.execute("PRAGMA encoding").fetchone()[0]
    rows = connection.execute(
        _LABELS_SELECT,
        {
            "title_bytes": 4 * (MAX_TITLE_CHARACTERS + 1),
            "source_bytes": 4 * (MAX_SOURCE_CHARACTERS + 1),
            "document_ids": json.dumps(collection.document_ids),
            "row_limit": MAX_DOCUMENT_IDS + 1,
        },
    ).fetchall()
    stored = {row["document_id"]: row for row in rows}
    try:
        if (
            len(stored) != len(rows)
            or set(stored) != set(collection.document_ids)
            or any(row["invalid"] for row in rows)
        ):
            raise DocumentCatalogError()
        result = {}
        for identifier, row in stored.items():
            title = _prefix(row["title"], encoding, MAX_TITLE_CHARACTERS)
            source = _prefix(row["source"], encoding, MAX_SOURCE_CHARACTERS)
            result[identifier] = _Labels(
                title=title[:MAX_TITLE_CHARACTERS],
                title_truncated=len(title) > MAX_TITLE_CHARACTERS,
                source=source[:MAX_SOURCE_CHARACTERS],
                source_truncated=len(source) > MAX_SOURCE_CHARACTERS,
            )
        return result
    except DocumentCatalogError as exc:
        error = DocumentCatalogError()
        raise CollectionError(error.code, str(error), 409) from exc


def _require_unblinded(protocol: ScreeningProtocol) -> None:
    if protocol.blind:
        raise _blind_mode_on()


def _effective_blind(protocol: ScreeningProtocol, requested: bool | None) -> bool:
    if requested is False and protocol.blind:
        raise CollectionError(
            "screening_blind_mode_on",
            "Blind mode is on for this collection; a request cannot turn it off. Change the "
            "screening protocol deliberately instead.",
            409,
        )
    return protocol.blind or requested is True


def _bounded(content: str | bytes, limit: int, code: str) -> None:
    size = len(content.encode("utf-8") if isinstance(content, str) else content)
    if size > limit:
        raise CollectionError(
            code, "The complete response exceeds its UTF-8 byte limit; nothing partial.", 413
        )


def _final_counts(snapshot: _Snapshot, finals: Mapping[str, FinalDecision]) -> FinalCounts:
    counts = dict.fromkeys(
        (
            "unscreened",
            "stale",
            "single_review",
            "agreed_include",
            "agreed_exclude",
            "agreed_unsure",
            "conflict",
            "resolved_include",
            "resolved_exclude",
        ),
        0,
    )
    for final in finals.values():
        key = (
            f"{final.status}_{final.decision}"
            if final.status in ("agreed", "resolved")
            else final.status
        )
        counts[key] += 1
    double = sum(
        counts[key]
        for key in counts
        if key.startswith(("agreed_", "resolved_")) or key == "conflict"
    )
    return FinalCounts(
        total_documents=len(snapshot.collection.document_ids),
        **counts,
        screened_at_least_once=double + counts["single_review"],
        double_screened=double,
        final_include=counts["agreed_include"] + counts["resolved_include"],
        final_exclude=counts["agreed_exclude"] + counts["resolved_exclude"],
        needs_review=counts["unscreened"]
        + counts["stale"]
        + counts["single_review"]
        + counts["agreed_unsure"]
        + counts["conflict"],
    )


def _conflict_counts(finals: Mapping[str, FinalDecision]) -> ConflictCounts:
    disputed = [final for final in finals.values() if final.status in ("conflict", "resolved")]
    return ConflictCounts(
        disagreements=len(disputed),
        resolved=sum(final.status == "resolved" for final in disputed),
        unresolved=sum(final.status == "conflict" for final in disputed),
        decision=sum(final.conflict_type == "decision" for final in disputed),
        exclusion_reason=sum(final.conflict_type == "exclusion_reason" for final in disputed),
        outdated_resolutions=sum(
            final.resolution_status == "outdated" for final in finals.values()
        ),
    )


def _pairs(snapshot: _Snapshot) -> tuple[PairAgreement, ...]:
    current = {
        document_id: {entry.reviewer: entry for entry in snapshot.decisions(document_id)}
        for document_id in snapshot.collection.document_ids
    }
    active = sorted(
        progress.reviewer for progress in snapshot.progress() if progress.current_decisions
    )
    results = []
    for first, second in combinations(active, 2):
        shared = [
            (entries[first], entries[second])
            for entries in current.values()
            if first in entries
            and second in entries
            and entries[first].current
            and entries[second].current
        ]
        unblinded = sum(entry.blinded is False for pair in shared for entry in pair)
        unknown = sum(entry.blinded is None for pair in shared for entry in pair)
        for scale, (categories, mapping) in _SCALES.items():
            index = {category: position for position, category in enumerate(categories)}
            table = [[0] * len(categories) for _ in categories]
            for left, right in shared:
                table[index[mapping[left.decision]]][index[mapping[right.decision]]] += 1
            results.append(
                PairAgreement(
                    reviewer_a=first,
                    reviewer_b=second,
                    scale=scale,
                    statistics=agreement_statistics(categories, table),
                    unblinded_decisions=unblinded,
                    unknown_blinding_decisions=unknown,
                )
            )
    return tuple(results)


def _reason_groups(
    finals: Mapping[str, FinalDecision],
) -> tuple[tuple[FinalReasonGroup, ...], tuple[str, ...]]:
    groups: dict[str, list[str]] = {}
    without: list[str] = []
    for document_id, final in finals.items():
        if final.decision != "exclude":
            continue
        if final.exclusion_reason is None:
            without.append(document_id)
        else:
            groups.setdefault(final.exclusion_reason, []).append(document_id)
    return (
        tuple(
            FinalReasonGroup(reason=reason, count=len(ids), document_ids=tuple(ids))
            for reason, ids in groups.items()
        ),
        tuple(without),
    )


def _agreement_report(snapshot: _Snapshot) -> AgreementReport:
    collection = snapshot.collection
    finals = {document_id: snapshot.final(document_id) for document_id in collection.document_ids}
    groups, without = _reason_groups(finals)
    return AgreementReport(
        collection_id=collection.collection_id,
        collection_name=collection.name,
        collection_revision=collection.revision,
        protocol=snapshot.protocol,
        reviewers=snapshot.progress(),
        pairs=_pairs(snapshot),
        conflicts=_conflict_counts(finals),
        unresolved_conflict_document_ids=tuple(
            document_id for document_id, final in finals.items() if final.status == "conflict"
        ),
        final_counts=_final_counts(snapshot, finals),
        final_exclusion_reasons=groups,
        final_excludes_without_single_reason=without,
    )


def _summary(snapshot: _Snapshot) -> DualReviewSummary:
    if snapshot.protocol.blind:
        return DualReviewSummary(
            protocol=snapshot.protocol,
            reviewers=snapshot.progress(),
            hidden=True,
            final_counts=None,
            conflicts=None,
            pairs=None,
            final_exclusion_reasons=None,
            notes=_HIDDEN_NOTES,
        )
    report = _agreement_report(snapshot)
    return DualReviewSummary(
        protocol=snapshot.protocol,
        reviewers=report.reviewers,
        hidden=False,
        final_counts=report.final_counts,
        conflicts=report.conflicts,
        pairs=report.pairs,
        final_exclusion_reasons=report.final_exclusion_reasons,
        notes=_SUMMARY_NOTES,
    )


def _number(value: float | None, digits: int = 3) -> str:
    return "undefined" if value is None else f"{value:.{digits}f}"


def agreement_line(pair: PairAgreement) -> str:
    """One readable line: n, p_o, kappa [95% CI], PABAK and band (or why kappa is null)."""
    stats = pair.statistics
    kappa = (
        f"kappa undefined ({stats.kappa_undefined_reason})"
        if stats.kappa is None
        else f"kappa {_number(stats.kappa)} (95% CI {_number(stats.kappa_ci95_low)} to "
        f"{_number(stats.kappa_ci95_high)}), {stats.landis_koch_band}"
    )
    return (
        f"n {stats.compared}, p_o {_number(stats.observed_agreement)}, "
        f"p_e {_number(stats.expected_agreement)}, {kappa}, PABAK {_number(stats.pabak)}"
    )


def _markdown_pairs(pairs: Sequence[PairAgreement]) -> list[str]:
    if not pairs:
        return ["Fewer than two reviewers have current decisions; no pair is compared.\n"]
    sections = [
        "| Reviewers | Scale | n | p_o | p_e | Kappa | 95% CI | PABAK | Landis-Koch |\n"
        "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- |\n"
        + "\n".join(
            f"| {pair.reviewer_a} / {pair.reviewer_b} | {pair.scale} | "
            f"{pair.statistics.compared} | {_number(pair.statistics.observed_agreement)} | "
            f"{_number(pair.statistics.expected_agreement)} | "
            + (
                f"undefined: {pair.statistics.kappa_undefined_reason} | - | "
                if pair.statistics.kappa is None
                else f"{_number(pair.statistics.kappa)} | "
                f"{_number(pair.statistics.kappa_ci95_low)} to "
                f"{_number(pair.statistics.kappa_ci95_high)} | "
            )
            + f"{_number(pair.statistics.pabak)} | {pair.statistics.landis_koch_band or '-'} |"
            for pair in pairs
        )
        + "\n"
    ]
    for pair in pairs:
        stats = pair.statistics
        header = " | ".join(("first \\ second", *stats.categories))
        rows = "\n".join(
            f"| {category} | " + " | ".join(str(count) for count in row) + " |"
            for category, row in zip(stats.categories, stats.confusion, strict=True)
        )
        sections.append(
            f"### {pair.reviewer_a} (rows) / {pair.reviewer_b} (columns): {pair.scale}\n\n"
            f"| {header} |\n|" + " --- |" * (len(stats.categories) + 1) + f"\n{rows}\n\n"
            f"Agreements {stats.agreements}, disagreements {stats.disagreements}; compared "
            f"decisions recorded without enforced blinding: {pair.unblinded_decisions}; "
            f"unlabeled default decisions (blinding unknown): "
            f"{pair.unknown_blinding_decisions}.\n"
        )
    return sections


def _markdown_counts(counts: FinalCounts, conflicts: ConflictCounts) -> str:
    rows = (
        ("Current members", counts.total_documents),
        ("Screened by at least one reviewer", counts.screened_at_least_once),
        ("Independently double-screened", counts.double_screened),
        ("Final include (agreement + resolution)", counts.final_include),
        ("Final exclude (agreement + resolution)", counts.final_exclude),
        ("Needs human review", counts.needs_review),
        ("- unscreened", counts.unscreened),
        ("- stale", counts.stale),
        ("- single review only", counts.single_review),
        ("- agreed unsure", counts.agreed_unsure),
        ("- unresolved conflict", counts.conflict),
        ("Disagreements", conflicts.disagreements),
        ("- resolved", conflicts.resolved),
        ("- unresolved", conflicts.unresolved),
        ("Outdated resolutions", conflicts.outdated_resolutions),
    )
    return (
        "| Count | Members |\n| --- | ---: |\n"
        + "\n".join(f"| {label} | {value} |" for label, value in rows)
        + "\n"
    )


def summary_markdown(summary: DualReviewSummary) -> str:
    """Literal-safe Markdown for the progress report's team section."""
    protocol = summary.protocol
    sections = [
        "## Independent dual-reviewer screening\n",
        *[note + "\n" for note in summary.notes],
        f"Protocol revision {protocol.revision}: blind mode "
        f"**{'on' if protocol.blind else 'off'}**; exclusion reasons must match: "
        f"**{'yes' if protocol.exclusion_reasons_must_match else 'no'}**.\n",
        "| Reviewer | Labeled | Current decisions | Stale decisions |\n"
        "| --- | --- | ---: | ---: |\n"
        + "\n".join(
            f"| {entry.reviewer} | {'yes' if entry.labeled else 'no (default)'} | "
            f"{entry.current_decisions} | {entry.stale_decisions} |"
            for entry in summary.reviewers
        )
        + "\n",
    ]
    if summary.final_counts is not None and summary.conflicts is not None:
        sections.extend(
            [
                "### Team outcomes and PRISMA-style counts\n",
                _markdown_counts(summary.final_counts, summary.conflicts),
                "### Inter-rater agreement\n",
                *_markdown_pairs(summary.pairs or ()),
            ]
        )
        for group in summary.final_exclusion_reasons or ():
            sections.extend(
                [
                    f"### Final exclusion reason: {group.count} member(s)\n",
                    literal_block(group.reason),
                    literal_block(
                        json.dumps(group.document_ids, ensure_ascii=True, indent=2), "json"
                    ),
                ]
            )
    return "\n".join(sections)


def _agreement_markdown(report: AgreementReport) -> str:
    method = report.method
    sections = [
        "# Independent screening agreement report\n",
        "Version 1.0. Agreement of independent human decisions, not scientific validation.\n",
        *[note + "\n" for note in report.notes],
        "## Exact collection\n",
        literal_block(
            json.dumps(
                {
                    "collection_id": report.collection_id,
                    "collection_name": report.collection_name,
                    "collection_revision": report.collection_revision,
                    "protocol_revision": report.protocol.revision,
                    "blind": report.protocol.blind,
                    "exclusion_reasons_must_match": report.protocol.exclusion_reasons_must_match,
                },
                ensure_ascii=True,
                indent=2,
            ),
            "json",
        ),
        "## Team outcomes and PRISMA-style counts\n",
        _markdown_counts(report.final_counts, report.conflicts),
        "Unresolved conflict member IDs:\n",
        literal_block(
            json.dumps(report.unresolved_conflict_document_ids, ensure_ascii=True, indent=2),
            "json",
        ),
        "## Inter-rater agreement\n",
        *_markdown_pairs(report.pairs),
        "## Method\n",
        *[
            f"- **{label}.** {text}\n"
            for label, text in (
                ("Decisions", method.decisions),
                ("Scales", method.scales),
                ("Kappa", method.kappa),
                ("Standard error", method.standard_error),
                ("Confidence interval", method.confidence_interval),
                ("PABAK", method.pabak),
                ("Landis-Koch bands", method.landis_koch),
                ("Blinding", method.blinding),
            )
        ],
        "## Complete JSON report for audit\n",
        literal_block(
            json.dumps(
                report.model_dump(mode="json"), ensure_ascii=False, indent=2, allow_nan=False
            ),
            "json",
        ),
    ]
    return "\n".join(sections)


_Cell = str | int | float | bool | None


def _csv(fields: Sequence[str], rows: Sequence[Mapping[str, _Cell]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, quoting=csv.QUOTE_ALL, lineterminator="\r\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: csv_cell(row[field]) for field in fields})
    return output.getvalue().encode("utf-8")


_AGREEMENT_FIELDS = (
    "schema_version",
    "csv_text_encoding",
    "collection_id",
    "collection_name",
    "collection_revision",
    "protocol_revision",
    "reviewer_a",
    "reviewer_b",
    "scale",
    "categories",
    "confusion",
    "compared",
    "agreements",
    "disagreements",
    "observed_agreement",
    "expected_agreement",
    "kappa",
    "kappa_undefined_reason",
    "kappa_standard_error",
    "kappa_ci95_low",
    "kappa_ci95_high",
    "pabak",
    "landis_koch_band",
    "unblinded_decisions",
    "unknown_blinding_decisions",
    "conflicts_unresolved",
    "conflicts_resolved",
    "final_include",
    "final_exclude",
    "needs_review",
)


def _agreement_csv(report: AgreementReport) -> bytes:
    rows: list[dict[str, _Cell]] = []
    for pair in report.pairs:
        stats = pair.statistics
        rows.append(
            {
                "schema_version": report.schema_version,
                "csv_text_encoding": "apostrophe-prefix-v1",
                "collection_id": report.collection_id,
                "collection_name": report.collection_name,
                "collection_revision": report.collection_revision,
                "protocol_revision": report.protocol.revision,
                "reviewer_a": pair.reviewer_a,
                "reviewer_b": pair.reviewer_b,
                "scale": pair.scale,
                "categories": json.dumps(stats.categories),
                "confusion": json.dumps(stats.confusion),
                "compared": stats.compared,
                "agreements": stats.agreements,
                "disagreements": stats.disagreements,
                "observed_agreement": stats.observed_agreement,
                "expected_agreement": stats.expected_agreement,
                "kappa": stats.kappa,
                "kappa_undefined_reason": stats.kappa_undefined_reason,
                "kappa_standard_error": stats.kappa_standard_error,
                "kappa_ci95_low": stats.kappa_ci95_low,
                "kappa_ci95_high": stats.kappa_ci95_high,
                "pabak": stats.pabak,
                "landis_koch_band": stats.landis_koch_band,
                "unblinded_decisions": pair.unblinded_decisions,
                "unknown_blinding_decisions": pair.unknown_blinding_decisions,
                "conflicts_unresolved": report.conflicts.unresolved,
                "conflicts_resolved": report.conflicts.resolved,
                "final_include": report.final_counts.final_include,
                "final_exclude": report.final_counts.final_exclude,
                "needs_review": report.final_counts.needs_review,
            }
        )
    return _csv(_AGREEMENT_FIELDS, rows)


_FINAL_FIELDS = (
    "schema_version",
    "csv_text_encoding",
    "collection_id",
    "collection_name",
    "collection_revision",
    "protocol_revision",
    "document_id",
    "title",
    "title_truncated",
    "source",
    "source_truncated",
    "final_status",
    "final_decision",
    "final_source",
    "final_exclusion_reason",
    "exclusion_reasons_differ",
    "conflict_type",
    "resolution_status",
    "resolution_revision",
    "resolver",
    "resolution_decision",
    "resolution_rationale",
    "resolution_updated_at",
    "reviewer",
    "reviewer_decision_current",
    "reviewer_decision",
    "reviewer_reason",
    "reviewer_decision_revision",
    "reviewer_collection_revision",
    "reviewer_updated_at",
    "reviewer_blinded",
)


def _final_csv(export: FinalDecisionExport) -> bytes:
    rows: list[dict[str, _Cell]] = []
    for item in export.items:
        final = item.final
        resolution = final.resolution
        base: dict[str, _Cell] = {
            "schema_version": export.schema_version,
            "csv_text_encoding": "apostrophe-prefix-v1",
            "collection_id": export.collection_id,
            "collection_name": export.collection_name,
            "collection_revision": export.collection_revision,
            "protocol_revision": export.protocol.revision,
            "document_id": item.document_id,
            "title": item.title,
            "title_truncated": item.title_truncated,
            "source": item.source,
            "source_truncated": item.source_truncated,
            "final_status": final.status,
            "final_decision": final.decision,
            "final_source": final.source,
            "final_exclusion_reason": final.exclusion_reason,
            "exclusion_reasons_differ": final.exclusion_reasons_differ,
            "conflict_type": final.conflict_type,
            "resolution_status": final.resolution_status,
            "resolution_revision": resolution.revision if resolution else None,
            "resolver": resolution.resolver if resolution else None,
            "resolution_decision": resolution.decision if resolution else None,
            "resolution_rationale": resolution.rationale if resolution else None,
            "resolution_updated_at": resolution.updated_at.isoformat() if resolution else None,
        }
        empty = dict.fromkeys(_FINAL_FIELDS[23:])
        if not item.decisions:
            rows.append({**base, **empty})
        for entry in item.decisions:
            rows.append(
                {
                    **base,
                    "reviewer": entry.reviewer,
                    "reviewer_decision_current": entry.current,
                    "reviewer_decision": entry.decision,
                    "reviewer_reason": entry.reason,
                    "reviewer_decision_revision": entry.revision,
                    "reviewer_collection_revision": entry.collection_revision,
                    "reviewer_updated_at": entry.updated_at.isoformat(),
                    "reviewer_blinded": entry.blinded,
                }
            )
    return _csv(_FINAL_FIELDS, rows)


def _json_bytes(model: BaseModel) -> bytes:
    return (
        json.dumps(model.model_dump(mode="json"), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _download(
    content: bytes, format: str, kind: str, collection_id: str, name: str, revision: int
) -> ScreeningDownload:
    _bounded(content, MAX_DOWNLOAD_BYTES, f"{kind.replace('-', '_')}_too_large")
    extension = {"json": "json", "markdown": "md", "csv": "csv"}[format]
    media = {
        "json": "application/json",
        "markdown": "text/markdown; charset=utf-8",
        "csv": "text/csv; charset=utf-8",
    }[format]
    slug = re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-")[:40] or "collection"
    return ScreeningDownload(
        content=content,
        media_type=media,
        filename=f"{kind}-{slug}-{collection_id}-r{revision}.{extension}",
    )


class _RevisionOptions(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    collection_id: CollectionId
    collection_revision: Revision


class _ReviewerOptions(_RevisionOptions):
    reviewer: NamedReviewer


class _QueueOptions(_ReviewerOptions):
    blind: bool | None = Field(default=None, strict=True)
    limit: int = Field(default=DEFAULT_PAGE_SIZE, strict=True, ge=1, le=MAX_DOCUMENT_IDS)
    cursor: Identity | None = None
    status: ScreeningStatus | None = None


class _HistoryOptions(_ReviewerOptions):
    document_id: ContextDocumentIdentity
    limit: int = Field(default=DEFAULT_PAGE_SIZE, strict=True, ge=1, le=MAX_PAGE_SIZE)
    cursor: Revision | None = None


class _ConflictOptions(_RevisionOptions):
    limit: int = Field(default=DEFAULT_PAGE_SIZE, strict=True, ge=1, le=MAX_DOCUMENT_IDS)
    cursor: Identity | None = None


class _AgreementOptions(_RevisionOptions):
    format: AgreementFormat = "json"


class _FinalOptions(_RevisionOptions):
    format: TeamExportFormat = "json"


def _page(
    collection: PaperCollection, cursor: str | None, selected: Sequence[str]
) -> Iterator[str]:
    if cursor is not None and cursor not in collection.document_ids:
        raise CollectionError(
            "screening_cursor_invalid", "Cursor must identify a collection member.", 422
        )
    start = 0 if cursor is None else collection.document_ids.index(cursor) + 1
    wanted = set(selected)
    return (document_id for document_id in collection.document_ids[start:] if document_id in wanted)


def _history_page(
    connection: sqlite3.Connection,
    collection: PaperCollection,
    document_id: str,
    reviewer: str,
    latest: ReviewerReview | None,
    *,
    limit: int,
    cursor: int | None,
) -> tuple[list[ReviewerReview], int | None]:
    """A contiguous newest-first page of one reviewer's complete history for one member."""
    newest = latest.revision if latest is not None else 0
    if cursor is not None and not 1 <= cursor <= newest:
        raise CollectionError(
            "screening_history_cursor_invalid",
            "Cursor must identify one of your retained decisions for this member.",
            422,
        )
    last = newest if cursor is None else cursor - 1
    rows = connection.execute(
        _HISTORY_SELECT,
        {
            "collection_id": collection.collection_id,
            "document_id": document_id,
            "reviewer": reviewer,
            "last_revision": last,
            "record_limit": MAX_RECORD_CHARACTERS,
            "fetch_limit": limit + 1,
        },
    ).fetchall()
    records = []
    expected = last
    for row in rows:
        if row["invalid"] or row["revision"] != expected:
            raise _invalid("reviewer_screening_history")
        try:
            review = ReviewerReview.model_validate_json(row["record"])
        except ValidationError as exc:
            raise _invalid("reviewer_screening_history") from exc
        if (
            review.collection_id != collection.collection_id
            or review.document_id != document_id
            or review.reviewer != reviewer
            or review.revision != expected
            or review.collection_revision > collection.revision
            or (review.revision == newest and review != latest)
        ):
            raise _invalid("reviewer_screening_history")
        records.append(review)
        expected -= 1
    if len(records) <= limit and expected != 0:
        raise _invalid("reviewer_screening_history")
    return records[:limit], records[limit - 1].revision if len(records) > limit else None


def _reviewer_item(
    snapshot: _Snapshot, labels: Mapping[str, _Labels], document_id: str, reviewer: str, blind: bool
) -> ReviewerItem:
    own = snapshot.named.get(document_id, {}).get(reviewer)
    status = _status(own, snapshot.collection.revision)
    hidden = blind and status not in _CURRENT
    return ReviewerItem(
        document_id=document_id,
        title=labels[document_id].title,
        title_truncated=labels[document_id].title_truncated,
        status=status,
        review=own,
        others_hidden=hidden,
        other_decisions=None
        if hidden
        else tuple(
            entry for entry in snapshot.decisions(document_id) if entry.reviewer != reviewer
        ),
        final=None if hidden else snapshot.final(document_id),
    )


def _conflict_items(
    snapshot: _Snapshot, labels: Mapping[str, _Labels], finals: Mapping[str, FinalDecision]
) -> dict[str, ConflictItem]:
    items = {}
    for document_id, final in finals.items():
        if final.status not in ("conflict", "resolved"):
            continue
        current = tuple(entry for entry in snapshot.decisions(document_id) if entry.current)
        if final.conflict_type is None:
            raise _invalid("screening_resolution")
        items[document_id] = ConflictItem(
            document_id=document_id,
            title=labels[document_id].title,
            title_truncated=labels[document_id].title_truncated,
            conflict_type=final.conflict_type,
            decisions=current,
            expected_basis=tuple(
                BasisReference(reviewer=entry.reviewer, revision=entry.revision)
                for entry in current
            ),
            resolution_revision=final.resolution.revision if final.resolution else 0,
            final=final,
        )
    return items


def has_team_screening(screening: SQLitePaperScreening, collection_id: str) -> bool:
    """Whether named-reviewer decisions or resolutions exist, without a revision check."""
    with screening.collections._connection() as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name IN (?, ?, ?)",
                _TABLES,
            )
        }
        if tables != set(_TABLES):
            return False
        row = connection.execute(
            """SELECT EXISTS (SELECT 1 FROM paper_screening_reviewer_decisions
                              WHERE collection_id = ?)
                   OR EXISTS (SELECT 1 FROM paper_screening_resolutions
                              WHERE collection_id = ?)""",
            (collection_id, collection_id),
        ).fetchone()
        return bool(row[0])


def report_summary(
    screening: SQLitePaperScreening,
    collection_id: str,
    *,
    collection_revision: int,
    expected_default: Mapping[str, ScreeningReview | None],
) -> DualReviewSummary | None:
    """Team section for the progress report, or None when no named reviewer exists.

    Read in its own snapshot; the unlabeled decisions it sees must equal ``expected_default``
    (from the report's export), so the combined report never mixes two decision states.
    """
    if not has_team_screening(screening, collection_id):
        return None
    with screening.collections.resolved_collection(
        collection_id, expected_revision=collection_revision
    ) as (connection, collection):
        snapshot = _read(screening, connection, collection)
    if any(
        snapshot.default.get(document_id) != expected_default.get(document_id)
        for document_id in collection.document_ids
    ) or set(expected_default) != set(collection.document_ids):
        raise CollectionError(
            "screening_report_snapshot_changed",
            "Screening decisions changed while the report was read; download it again.",
            409,
        )
    return _summary(snapshot)


class SQLiteDualScreening:
    """Named reviewers beside the existing unlabeled decisions; one transaction per call."""

    def __init__(self, screening: SQLitePaperScreening, database_path: Path | str) -> None:
        self.screening = screening
        try:
            with closing(sqlite3.connect(database_path)) as connection, connection:
                connection.execute(
                    """CREATE TABLE IF NOT EXISTS paper_screening_reviewer_decisions (
                        collection_id TEXT NOT NULL,
                        document_id TEXT NOT NULL,
                        reviewer TEXT NOT NULL,
                        record TEXT NOT NULL,
                        PRIMARY KEY (collection_id, document_id, reviewer)
                    )"""
                )
                connection.execute(
                    """CREATE TABLE IF NOT EXISTS paper_screening_reviewer_history (
                        collection_id TEXT NOT NULL,
                        document_id TEXT NOT NULL,
                        reviewer TEXT NOT NULL,
                        revision INTEGER NOT NULL CHECK(revision >= 1),
                        record TEXT NOT NULL,
                        PRIMARY KEY (collection_id, document_id, reviewer, revision)
                    )"""
                )
                connection.execute(
                    """CREATE TABLE IF NOT EXISTS paper_screening_resolutions (
                        collection_id TEXT NOT NULL,
                        document_id TEXT NOT NULL,
                        revision INTEGER NOT NULL CHECK(revision >= 1),
                        record TEXT NOT NULL,
                        PRIMARY KEY (collection_id, document_id, revision)
                    )"""
                )
                connection.execute(
                    """CREATE TABLE IF NOT EXISTS paper_screening_protocols (
                        collection_id TEXT NOT NULL,
                        revision INTEGER NOT NULL CHECK(revision >= 1),
                        record TEXT NOT NULL,
                        PRIMARY KEY (collection_id, revision)
                    )"""
                )
        except sqlite3.Error as exc:
            raise _storage_unavailable() from exc

    def protocol(self, collection_id: str) -> ScreeningProtocol:
        """The saved protocol, or the unsaved blind default (revision 0)."""
        with self.screening.collections.resolved_collection(collection_id) as (connection, _):
            return _protocol(connection, collection_id)

    def set_protocol(self, collection_id: str, change: ProtocolChange) -> ScreeningProtocol:
        """Append one protocol revision; blind mode changes never touch any decision."""
        payload = ProtocolChange.model_validate(change.model_dump())
        with self.screening.collections.resolved_collection(collection_id, write=True) as (
            connection,
            collection,
        ):
            current = _protocol(connection, collection.collection_id)
            if current.revision != payload.expected_protocol_revision:
                raise CollectionError(
                    "screening_protocol_revision_conflict",
                    "The protocol changed; read it again before changing it.",
                    409,
                )
            if (current.blind, current.exclusion_reasons_must_match) == (
                payload.blind,
                payload.exclusion_reasons_must_match,
            ):
                raise CollectionError(
                    "screening_protocol_unchanged",
                    "Choose a different blind mode or reason rule before saving.",
                    422,
                )
            if current.revision == MAX_REVISION:
                raise CollectionError(
                    "screening_protocol_revision_exhausted",
                    "Protocol revision cannot increase.",
                    409,
                )
            protocol = ScreeningProtocol(
                collection_id=collection.collection_id,
                revision=current.revision + 1,
                blind=payload.blind,
                exclusion_reasons_must_match=payload.exclusion_reasons_must_match,
                changed_by=payload.changed_by,
                updated_at=datetime.now(UTC),
            )
            connection.execute(
                "INSERT INTO paper_screening_protocols (collection_id, revision, record) "
                "VALUES (?, ?, ?)",
                (collection.collection_id, protocol.revision, protocol.model_dump_json()),
            )
            return protocol

    def submit(
        self, collection_id: str, document_id: str, submission: ReviewerSubmission
    ) -> ReviewerReview:
        """Record one named reviewer's decision and append it to that reviewer's history."""
        identifier = _IDENTITY.validate_python(document_id)
        payload = ReviewerSubmission.model_validate(submission.model_dump())
        with self.screening.collections.resolved_collection(
            collection_id, expected_revision=payload.collection_revision, write=True
        ) as (connection, collection):
            if identifier not in collection.document_ids:
                raise _not_member()
            snapshot = _read(self.screening, connection, collection)
            previous = snapshot.named.get(identifier, {}).get(payload.reviewer)
            if previous is None:
                labels = {
                    row[0]
                    for row in connection.execute(
                        "SELECT DISTINCT reviewer FROM paper_screening_reviewer_decisions "
                        "WHERE collection_id = ? LIMIT ?",
                        (collection.collection_id, MAX_NAMED_REVIEWERS + 1),
                    )
                }
                if payload.reviewer not in labels and len(labels) >= MAX_NAMED_REVIEWERS:
                    raise CollectionError(
                        "screening_reviewer_limit",
                        f"A collection supports at most {MAX_NAMED_REVIEWERS} named reviewers.",
                        409,
                    )
            revision = previous.revision if previous is not None else 0
            if revision != payload.expected_decision_revision:
                raise CollectionError(
                    "screening_revision_conflict",
                    "Your decision changed; read your queue before updating.",
                    409,
                )
            if revision == MAX_REVISION:
                raise CollectionError(
                    "screening_revision_exhausted", "Decision revision cannot increase.", 409
                )
            review = ReviewerReview(
                schema_version="1.0",
                collection_id=collection.collection_id,
                document_id=identifier,
                collection_revision=collection.revision,
                revision=revision + 1,
                decision=payload.decision,
                reason=payload.reason,
                updated_at=datetime.now(UTC),
                reviewer=payload.reviewer,
                blinded=snapshot.protocol.blind and previous is None,
            )
            record = review.model_dump_json()
            if len(record) > MAX_RECORD_CHARACTERS:
                raise _invalid("reviewer_screening")
            connection.execute(
                """INSERT INTO paper_screening_reviewer_decisions
                   (collection_id, document_id, reviewer, record) VALUES (?, ?, ?, ?)
                   ON CONFLICT (collection_id, document_id, reviewer)
                   DO UPDATE SET record = excluded.record""",
                (collection.collection_id, identifier, payload.reviewer, record),
            )
            connection.execute(
                """INSERT INTO paper_screening_reviewer_history
                   (collection_id, document_id, reviewer, revision, record)
                   VALUES (?, ?, ?, ?, ?)""",
                (collection.collection_id, identifier, payload.reviewer, review.revision, record),
            )
            return review

    def reviewer_queue(
        self,
        collection_id: str,
        *,
        reviewer: str,
        collection_revision: int,
        blind: bool | None = None,
        limit: int = DEFAULT_PAGE_SIZE,
        cursor: str | None = None,
        status: ScreeningStatus | None = None,
    ) -> ReviewerQueue:
        """A reviewer's own statuses; others appear only where that reviewer is unblinded.

        In blind mode a member stays hidden until this reviewer has a decision at the current
        collection revision; a stale own decision hides it again.
        """
        options = _QueueOptions(
            collection_id=collection_id,
            collection_revision=collection_revision,
            reviewer=reviewer,
            blind=blind,
            limit=limit,
            cursor=cursor,
            status=status,
        )
        with self.screening.collections.resolved_collection(
            options.collection_id, expected_revision=options.collection_revision
        ) as (connection, collection):
            snapshot = _read(self.screening, connection, collection)
            labels = _labels(connection, collection)
        effective = _effective_blind(snapshot.protocol, options.blind)
        counts = dict.fromkeys(("unscreened", "include", "exclude", "unsure", "stale"), 0)
        items = {}
        for document_id in collection.document_ids:
            item = _reviewer_item(snapshot, labels, document_id, options.reviewer, effective)
            counts[item.status] += 1
            items[document_id] = item
        selected = list(
            _page(
                collection,
                options.cursor,
                [
                    document_id
                    for document_id, item in items.items()
                    if options.status is None or item.status == options.status
                ],
            )
        )
        result = ReviewerQueue(
            collection_id=collection.collection_id,
            collection_revision=collection.revision,
            reviewer=options.reviewer,
            blind=effective,
            collection_blind=snapshot.protocol.blind,
            total_documents=len(collection.document_ids),
            counts=ScreeningCounts.model_validate(counts),
            items=[items[document_id] for document_id in selected[: options.limit]],
            next_cursor=selected[options.limit - 1] if len(selected) > options.limit else None,
        )
        _bounded(
            result.model_dump_json(), MAX_REVIEWER_RESPONSE_BYTES, "screening_response_too_large"
        )
        return result

    def reviewer_history(
        self,
        collection_id: str,
        document_id: str,
        *,
        reviewer: str,
        collection_revision: int,
        limit: int = DEFAULT_PAGE_SIZE,
        cursor: int | None = None,
    ) -> ReviewerHistory:
        """One reviewer's own append-only decisions for one member; never other reviewers."""
        options = _HistoryOptions(
            collection_id=collection_id,
            collection_revision=collection_revision,
            reviewer=reviewer,
            document_id=document_id,
            limit=limit,
            cursor=cursor,
        )
        with self.screening.collections.resolved_collection(
            options.collection_id, expected_revision=options.collection_revision
        ) as (connection, collection):
            if options.document_id not in collection.document_ids:
                raise _not_member()
            latest = (
                _named(connection, collection).get(options.document_id, {}).get(options.reviewer)
            )
            records, next_cursor = _history_page(
                connection,
                collection,
                options.document_id,
                options.reviewer,
                latest,
                limit=options.limit,
                cursor=options.cursor,
            )
        result = ReviewerHistory(
            collection_id=collection.collection_id,
            collection_name=collection.name,
            collection_revision=collection.revision,
            reviewer=options.reviewer,
            document_id=options.document_id,
            current_status=_status(latest, collection.revision),
            latest_review=latest,
            items=records,
            next_cursor=next_cursor,
        )
        _bounded(
            result.model_dump_json(),
            MAX_REVIEWER_RESPONSE_BYTES,
            "screening_history_response_too_large",
        )
        return result

    def reviewer_member(
        self,
        collection_id: str,
        document_id: str,
        *,
        reviewer: str,
        collection_revision: int,
        blind: bool | None = None,
        history_limit: int = 5,
    ) -> ReviewerMember:
        """One member for one reviewer plus that reviewer's recent own history, one read."""
        options = _HistoryOptions(
            collection_id=collection_id,
            collection_revision=collection_revision,
            reviewer=reviewer,
            document_id=document_id,
            limit=history_limit,
        )
        with self.screening.collections.resolved_collection(
            options.collection_id, expected_revision=options.collection_revision
        ) as (connection, collection):
            if options.document_id not in collection.document_ids:
                raise _not_member()
            snapshot = _read(self.screening, connection, collection)
            effective = _effective_blind(snapshot.protocol, blind)
            labels = _labels(connection, collection)
            item = _reviewer_item(
                snapshot, labels, options.document_id, options.reviewer, effective
            )
            history, next_cursor = _history_page(
                connection,
                collection,
                options.document_id,
                options.reviewer,
                item.review,
                limit=options.limit,
                cursor=None,
            )
        return ReviewerMember(
            collection_id=collection.collection_id,
            collection_name=collection.name,
            collection_revision=collection.revision,
            reviewer=options.reviewer,
            blind=effective,
            collection_blind=snapshot.protocol.blind,
            item=item,
            recent_history=history,
            older_history=next_cursor is not None,
        )

    def team_overview(self, collection_id: str, *, collection_revision: int) -> TeamOverview:
        """Progress in any mode; conflicts, outcomes and agreement only once unblinded."""
        options = _RevisionOptions(
            collection_id=collection_id, collection_revision=collection_revision
        )
        with self.screening.collections.resolved_collection(
            options.collection_id, expected_revision=options.collection_revision
        ) as (connection, collection):
            snapshot = _read(self.screening, connection, collection)
            labels = None if snapshot.protocol.blind else _labels(connection, collection)
        summary = _summary(snapshot)
        conflicts = None
        if labels is not None:
            finals = {
                document_id: snapshot.final(document_id) for document_id in collection.document_ids
            }
            conflicts = tuple(_conflict_items(snapshot, labels, finals).values())
        return TeamOverview(
            collection_id=collection.collection_id,
            collection_name=collection.name,
            collection_revision=collection.revision,
            summary=summary,
            conflicts=conflicts,
        )

    def reviewer_priority(
        self, collection_id: str, *, reviewer: str, collection_revision: int
    ) -> ReviewerPriority:
        """Order this reviewer's pending members using only this reviewer's own labels."""
        options = _ReviewerOptions(
            collection_id=collection_id,
            collection_revision=collection_revision,
            reviewer=reviewer,
        )
        with self.screening.collections.resolved_collection(
            options.collection_id, expected_revision=options.collection_revision
        ) as (connection, collection):
            own = {
                document_id: reviews[options.reviewer]
                for document_id, reviews in _named(connection, collection).items()
                if options.reviewer in reviews
            }
            encoding = connection.execute("PRAGMA encoding").fetchone()[0]
            rows = connection.execute(
                _MEMBERS_SQL,
                {
                    "title_bytes": 4 * (MAX_TITLE_CHARACTERS + 1),
                    "text_bytes": 4 * (MAX_FEATURE_TEXT_CHARACTERS + 1),
                    "document_ids": json.dumps(collection.document_ids),
                    "row_limit": MAX_DOCUMENT_IDS + 1,
                },
            ).fetchall()
        statuses = {
            document_id: _status(own.get(document_id), collection.revision)
            for document_id in collection.document_ids
        }
        counts = dict.fromkeys(("unscreened", "include", "exclude", "unsure", "stale"), 0)
        for value in statuses.values():
            counts[value] += 1
        queue = ScreeningQueue(
            collection_id=collection.collection_id,
            collection_revision=collection.revision,
            total_documents=len(collection.document_ids),
            counts=ScreeningCounts.model_validate(counts),
            items=[
                ScreeningItem(document_id=document_id, status=statuses[document_id], review=None)
                for document_id in collection.document_ids
            ],
            next_cursor=None,
            included_document_ids=tuple(
                document_id for document_id, value in statuses.items() if value == "include"
            ),
        )
        stored = {row["document_id"]: row for row in rows}
        try:
            if (
                len(stored) != len(rows)
                or set(stored) != set(collection.document_ids)
                or any(row["invalid"] for row in rows)
            ):
                raise DocumentCatalogError()
            members = []
            for document_id in collection.document_ids:
                row = stored[document_id]
                title = _prefix(row["title"], encoding, MAX_TITLE_CHARACTERS)
                text = _prefix(row["text"], encoding, MAX_FEATURE_TEXT_CHARACTERS)
                members.append(
                    PriorityMember(
                        document_id=document_id,
                        status=statuses[document_id],
                        title=title[:MAX_TITLE_CHARACTERS],
                        title_truncated=len(title) > MAX_TITLE_CHARACTERS,
                        text=text[:MAX_FEATURE_TEXT_CHARACTERS],
                    )
                )
        except DocumentCatalogError as exc:
            error = DocumentCatalogError()
            raise CollectionError(error.code, str(error), 409) from exc
        result = ReviewerPriority(
            reviewer=options.reviewer, priority=build_priority(collection, queue, members)
        )
        _bounded(
            result.model_dump_json(), MAX_PRIORITY_RESPONSE_BYTES, "screening_priority_too_large"
        )
        return result

    def conflicts(
        self,
        collection_id: str,
        *,
        collection_revision: int,
        limit: int = DEFAULT_PAGE_SIZE,
        cursor: str | None = None,
    ) -> ScreeningConflicts:
        """Current disagreements, resolved or not, in membership order; unblinded only."""
        options = _ConflictOptions(
            collection_id=collection_id,
            collection_revision=collection_revision,
            limit=limit,
            cursor=cursor,
        )
        with self.screening.collections.resolved_collection(
            options.collection_id, expected_revision=options.collection_revision
        ) as (connection, collection):
            snapshot = _read(self.screening, connection, collection)
            _require_unblinded(snapshot.protocol)
            labels = _labels(connection, collection)
        finals = {
            document_id: snapshot.final(document_id) for document_id in collection.document_ids
        }
        disputed = _conflict_items(snapshot, labels, finals)
        selected = list(_page(collection, options.cursor, list(disputed)))
        items = [disputed[document_id] for document_id in selected[: options.limit]]
        result = ScreeningConflicts(
            collection_id=collection.collection_id,
            collection_name=collection.name,
            collection_revision=collection.revision,
            protocol=snapshot.protocol,
            reviewers=snapshot.progress(),
            total_documents=len(collection.document_ids),
            counts=_conflict_counts(finals),
            items=items,
            next_cursor=selected[options.limit - 1] if len(selected) > options.limit else None,
        )
        _bounded(result.model_dump_json(), MAX_TEAM_RESPONSE_BYTES, "screening_response_too_large")
        return result

    def resolve(
        self, collection_id: str, document_id: str, submission: ResolutionSubmission
    ) -> ScreeningResolution:
        """Append a final decision for one current disagreement; decisions stay unchanged."""
        identifier = _IDENTITY.validate_python(document_id)
        payload = ResolutionSubmission.model_validate(submission.model_dump())
        with self.screening.collections.resolved_collection(
            collection_id, expected_revision=payload.collection_revision, write=True
        ) as (connection, collection):
            if identifier not in collection.document_ids:
                raise _not_member()
            snapshot = _read(self.screening, connection, collection)
            _require_unblinded(snapshot.protocol)
            current = [entry for entry in snapshot.decisions(identifier) if entry.current]
            conflict = _conflict_type(current, snapshot.protocol)
            if conflict is None:
                raise CollectionError(
                    "screening_no_conflict",
                    "This member has no current disagreement between independent decisions.",
                    409,
                )
            basis = _basis(current)
            if [(entry.reviewer, entry.revision) for entry in basis] != [
                (entry.reviewer, entry.revision) for entry in payload.expected_basis
            ]:
                raise CollectionError(
                    "screening_conflict_changed",
                    "Reviewer decisions changed after you opened this conflict; reread it.",
                    409,
                )
            latest = snapshot.resolutions.get(identifier)
            revision = latest.revision if latest is not None else 0
            if revision != payload.expected_resolution_revision:
                raise CollectionError(
                    "screening_resolution_revision_conflict",
                    "Another resolution was recorded; reread the conflict before resolving.",
                    409,
                )
            if revision == MAX_REVISION:
                raise CollectionError(
                    "screening_resolution_revision_exhausted",
                    "Resolution revision cannot increase.",
                    409,
                )
            resolution = ScreeningResolution(
                schema_version="1.0",
                collection_id=collection.collection_id,
                document_id=identifier,
                collection_revision=collection.revision,
                revision=revision + 1,
                resolver=payload.resolver,
                decision=payload.decision,
                rationale=payload.rationale,
                conflict_type=conflict,
                basis=basis,
                updated_at=datetime.now(UTC),
            )
            record = resolution.model_dump_json()
            if len(record) > MAX_RECORD_CHARACTERS:
                raise _invalid("screening_resolution")
            connection.execute(
                "INSERT INTO paper_screening_resolutions "
                "(collection_id, document_id, revision, record) VALUES (?, ?, ?, ?)",
                (collection.collection_id, identifier, resolution.revision, record),
            )
            return resolution

    def agreement(self, collection_id: str, *, collection_revision: int) -> AgreementReport:
        """Kappa, PABAK, conflicts and final counts from one snapshot; unblinded only."""
        options = _RevisionOptions(
            collection_id=collection_id, collection_revision=collection_revision
        )
        with self.screening.collections.resolved_collection(
            options.collection_id, expected_revision=options.collection_revision
        ) as (connection, collection):
            snapshot = _read(self.screening, connection, collection)
        _require_unblinded(snapshot.protocol)
        return _agreement_report(snapshot)

    def export_agreement(
        self, collection_id: str, *, collection_revision: int, format: AgreementFormat = "json"
    ) -> ScreeningDownload:
        options = _AgreementOptions(
            collection_id=collection_id, collection_revision=collection_revision, format=format
        )
        report = self.agreement(
            options.collection_id, collection_revision=options.collection_revision
        )
        content = (
            _json_bytes(report)
            if options.format == "json"
            else _agreement_csv(report)
            if options.format == "csv"
            else _agreement_markdown(report).encode("utf-8")
        )
        return _download(
            content,
            options.format,
            "screening-agreement",
            report.collection_id,
            report.collection_name,
            report.collection_revision,
        )

    def final_decisions(
        self, collection_id: str, *, collection_revision: int
    ) -> FinalDecisionExport:
        """Every member's team outcome with every reviewer's latest decision; unblinded only."""
        options = _RevisionOptions(
            collection_id=collection_id, collection_revision=collection_revision
        )
        with self.screening.collections.resolved_collection(
            options.collection_id, expected_revision=options.collection_revision
        ) as (connection, collection):
            snapshot = _read(self.screening, connection, collection)
            _require_unblinded(snapshot.protocol)
            labels = _labels(connection, collection)
        finals = {
            document_id: snapshot.final(document_id) for document_id in collection.document_ids
        }
        return FinalDecisionExport(
            collection_id=collection.collection_id,
            collection_name=collection.name,
            collection_revision=collection.revision,
            protocol=snapshot.protocol,
            reviewers=snapshot.progress(),
            final_counts=_final_counts(snapshot, finals),
            items=[
                FinalDecisionItem(
                    document_id=document_id,
                    title=labels[document_id].title,
                    title_truncated=labels[document_id].title_truncated,
                    source=labels[document_id].source,
                    source_truncated=labels[document_id].source_truncated,
                    decisions=snapshot.decisions(document_id),
                    final=finals[document_id],
                )
                for document_id in collection.document_ids
            ],
        )

    def export_final_decisions(
        self, collection_id: str, *, collection_revision: int, format: TeamExportFormat = "json"
    ) -> ScreeningDownload:
        options = _FinalOptions(
            collection_id=collection_id, collection_revision=collection_revision, format=format
        )
        export = self.final_decisions(
            options.collection_id, collection_revision=options.collection_revision
        )
        content = _json_bytes(export) if options.format == "json" else _final_csv(export)
        return _download(
            content,
            options.format,
            "screening-final-decisions",
            export.collection_id,
            export.collection_name,
            export.collection_revision,
        )
