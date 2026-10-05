"""Strict, immutable evidence bounds and non-destructive document-count assessments."""

from collections import Counter
from collections.abc import Iterable
from typing import Annotated, Self, TypedDict

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

from retrieval.min_unique_sources_gate import count_evidence_documents
from retrieval.models import SearchResult
from retrieval.near_duplicate_collapse import NearDuplicateCollapser

MaxChunksPerDocument = Annotated[int, Field(strict=True, ge=1, le=50)]
MinEvidenceDocuments = Annotated[int, Field(strict=True, ge=1, le=50)]
NearDuplicateThreshold = Annotated[float, Field(strict=True, gt=0, le=1, allow_inf_nan=False)]


class EvidencePolicy(BaseModel):
    """Requested bounds, not a promise of relevance or independent scientific evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_chunks_per_document: MaxChunksPerDocument | None = None
    min_evidence_documents: MinEvidenceDocuments | None = None
    near_duplicate_threshold: NearDuplicateThreshold | None = None

    @model_validator(mode="after")
    def require_bound(self) -> Self:
        if (
            self.max_chunks_per_document is None
            and self.min_evidence_documents is None
            and self.near_duplicate_threshold is None
        ):
            raise ValueError("Evidence policy requires at least one bound or transformation.")
        return self

    @model_serializer(mode="wrap")
    def serialize_bounds(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        """Keep existing records unchanged and omit unrequested policy fields."""
        fields: dict[str, object] = handler(self)
        return {key: value for key, value in fields.items() if value is not None}


def normalize_evidence_policy(
    max_chunks_per_document: int | None,
    min_evidence_documents: int | None = None,
    near_duplicate_threshold: float | None = None,
) -> EvidencePolicy | None:
    """Validate before asynchronous work; Python None leaves that option unconfigured."""
    return (
        None
        if (
            max_chunks_per_document is None
            and min_evidence_documents is None
            and near_duplicate_threshold is None
        )
        else EvidencePolicy(
            max_chunks_per_document=max_chunks_per_document,
            min_evidence_documents=min_evidence_documents,
            near_duplicate_threshold=near_duplicate_threshold,
        )
    )


class EvidenceLimitArguments(TypedDict, total=False):
    max_chunks_per_document: int
    min_evidence_documents: int
    near_duplicate_threshold: float


def evidence_limit_arguments(
    max_chunks_per_document: int | None,
    min_evidence_documents: int | None = None,
    near_duplicate_threshold: float | None = None,
) -> EvidenceLimitArguments:
    """Omit unconfigured runner keywords entirely for legacy callers."""
    arguments: EvidenceLimitArguments = {}
    if max_chunks_per_document is not None:
        arguments["max_chunks_per_document"] = max_chunks_per_document
    if min_evidence_documents is not None:
        arguments["min_evidence_documents"] = min_evidence_documents
    if near_duplicate_threshold is not None:
        arguments["near_duplicate_threshold"] = near_duplicate_threshold
    return arguments


class EvidencePolicyArguments(TypedDict, total=False):
    evidence_policy: EvidencePolicy


def policy_arguments(policy: EvidencePolicy | None) -> EvidencePolicyArguments:
    """Require explicit executor support only when the request opts in."""
    return {} if policy is None else {"evidence_policy": policy}


def ensure_evidence_policy(policy: EvidencePolicy | None, sources: Iterable[SearchResult]) -> None:
    """Validate transformations; assess a minimum separately so previews retain evidence."""
    if policy is None:
        return
    prepared = list(sources)
    if policy.near_duplicate_threshold is not None:
        for source in prepared:
            if policy.max_chunks_per_document is None:
                valid_path = source.retriever == "near_duplicate_collapse" and bool(source.path)
            else:
                valid_path = len(source.path) >= 2 and source.path[-1] == "near_duplicate_collapse"
            if not valid_path:
                raise ValueError("Prepared evidence is missing near_duplicate_collapse provenance.")
        if NearDuplicateCollapser(policy.near_duplicate_threshold).collapse(prepared) != prepared:
            raise ValueError("Prepared evidence violates near_duplicate_threshold or score order.")
    if policy.max_chunks_per_document is None:
        return
    counts: Counter[str] = Counter()
    for source in prepared:
        counts[source.chunk.document_id] += 1
        if counts[source.chunk.document_id] > policy.max_chunks_per_document:
            raise ValueError("Prepared evidence exceeds max_chunks_per_document.")
        if source.retriever != "diversity_cap_gate" or not source.path:
            raise ValueError("Prepared evidence is missing diversity_cap_gate provenance.")


class EvidenceAssessment(BaseModel):
    """Final-context document counts only, not an answerability or quality judgment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    required_documents: MinEvidenceDocuments
    observed_documents: int = Field(strict=True, ge=0, le=50)
    passed: bool = Field(strict=True)

    @model_validator(mode="after")
    def consistent_counts(self) -> Self:
        if self.passed != (self.observed_documents >= self.required_documents):
            raise ValueError("Evidence assessment does not match its document counts.")
        return self


def assess_evidence(
    policy: EvidencePolicy | None, sources: Iterable[SearchResult]
) -> EvidenceAssessment | None:
    """Inspect the final captured sources, leaving their order and provenance unchanged."""
    if policy is None or policy.min_evidence_documents is None:
        return None
    observed = count_evidence_documents(sources)
    return EvidenceAssessment(
        required_documents=policy.min_evidence_documents,
        observed_documents=observed,
        passed=observed >= policy.min_evidence_documents,
    )


class InsufficientEvidenceError(ValueError):
    """An explicit pre-generation failure carrying durable, typed count diagnostics."""

    def __init__(self, assessment: EvidenceAssessment) -> None:
        self.assessment = assessment
        super().__init__(
            f"min_evidence_documents: required {assessment.required_documents} distinct "
            f"evidence documents, observed {assessment.observed_documents}; "
            "answer generation was not started."
        )


def ensure_evidence_requirement(
    policy: EvidencePolicy | None, sources: Iterable[SearchResult]
) -> None:
    """Fail without discarding captured evidence or inventing an answer."""
    assessment = assess_evidence(policy, sources)
    if assessment is not None and not assessment.passed:
        raise InsufficientEvidenceError(assessment)
