"""Version-one inspection contracts, not model or scientific quality scores."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from agent.evidence import SHA256, RunConfiguration
from llm.schemas import TaskType
from retrieval.scope import DocumentIds

PREVIEW_CHARACTERS = 240


class _ComparisonModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class TextSummary(_ComparisonModel):
    """Exact UTF-8 digest plus an explicitly bounded, unnormalized text prefix."""

    preview: str = Field(max_length=PREVIEW_CHARACTERS)
    truncated: bool
    characters: int = Field(ge=0)
    utf8_bytes: int = Field(ge=0)
    sha256: SHA256


class SavedGenerationIdentity(_ComparisonModel):
    """Recorded adapter/provider and configured model, not a resolved model revision."""

    provider: str
    model_name: str | None
    task_type: TaskType


class AnswerSummary(_ComparisonModel):
    """Bounded answer preview and exact digests; full details remain in the export."""

    text: TextSummary
    record_sha256: SHA256
    claims_sha256: SHA256
    citations_sha256: SHA256
    citation_associations_sha256: SHA256
    warnings_sha256: SHA256
    ungrounded: bool
    claim_count: int = Field(ge=0)
    grounded_claim_count: int = Field(ge=0)
    citation_count: int = Field(ge=0)
    answer_warning_count: int = Field(ge=0)
    bundle_warning_count: int = Field(ge=0)


class ComparedRun(_ComparisonModel):
    """One completed run's saved comparison inputs, never hydrated from the corpus."""

    run_id: str
    agent_id: str
    completed_at: str
    export_url: str | None
    query: TextSummary
    document_ids: DocumentIds | None
    configuration: RunConfiguration
    generation: SavedGenerationIdentity
    context_sha256: SHA256
    request_sha256: SHA256
    answer: AnswerSummary


class SourceSummary(_ComparisonModel):
    """Identity, rank, final score, and digests without full passage/metadata/path copies."""

    chunk_id: str
    document_id: str
    rank: int = Field(ge=1, le=50)
    title: TextSummary
    text_sha256: SHA256
    source_sha256: SHA256
    metadata_sha256: SHA256
    path_sha256: SHA256
    retriever: TextSummary
    score: float
    record_sha256: SHA256


class SourceChanges(_ComparisonModel):
    """Exact saved field comparisons for one shared (chunk ID, document ID) identity."""

    rank_changed: bool
    text_changed: bool
    title_changed: bool
    source_changed: bool
    metadata_changed: bool
    score_changed: bool
    retriever_changed: bool
    path_changed: bool
    record_changed: bool


class SharedSourceComparison(_ComparisonModel):
    """Baseline order; positive rank_delta means a later position in the candidate."""

    baseline: SourceSummary
    candidate: SourceSummary
    rank_delta: int = Field(ge=-49, le=49)
    changes: SourceChanges


class IdentityOverlap(_ComparisonModel):
    """Set cardinalities only; empty/empty overlap is undefined, never perfect."""

    baseline_count: int = Field(ge=0, le=50)
    candidate_count: int = Field(ge=0, le=50)
    shared_count: int = Field(ge=0, le=50)
    added_count: int = Field(ge=0, le=50)
    removed_count: int = Field(ge=0, le=50)
    union_count: int = Field(ge=0, le=100)
    identity_jaccard: float | None = Field(ge=0, le=1)


class EvidenceComparison(_ComparisonModel):
    """Added in candidate order; removed/shared in baseline order."""

    statistics: IdentityOverlap
    added: list[SourceSummary] = Field(max_length=50)
    removed: list[SourceSummary] = Field(max_length=50)
    shared: list[SharedSourceComparison] = Field(max_length=50)
    reassigned_chunk_ids: list[str] = Field(max_length=50)


class RunChanges(_ComparisonModel):
    """Exact changes in selected saved content, excluding trace identities and clocks."""

    query_changed: bool
    document_scope_changed: bool
    document_scope_membership_changed: bool
    configuration_changed: bool
    provider_changed: bool
    model_changed: bool
    task_type_changed: bool
    request_changed: bool
    context_changed: bool
    source_membership_changed: bool
    source_order_changed: bool
    evidence_changed: bool
    answer_changed: bool
    answer_text_changed: bool
    claim_text_changed: bool
    grounding_changed: bool
    citations_changed: bool
    citation_associations_changed: bool
    warnings_changed: bool


class RunComparison(_ComparisonModel):
    """Deterministic baseline-to-candidate inspection; no retrieval or generation."""

    schema_version: Literal["1.0"] = "1.0"
    baseline: ComparedRun
    candidate: ComparedRun
    changes: RunChanges
    any_changes: bool
    evidence: EvidenceComparison
    notices: list[str]
