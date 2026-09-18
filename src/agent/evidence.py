"""Versioned evidence contracts, independent of providers and mutable indexes."""

import hashlib
import json
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from agent.models import AgentAnswer, QueryPlan
from llm.schemas import LLMRequest, TaskType
from retrieval.models import SearchResult

SHA256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


def text_digest(text: str) -> str:
    """Hash the exact UTF-8 text, without normalization."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class CaptureLimits(BaseModel):
    """Version-one bounds; never truncate evidence to fit an artifact."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_sources: Literal[50] = 50
    max_context_bytes: Literal[262144] = 262144
    max_snapshot_bytes: Literal[1048576] = 1048576


class RunConfiguration(BaseModel):
    """Allowlisted effective limits frozen before any asynchronous run work."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    max_source_docs: int = Field(ge=1)
    max_hops: int = Field(ge=0)
    retrieval_timeout_seconds: float = Field(gt=0)
    reasoning_timeout_seconds: float = Field(gt=0)


class EvidenceSource(SearchResult):
    """One full post-rerank passage with its final score and one-based rank."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    rank: int = Field(ge=1, le=50)
    text_sha256: SHA256


class EvidenceSnapshot(BaseModel):
    """A detached copy of the exact provider-independent generation request."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    context_format: Literal["chunk-id-title-text-v1"] = "chunk-id-title-text-v1"
    request: LLMRequest
    sources: list[EvidenceSource] = Field(max_length=50)
    context_sha256: SHA256
    limits: CaptureLimits = Field(default_factory=CaptureLimits)

    @classmethod
    def capture(cls, query: str, reranked: list[SearchResult]) -> Self:
        """Build the request from detached passages, before any generation await."""
        if len(reranked) > CaptureLimits().max_sources:
            raise ValueError("Evidence snapshot exceeds 50 sources; reduce source scope.")
        sources = [
            EvidenceSource(
                **result.model_dump(),
                rank=rank,
                text_sha256=text_digest(result.chunk.text),
            )
            for rank, result in enumerate(reranked, start=1)
        ]
        context = "\n".join(
            f"[{source.chunk.chunk_id}] {source.chunk.title}: {source.chunk.text}"
            for source in sources
        )
        return cls(
            request=LLMRequest(
                task_type=TaskType.REASONING,
                prompt=query,
                context=context,
                citation_chunk_ids=[source.chunk.chunk_id for source in sources],
            ),
            sources=sources,
            context_sha256=text_digest(context),
        )

    @model_validator(mode="after")
    def validate_capture(self) -> Self:
        """Reject oversized or inconsistent records instead of guessing evidence."""
        if len(self.request.context.encode("utf-8")) > self.limits.max_context_bytes:
            raise ValueError("Evidence context exceeds 262144 UTF-8 bytes; reduce source scope.")
        chunk_ids = [source.chunk.chunk_id for source in self.sources]
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("Evidence chunk IDs must be unique.")
        if chunk_ids != self.request.citation_chunk_ids:
            raise ValueError("Evidence sources do not match the request citation IDs.")
        if [source.rank for source in self.sources] != list(range(1, len(self.sources) + 1)):
            raise ValueError("Evidence ranks must follow the final context order.")
        context = "\n".join(
            f"[{source.chunk.chunk_id}] {source.chunk.title}: {source.chunk.text}"
            for source in self.sources
        )
        if context != self.request.context or text_digest(context) != self.context_sha256:
            raise ValueError("Evidence context does not match its sources and digest.")
        if any(text_digest(source.chunk.text) != source.text_sha256 for source in self.sources):
            raise ValueError("Evidence passage does not match its digest.")
        # Match SQLiteEventLog's JSON encoding, including escaped Unicode and spacing.
        stored_bytes = json.dumps(self.model_dump(mode="json"), sort_keys=True).encode("utf-8")
        if len(stored_bytes) > self.limits.max_snapshot_bytes:
            raise ValueError("Evidence snapshot exceeds 1048576 bytes; reduce source metadata.")
        return self


class GenerationRecord(BaseModel):
    """Allowlisted returned identity and proposed citations, never raw provider data."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str
    model_name: str | None = None
    task_type: TaskType
    claim_chunk_ids: list[list[str]]


class ClaimEvidenceLink(BaseModel):
    """Resolve proposed and accepted citation IDs, without claiming entailment."""

    claim_number: int = Field(ge=1)
    proposed_chunk_ids: list[str]
    grounded_chunk_ids: list[str]
    evidence_ranks: list[int]
    missing_chunk_ids: list[str]


class CitationEvidenceLink(BaseModel):
    """Locate a final answer citation in the frozen source list."""

    citation_number: int = Field(ge=1)
    chunk_id: str
    evidence_rank: int | None


class ExportEvent(BaseModel):
    """Ordered operational trace; large payloads may reference bundle fields."""

    id: int
    timestamp: str
    agent_id: str
    run_id: str
    event_type: str
    payload: dict[str, JsonValue] | None
    payload_ref: Literal["snapshot", "generation"] | None = None
    payload_omitted: bool = False


class EvidenceBundle(BaseModel):
    """Portable JSON version one, assembled exclusively from saved run records."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    agent_id: str
    status: Literal["DONE"] = "DONE"
    query: str
    completed_at: str
    configuration: RunConfiguration
    plan: QueryPlan
    answer: AgentAnswer
    snapshot: EvidenceSnapshot
    generation: GenerationRecord
    claim_evidence: list[ClaimEvidenceLink]
    citation_evidence: list[CitationEvidenceLink]
    events: list[ExportEvent]
    warnings: list[str]
