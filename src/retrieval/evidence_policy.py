"""Strict, immutable request policy for the existing post-rerank document quota."""

from collections import Counter
from collections.abc import Iterable
from typing import Annotated, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from retrieval.models import SearchResult

MaxChunksPerDocument = Annotated[int, Field(strict=True, ge=1, le=50)]


class EvidencePolicy(BaseModel):
    """A requested quota, not a promise of distinct papers or independent evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_chunks_per_document: MaxChunksPerDocument


def normalize_evidence_policy(max_chunks_per_document: int | None) -> EvidencePolicy | None:
    """Validate before asynchronous work; Python None means no requested quota."""
    return (
        None
        if max_chunks_per_document is None
        else EvidencePolicy(max_chunks_per_document=max_chunks_per_document)
    )


class EvidenceLimitArguments(TypedDict, total=False):
    max_chunks_per_document: int


def evidence_limit_arguments(max_chunks_per_document: int | None) -> EvidenceLimitArguments:
    """Omit the new runner keyword entirely for unconfigured legacy callers."""
    return (
        {}
        if max_chunks_per_document is None
        else {"max_chunks_per_document": max_chunks_per_document}
    )


class EvidencePolicyArguments(TypedDict, total=False):
    evidence_policy: EvidencePolicy


def policy_arguments(policy: EvidencePolicy | None) -> EvidencePolicyArguments:
    """Require explicit executor support only when the request opts in."""
    return {} if policy is None else {"evidence_policy": policy}


def ensure_evidence_policy(policy: EvidencePolicy | None, sources: Iterable[SearchResult]) -> None:
    """Validate quota and gate provenance without reselecting or modifying evidence."""
    if policy is None:
        return
    counts: Counter[str] = Counter()
    for source in sources:
        counts[source.chunk.document_id] += 1
        if counts[source.chunk.document_id] > policy.max_chunks_per_document:
            raise ValueError("Prepared evidence exceeds max_chunks_per_document.")
        if source.retriever != "diversity_cap_gate" or not source.path:
            raise ValueError("Prepared evidence is missing diversity_cap_gate provenance.")
