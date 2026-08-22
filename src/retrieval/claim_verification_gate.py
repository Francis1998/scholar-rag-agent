"""Deterministic claim-level groundedness verification for draft answers."""

import math
import re
from dataclasses import dataclass

from llm.base import BaseLLMAdapter
from retrieval.models import SearchResult
from retrieval.sparse import meaningful_terms

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass(frozen=True)
class ClaimVerdict:
    """Lexical support decision for one claim sentence."""

    claim: str
    supported: bool
    support_score: float
    supporting_chunk_ids: tuple[str, ...]


@dataclass(frozen=True)
class ClaimVerificationReport:
    """Per-claim support decisions and aggregate groundedness."""

    claims: tuple[ClaimVerdict, ...]
    groundedness: float
    supported_count: int
    unsupported_count: int


class ClaimVerificationGate:
    """Split a draft answer into claims and score lexical evidence support.

    Inspired by RAGAS faithfulness and TruLens groundedness metrics, but fully
    deterministic: no network call is required. An optional LLM adapter may be
    supplied for future semantic checks and is unused by the lexical path.
    """

    def __init__(
        self,
        support_threshold: float = 0.5,
        llm: BaseLLMAdapter | None = None,
    ) -> None:
        """Create a lexical claim verification gate.

        Args:
            support_threshold: Inclusive fraction of claim terms that must
                appear in retrieved evidence for a claim to count as supported.
            llm: Optional provider adapter reserved for future semantic
                verification; ignored by the deterministic path.

        Raises:
            ValueError: If ``support_threshold`` is non-finite or outside
                ``[0.0, 1.0]``.
        """
        if not math.isfinite(support_threshold) or not 0.0 <= support_threshold <= 1.0:
            raise ValueError("support_threshold must be a finite number within [0.0, 1.0]")
        self._support_threshold = support_threshold
        self._llm = llm  # reserved; lexical verification does not call it

    def verify(self, answer: str, results: list[SearchResult]) -> ClaimVerificationReport:
        """Return per-claim support and overall groundedness for ``answer``."""
        claims = self._split_claims(answer)
        if not claims:
            return ClaimVerificationReport(
                claims=(),
                groundedness=0.0,
                supported_count=0,
                unsupported_count=0,
            )

        evidence_terms = [
            (
                result.chunk.chunk_id,
                meaningful_terms(f"{result.chunk.title} {result.chunk.text}"),
            )
            for result in results
        ]

        verdicts: list[ClaimVerdict] = []
        for claim in claims:
            claim_terms = meaningful_terms(claim)
            if not claim_terms:
                verdicts.append(
                    ClaimVerdict(
                        claim=claim,
                        supported=False,
                        support_score=0.0,
                        supporting_chunk_ids=(),
                    )
                )
                continue

            supporting_ids: list[str] = []
            covered: set[str] = set()
            for chunk_id, terms in evidence_terms:
                overlap = claim_terms & terms
                if overlap:
                    covered.update(overlap)
                    supporting_ids.append(chunk_id)

            support_score = len(covered) / len(claim_terms)
            supported = support_score >= self._support_threshold and bool(supporting_ids)
            verdicts.append(
                ClaimVerdict(
                    claim=claim,
                    supported=supported,
                    support_score=support_score,
                    supporting_chunk_ids=tuple(supporting_ids),
                )
            )

        supported_count = sum(1 for verdict in verdicts if verdict.supported)
        unsupported_count = len(verdicts) - supported_count
        groundedness = supported_count / len(verdicts)
        return ClaimVerificationReport(
            claims=tuple(verdicts),
            groundedness=groundedness,
            supported_count=supported_count,
            unsupported_count=unsupported_count,
        )

    @staticmethod
    def _split_claims(answer: str) -> list[str]:
        normalized = " ".join(answer.split())
        if not normalized:
            return []
        return [
            sentence.strip()
            for sentence in _SENTENCE_BOUNDARY.split(normalized)
            if sentence.strip()
        ]
