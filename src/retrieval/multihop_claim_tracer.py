"""Multi-hop claim → evidence span → cited-document trace reports.

Produces an inspectable claim path for literature synthesis. Distinct from
:class:`~retrieval.multihop.MultiHopRetriever` (entity-chain retrieval),
:class:`~retrieval.claim_verification_gate.ClaimVerificationGate` (support /
refuse gate), and
:class:`~retrieval.citation_graph.CitationGraphExpander` (citation-edge
expansion). Fills a PaperQA-style claim verification UI gap with a local,
deterministic hop report. Optional later narrative can use GPT-5.5 /
Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

from retrieval.models import Chunk, SearchResult
from retrieval.sparse import meaningful_terms

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")
_CITATION_MARKER = re.compile(
    r"\[(\d+)\]|\(([^)]+,\s*\d{4}[a-z]?)\)",
    re.IGNORECASE,
)
_DOI_FIELDS = ("doi", "paper_doi", "work_doi")


@dataclass(frozen=True)
class ClaimHop:
    """One hop in a claim evidence path."""

    hop_index: int
    hop_type: str
    label: str
    detail: str
    score: float
    document_id: str = ""
    chunk_id: str = ""


@dataclass(frozen=True)
class ClaimTrace:
    """Full multi-hop path for one claim."""

    claim: str
    hops: tuple[ClaimHop, ...]
    grounded: bool
    support_score: float

    def to_markdown(self) -> str:
        """Render the claim trace as a markdown path report."""
        status = "grounded" if self.grounded else "ungrounded"
        lines = [
            f"## Claim trace ({status}, score={self.support_score:.3f})",
            "",
            f"**Claim:** {self.claim}",
            "",
            "_Deterministic multi-hop claim path. Optional narrative can use "
            "GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 "
            "(PaperQA claim-verification UI gap)._",
            "",
        ]
        if not self.hops:
            lines.append("- _(no hops)_")
            return "\n".join(lines).rstrip() + "\n"
        for hop in self.hops:
            lines.append(
                f"{hop.hop_index}. **{hop.hop_type}** — {hop.label}"
                + (f" (`{hop.document_id}`)" if hop.document_id else "")
            )
            if hop.detail:
                lines.append(f"   - {hop.detail}")
        return "\n".join(lines).rstrip() + "\n"


class MultiHopClaimTracer:
    """Trace a claim through supporting spans into cited documents.

    Not a retrieval gate: always returns a report/path rather than filtering
    hits. Hop 0 is the claim; hop 1+ are supporting spans ranked by lexical
    overlap; final hops surface cited document identities from chunk
    metadata / citation markers. Local tracer for GPT-5.5 / Claude Sonnet 4.6 /
    Gemini 3.x / Kimi K2 pipelines — PaperQA claim verification UI gap.
    """

    def __init__(
        self,
        support_threshold: float = 0.35,
        max_span_hops: int = 3,
    ) -> None:
        """Create a multi-hop claim tracer.

        Args:
            support_threshold: Inclusive fraction of claim terms that must
                appear in at least one supporting span for ``grounded=True``.
            max_span_hops: Maximum supporting-span hops to emit (``>= 1``).

        Raises:
            ValueError: If knobs are non-finite / out of range.
        """
        if not math.isfinite(support_threshold) or not 0.0 <= support_threshold <= 1.0:
            raise ValueError("support_threshold must be a finite number within [0.0, 1.0]")
        if not isinstance(max_span_hops, int) or max_span_hops < 1:
            raise ValueError("max_span_hops must be an int >= 1")
        self._support_threshold = support_threshold
        self._max_span_hops = max_span_hops

    def trace(
        self,
        claim: str,
        evidence: Sequence[SearchResult | Chunk],
    ) -> ClaimTrace:
        """Return a claim → spans → cited-docs path for ``claim``.

        Inputs are not mutated. Empty claims yield an ungrounded empty path.
        """
        cleaned = " ".join(claim.split()).strip()
        if not cleaned:
            return ClaimTrace(claim="", hops=(), grounded=False, support_score=0.0)

        claim_terms = meaningful_terms(cleaned)
        hops: list[ClaimHop] = [
            ClaimHop(
                hop_index=0,
                hop_type="claim",
                label=cleaned,
                detail=f"{len(claim_terms)} claim terms",
                score=1.0,
            )
        ]

        scored: list[tuple[float, str, SearchResult | Chunk, str]] = []
        for item in evidence:
            chunk = item.chunk if isinstance(item, SearchResult) else item
            best_span = ""
            best_score = 0.0
            for sentence in self._sentences(chunk.text):
                overlap = self._coverage(claim_terms, meaningful_terms(sentence))
                shorter_tie = overlap == best_score and len(sentence) < len(best_span)
                if overlap > best_score or shorter_tie:
                    best_score = overlap
                    best_span = sentence
            if best_score <= 0.0 and chunk.text.strip():
                best_span = chunk.text.strip()[:160]
                best_score = self._coverage(claim_terms, meaningful_terms(chunk.text))
            if best_span:
                scored.append((best_score, chunk.chunk_id, item, best_span))

        scored.sort(key=lambda row: (-row[0], row[1]))
        covered: set[str] = set()
        cited_docs: list[tuple[str, str, float]] = []
        seen_docs: set[str] = set()

        for score, _chunk_id, item, span in scored[: self._max_span_hops]:
            chunk = item.chunk if isinstance(item, SearchResult) else item
            span_terms = meaningful_terms(span)
            covered |= claim_terms & span_terms
            hops.append(
                ClaimHop(
                    hop_index=len(hops),
                    hop_type="supporting_span",
                    label=span if len(span) <= 200 else span[:197] + "...",
                    detail=f"coverage={score:.3f}",
                    score=score,
                    document_id=chunk.document_id,
                    chunk_id=chunk.chunk_id,
                )
            )
            doc_label = self._document_label(chunk)
            if doc_label not in seen_docs:
                seen_docs.add(doc_label)
                cited_docs.append((doc_label, chunk.document_id, score))
            for marker in self._citation_markers(f"{span} {chunk.text}"):
                key = f"cite:{marker}"
                if key not in seen_docs:
                    seen_docs.add(key)
                    cited_docs.append((f"citation {marker}", chunk.document_id, score))

        for label, document_id, score in cited_docs:
            hops.append(
                ClaimHop(
                    hop_index=len(hops),
                    hop_type="cited_document",
                    label=label,
                    detail="linked from supporting span",
                    score=score,
                    document_id=document_id,
                )
            )

        support_score = (len(covered) / len(claim_terms)) if claim_terms else 0.0
        grounded = support_score >= self._support_threshold and any(
            hop.hop_type == "supporting_span" for hop in hops
        )
        return ClaimTrace(
            claim=cleaned,
            hops=tuple(hops),
            grounded=grounded,
            support_score=support_score,
        )

    def trace_many(
        self,
        claims: Sequence[str],
        evidence: Sequence[SearchResult | Chunk],
    ) -> list[ClaimTrace]:
        """Trace each claim independently against the same evidence pool."""
        return [self.trace(claim, evidence) for claim in claims]

    @staticmethod
    def _coverage(claim_terms: set[str], span_terms: set[str]) -> float:
        if not claim_terms:
            return 0.0
        return len(claim_terms & span_terms) / len(claim_terms)

    @staticmethod
    def _sentences(text: str) -> list[str]:
        parts = [part.strip() for part in _SENTENCE_BOUNDARY.split(text) if part.strip()]
        return parts or ([text.strip()] if text.strip() else [])

    @staticmethod
    def _citation_markers(text: str) -> list[str]:
        markers: list[str] = []
        for match in _CITATION_MARKER.finditer(text):
            numeric, author_year = match.groups()
            marker = numeric or (author_year or "").strip()
            if marker and marker not in markers:
                markers.append(marker)
        return markers

    @staticmethod
    def _document_label(chunk: Chunk) -> str:
        for field in _DOI_FIELDS:
            value = chunk.metadata.get(field, "").strip()
            if value:
                return value
        title = chunk.title.strip()
        if title:
            return title
        return chunk.document_id
