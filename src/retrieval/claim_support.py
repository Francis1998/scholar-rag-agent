"""Score how strongly evidence passages support a scholarly claim.

Offline lexical overlap and claim-term coverage heuristics produce per-passage
support strength labels (``supported`` / ``partial`` / ``unsupported``). Fills
an Elicit / Semantic Scholar claim-table gap without an LLM or network call.
Optional later narrative can use GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
Kimi K2.

Distinct from :class:`~retrieval.claim_verification_gate.ClaimVerificationGate`
(answer-level claim groundedness) and
:class:`~retrieval.citation_groundedness_score.CitationGroundednessScorer`
(inline citation-marker grounding).
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from retrieval.sparse import meaningful_terms

_WHITESPACE = re.compile(r"\s+")

ClaimSupportLabel = Literal["supported", "partial", "unsupported"]


@dataclass(frozen=True)
class ClaimSupportResult:
    """Support-strength score for one claim↔evidence pairing."""

    claim: str
    support_score: float
    matched_terms: tuple[str, ...]
    evidence_id: str | None
    label: ClaimSupportLabel


class ClaimSupportScorer:
    """Score claim↔evidence support strength with offline lexical heuristics.

    Blends claim-term coverage with Jaccard overlap to rank evidence passages
    by how strongly they support a caller-provided claim. Offline scorer for
    GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 pipelines — Elicit /
    Semantic Scholar claim-table gap. Distinct from
    :class:`~retrieval.claim_verification_gate.ClaimVerificationGate`
    (answer groundedness) and
    :class:`~retrieval.citation_groundedness_score.CitationGroundednessScorer`
    (citation-marker grounding).
    """

    def __init__(
        self,
        *,
        coverage_weight: float = 0.7,
        supported_threshold: float = 0.6,
        partial_threshold: float = 0.25,
    ) -> None:
        """Create a claim↔evidence support scorer.

        Args:
            coverage_weight: Blend weight for claim-term coverage versus Jaccard
                overlap (``[0.0, 1.0]``). Score is
                ``coverage_weight * coverage + (1 - coverage_weight) * jaccard``.
            supported_threshold: Inclusive score floor for the ``supported``
                label (``[0.0, 1.0]``).
            partial_threshold: Inclusive score floor for the ``partial`` label;
                must be ``<= supported_threshold``.

        Raises:
            ValueError: If any knob is non-finite or outside ``[0.0, 1.0]``, or
                ``partial_threshold`` exceeds ``supported_threshold``.
        """
        for name, value in (
            ("coverage_weight", coverage_weight),
            ("supported_threshold", supported_threshold),
            ("partial_threshold", partial_threshold),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a finite number within [0.0, 1.0]")
        if partial_threshold > supported_threshold:
            raise ValueError("partial_threshold must be <= supported_threshold")
        self._coverage_weight = coverage_weight
        self._supported_threshold = supported_threshold
        self._partial_threshold = partial_threshold

    def score(
        self,
        claim: str,
        evidence_passages: Sequence[Mapping[str, object] | str],
    ) -> tuple[ClaimSupportResult, ...]:
        """Return per-passage support results ranked by descending score.

        Empty claims raise ``ValueError``. Empty evidence yields a single
        ``unsupported`` result with score ``0.0`` and ``evidence_id=None``.
        Inputs are not mutated.
        """
        cleaned = self._clean_claim(claim)
        if not evidence_passages:
            return (
                ClaimSupportResult(
                    claim=cleaned,
                    support_score=0.0,
                    matched_terms=(),
                    evidence_id=None,
                    label="unsupported",
                ),
            )

        results = [self._score_passage(cleaned, passage) for passage in evidence_passages]
        results.sort(
            key=lambda item: (
                -item.support_score,
                item.evidence_id or "",
                " ".join(item.matched_terms),
            )
        )
        return tuple(results)

    def score_one(
        self,
        claim: str,
        evidence: Mapping[str, object] | str,
    ) -> ClaimSupportResult:
        """Return support strength for a single evidence passage."""
        cleaned = self._clean_claim(claim)
        return self._score_passage(cleaned, evidence)

    def _score_passage(
        self,
        claim: str,
        evidence: Mapping[str, object] | str,
    ) -> ClaimSupportResult:
        evidence_id, text = self._coerce_evidence(evidence)
        claim_terms = frozenset(meaningful_terms(claim))
        evidence_terms = frozenset(meaningful_terms(text))
        if not claim_terms:
            return ClaimSupportResult(
                claim=claim,
                support_score=0.0,
                matched_terms=(),
                evidence_id=evidence_id,
                label="unsupported",
            )

        matched = claim_terms & evidence_terms
        coverage = len(matched) / len(claim_terms)
        union = claim_terms | evidence_terms
        jaccard = len(matched) / len(union) if union else 0.0
        support_score = self._coverage_weight * coverage + (1.0 - self._coverage_weight) * jaccard
        return ClaimSupportResult(
            claim=claim,
            support_score=support_score,
            matched_terms=tuple(sorted(matched)),
            evidence_id=evidence_id,
            label=self._label(support_score),
        )

    def _label(self, support_score: float) -> ClaimSupportLabel:
        if support_score >= self._supported_threshold:
            return "supported"
        if support_score >= self._partial_threshold:
            return "partial"
        return "unsupported"

    @staticmethod
    def _clean_claim(claim: str) -> str:
        cleaned = _WHITESPACE.sub(" ", str(claim).strip())
        if not cleaned:
            raise ValueError("claim must be a non-empty string")
        return cleaned

    @staticmethod
    def _coerce_evidence(
        evidence: Mapping[str, object] | str,
    ) -> tuple[str | None, str]:
        if isinstance(evidence, str):
            return None, _WHITESPACE.sub(" ", evidence.strip())

        evidence_id: str | None = None
        for key in ("evidence_id", "id", "chunk_id", "document_id"):
            raw = evidence.get(key)
            if raw is None or raw == "":
                continue
            text_id = str(raw).strip()
            if text_id:
                evidence_id = text_id
                break

        parts: list[str] = []
        for key in ("text", "passage", "content", "abstract", "evidence", "body", "title"):
            raw = evidence.get(key)
            if raw is None or raw == "":
                continue
            if isinstance(raw, (list, tuple)):
                parts.extend(str(item) for item in raw if str(item).strip())
            else:
                part = str(raw).strip()
                if part:
                    parts.append(part)
        text = _WHITESPACE.sub(" ", " ".join(parts).strip())
        return evidence_id, text
