"""Cluster evidence passages that lexically conflict on a claim.

Offline heuristics partition passages into supporting vs contradicting clusters
using negation/antonym cues and claim-term overlap. Fills an Elicit
conflicting-evidence table gap without an LLM or network call. Optional later
narrative can use GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.

Distinct from :class:`~retrieval.claim_support.ClaimSupportScorer` (per-passage
support strength) and
:class:`~retrieval.claim_verification_gate.ClaimVerificationGate`
(answer-level claim groundedness). Also distinct from
:class:`~retrieval.evidence_conflict.EvidenceConflictDetector` (pairwise
snippet polarity clashes without a caller claim).
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from retrieval.sparse import meaningful_terms, tokenize

_WHITESPACE = re.compile(r"\s+")

ContradictionLabel = Literal["supporting", "contradicting", "neutral"]

# Explicit lexical negation cues (fail*/contradict* matched via prefixes).
_NEGATION_WORDS = frozenset({"no", "not", "never"})
_NEGATION_PREFIXES = ("fail", "contradict", "refut", "deni", "disprov")

# Antonym / polarity cue pairs (positive claim cue ↔ opposing cue).
_ANTONYM_PAIRS: tuple[tuple[frozenset[str], frozenset[str]], ...] = (
    (
        frozenset(
            {
                "better",
                "higher",
                "improve",
                "improved",
                "improves",
                "improving",
                "increase",
                "increased",
                "increases",
                "increasing",
                "positive",
                "effective",
                "beneficial",
                "benefit",
                "benefits",
                "support",
                "supports",
                "supported",
                "confirm",
                "confirms",
                "confirmed",
            }
        ),
        frozenset(
            {
                "worse",
                "worsen",
                "worsens",
                "worsened",
                "lower",
                "decrease",
                "decreased",
                "decreases",
                "decreasing",
                "negative",
                "ineffective",
                "harmful",
                "harms",
                "refute",
                "refutes",
                "refuted",
                "contradict",
                "contradicts",
                "contradicted",
                "unsupported",
            }
        ),
    ),
)


@dataclass(frozen=True)
class ContradictionClusterResult:
    """Claim-centered supporting vs contradicting evidence cluster."""

    claim: str
    supporting_ids: tuple[str, ...]
    contradicting_ids: tuple[str, ...]
    tension_score: float
    labels: tuple[tuple[str, ContradictionLabel], ...]


class ContradictionClusterFinder:
    """Cluster passages that lexically conflict on a claim.

    Offline claim-tension clusters for GPT-5.5 / Claude Sonnet 4.6 /
    Gemini 3.x / Kimi K2 pipelines — Elicit conflicting-evidence table gap.
    Passages with high claim overlap and no negation/antonym cues are
    supporting; passages with negation (``not``/``no``/``never``/``fail*``/
    ``contradict*``) or antonym polarity plus topical overlap are
    contradicting. Distinct from
    :class:`~retrieval.claim_support.ClaimSupportScorer` (support strength)
    and :class:`~retrieval.claim_verification_gate.ClaimVerificationGate`
    (answer groundedness).
    """

    def __init__(
        self,
        *,
        support_overlap: float = 0.35,
        contradict_overlap: float = 0.15,
    ) -> None:
        """Create a claim contradiction cluster finder.

        Args:
            support_overlap: Inclusive claim-term coverage floor for the
                ``supporting`` label when no negation/antonym cues are present
                (``[0.0, 1.0]``).
            contradict_overlap: Inclusive claim-term coverage floor for the
                ``contradicting`` label when negation/antonym cues are present
                (``[0.0, 1.0]``); must be ``<= support_overlap``.

        Raises:
            ValueError: If either knob is non-finite or outside ``[0.0, 1.0]``,
                or ``contradict_overlap`` exceeds ``support_overlap``.
        """
        for name, value in (
            ("support_overlap", support_overlap),
            ("contradict_overlap", contradict_overlap),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a finite number within [0.0, 1.0]")
        if contradict_overlap > support_overlap:
            raise ValueError("contradict_overlap must be <= support_overlap")
        self._support_overlap = support_overlap
        self._contradict_overlap = contradict_overlap

    def find(
        self,
        claim: str,
        passages: Sequence[Mapping[str, object] | str],
    ) -> ContradictionClusterResult:
        """Return supporting/contradicting clusters for ``claim``.

        Empty claims raise ``ValueError``. Empty passages yield empty id lists,
        ``tension_score=0.0``, and empty ``labels``. Inputs are not mutated.
        """
        cleaned = self._clean_claim(claim)
        claim_terms = frozenset(meaningful_terms(cleaned))
        if not passages:
            return ContradictionClusterResult(
                claim=cleaned,
                supporting_ids=(),
                contradicting_ids=(),
                tension_score=0.0,
                labels=(),
            )

        supporting: list[str] = []
        contradicting: list[str] = []
        labels: list[tuple[str, ContradictionLabel]] = []

        for index, passage in enumerate(passages):
            evidence_id, text = self._coerce_passage(passage, index)
            passage_terms = frozenset(meaningful_terms(text))
            tokens = tokenize(text)
            coverage = self._coverage(claim_terms, passage_terms)
            has_negation = self._has_negation_cue(tokens)
            has_antonym = self._has_antonym_clash(claim_terms, passage_terms)
            conflicts = has_negation or has_antonym

            if conflicts and coverage >= self._contradict_overlap:
                label: ContradictionLabel = "contradicting"
                contradicting.append(evidence_id)
            elif (not conflicts) and coverage >= self._support_overlap:
                label = "supporting"
                supporting.append(evidence_id)
            else:
                label = "neutral"
            labels.append((evidence_id, label))

        return ContradictionClusterResult(
            claim=cleaned,
            supporting_ids=tuple(supporting),
            contradicting_ids=tuple(contradicting),
            tension_score=self._tension(len(supporting), len(contradicting)),
            labels=tuple(labels),
        )

    @staticmethod
    def _tension(n_supporting: int, n_contradicting: int) -> float:
        """Balance-aware tension in ``[0, 1]``; needs both sides non-empty."""
        total = n_supporting + n_contradicting
        if n_supporting <= 0 or n_contradicting <= 0 or total <= 0:
            return 0.0
        # 1.0 when counts are equal; lower when one side dominates.
        balance = 2.0 * min(n_supporting, n_contradicting) / total
        # Soft presence boost so larger clusters do not collapse to tiny scores.
        presence = min(1.0, total / 4.0)
        return min(1.0, balance * (0.5 + 0.5 * presence))

    @staticmethod
    def _coverage(claim_terms: frozenset[str], passage_terms: frozenset[str]) -> float:
        if not claim_terms:
            return 0.0
        return len(claim_terms & passage_terms) / len(claim_terms)

    @staticmethod
    def _has_negation_cue(tokens: Sequence[str]) -> bool:
        for token in tokens:
            if token in _NEGATION_WORDS:
                return True
            if any(token.startswith(prefix) for prefix in _NEGATION_PREFIXES):
                return True
        return False

    @staticmethod
    def _has_antonym_clash(
        claim_terms: frozenset[str],
        passage_terms: frozenset[str],
    ) -> bool:
        for positive, negative in _ANTONYM_PAIRS:
            claim_pos = bool(claim_terms & positive)
            claim_neg = bool(claim_terms & negative)
            passage_pos = bool(passage_terms & positive)
            passage_neg = bool(passage_terms & negative)
            if claim_pos and passage_neg and not passage_pos:
                return True
            if claim_neg and passage_pos and not passage_neg:
                return True
        return False

    @staticmethod
    def _clean_claim(claim: str) -> str:
        cleaned = _WHITESPACE.sub(" ", str(claim).strip())
        if not cleaned:
            raise ValueError("claim must be a non-empty string")
        return cleaned

    @staticmethod
    def _coerce_passage(
        passage: Mapping[str, object] | str,
        index: int,
    ) -> tuple[str, str]:
        if isinstance(passage, str):
            text = _WHITESPACE.sub(" ", passage.strip())
            return f"passage-{index}", text

        evidence_id: str | None = None
        for key in ("evidence_id", "id", "chunk_id", "document_id"):
            raw = passage.get(key)
            if raw is None or raw == "":
                continue
            text_id = str(raw).strip()
            if text_id:
                evidence_id = text_id
                break
        if evidence_id is None:
            evidence_id = f"passage-{index}"

        parts: list[str] = []
        for key in ("text", "passage", "content", "abstract", "evidence", "body", "title"):
            raw = passage.get(key)
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
