"""Deterministic Self-RAG-style evidence reflection gate."""

import math
from dataclasses import dataclass
from enum import StrEnum
from itertools import combinations

from retrieval.models import SearchResult
from retrieval.sparse import meaningful_terms, tokenize

_NEGATIONS = frozenset({"no", "not", "never", "neither", "without"})
_CONFLICT_AXES: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "direction": (
        frozenset(
            {
                "better",
                "higher",
                "improve",
                "improved",
                "improves",
                "increase",
                "increased",
                "increases",
            }
        ),
        frozenset(
            {
                "decrease",
                "decreased",
                "decreases",
                "lower",
                "worse",
                "worsen",
                "worsened",
                "worsens",
            }
        ),
    ),
    "efficacy": (
        frozenset({"beneficial", "benefit", "benefits", "effective"}),
        frozenset({"harmful", "harms", "ineffective"}),
    ),
    "support": (
        frozenset(
            {
                "confirm",
                "confirmed",
                "confirms",
                "consistent",
                "support",
                "supported",
                "supports",
            }
        ),
        frozenset(
            {
                "contradict",
                "contradicted",
                "contradicts",
                "inconsistent",
                "refute",
                "refuted",
                "refutes",
                "unsupported",
            }
        ),
    ),
    "significance": (
        frozenset({"significant"}),
        frozenset({"insignificant", "nonsignificant"}),
    ),
}


class SelfRagSignal(StrEnum):
    """Safety-oriented synthesis recommendation."""

    SUPPORT = "SUPPORT"
    PARTIAL = "PARTIAL"
    REFUSE = "REFUSE"


@dataclass(frozen=True)
class EvidenceConflict:
    """Opposing lexical cues found in two query-relevant chunks."""

    left_chunk_id: str
    right_chunk_id: str
    axis: str


@dataclass(frozen=True)
class SelfRagDecision:
    """Reflection signal with inspectable coverage and conflict evidence."""

    signal: SelfRagSignal
    term_coverage: float
    covered_terms: tuple[str, ...]
    missing_terms: tuple[str, ...]
    conflicts: tuple[EvidenceConflict, ...]
    reason: str


class SelfRagReflectionGate:
    """Decide whether retrieved evidence can support grounded synthesis."""

    def __init__(
        self,
        support_threshold: float = 0.75,
        partial_threshold: float = 0.25,
    ) -> None:
        """Create a lexical reflection gate.

        Thresholds are inclusive fractions in ``[0, 1]``. A conflict always
        downgrades otherwise sufficient evidence from ``SUPPORT`` to
        ``PARTIAL``; evidence below ``partial_threshold`` returns ``REFUSE``.
        """
        for name, value in {
            "support_threshold": support_threshold,
            "partial_threshold": partial_threshold,
        }.items():
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a finite number within [0.0, 1.0]")
        if partial_threshold > support_threshold:
            raise ValueError("partial_threshold must not exceed support_threshold")
        self._support_threshold = support_threshold
        self._partial_threshold = partial_threshold

    def evaluate(self, query: str, results: list[SearchResult]) -> SelfRagDecision:
        """Return SUPPORT, PARTIAL, or REFUSE from coverage and conflict cues."""
        query_terms = meaningful_terms(query)
        if not query_terms:
            return SelfRagDecision(
                signal=SelfRagSignal.REFUSE,
                term_coverage=0.0,
                covered_terms=(),
                missing_terms=(),
                conflicts=(),
                reason="Query contains no content terms to reflect on.",
            )

        result_terms = [
            meaningful_terms(f"{result.chunk.title} {result.chunk.text}") for result in results
        ]
        all_result_terms: set[str] = set()
        for terms in result_terms:
            all_result_terms.update(terms)
        covered_terms = query_terms & all_result_terms
        missing_terms = query_terms - covered_terms
        coverage = len(covered_terms) / len(query_terms)
        conflicts = self._find_conflicts(query_terms, results, result_terms)

        if not covered_terms or coverage < self._partial_threshold:
            signal = SelfRagSignal.REFUSE
            reason = "Retrieved evidence does not cover enough query terms for grounded synthesis."
        elif coverage >= self._support_threshold and not conflicts:
            signal = SelfRagSignal.SUPPORT
            reason = "Retrieved evidence covers the query without detected lexical conflicts."
        elif conflicts:
            signal = SelfRagSignal.PARTIAL
            axes = ", ".join(sorted({conflict.axis for conflict in conflicts}))
            reason = f"Evidence coverage is usable, but opposing cues were detected for: {axes}."
        else:
            signal = SelfRagSignal.PARTIAL
            reason = "Retrieved evidence covers only part of the query."

        return SelfRagDecision(
            signal=signal,
            term_coverage=coverage,
            covered_terms=tuple(sorted(covered_terms)),
            missing_terms=tuple(sorted(missing_terms)),
            conflicts=conflicts,
            reason=reason,
        )

    def _find_conflicts(
        self,
        query_terms: set[str],
        results: list[SearchResult],
        result_terms: list[set[str]],
    ) -> tuple[EvidenceConflict, ...]:
        polarities = [
            self._conflict_polarities(f"{result.chunk.title} {result.chunk.text}")
            for result in results
        ]
        conflicts: list[EvidenceConflict] = []
        for left_index, right_index in combinations(range(len(results)), 2):
            left_overlap = query_terms & result_terms[left_index]
            right_overlap = query_terms & result_terms[right_index]
            if not left_overlap or not right_overlap or not left_overlap & right_overlap:
                continue
            for axis in _CONFLICT_AXES:
                left_sides = polarities[left_index].get(axis, set())
                right_sides = polarities[right_index].get(axis, set())
                if any(
                    left_side != right_side
                    for left_side in left_sides
                    for right_side in right_sides
                ):
                    conflicts.append(
                        EvidenceConflict(
                            left_chunk_id=results[left_index].chunk.chunk_id,
                            right_chunk_id=results[right_index].chunk.chunk_id,
                            axis=axis,
                        )
                    )
        return tuple(conflicts)

    @staticmethod
    def _conflict_polarities(text: str) -> dict[str, set[bool]]:
        tokens = tokenize(text)
        polarities: dict[str, set[bool]] = {}
        for index, token in enumerate(tokens):
            negated = any(previous in _NEGATIONS for previous in tokens[max(0, index - 2) : index])
            for axis, (positive_cues, negative_cues) in _CONFLICT_AXES.items():
                if token in positive_cues:
                    polarities.setdefault(axis, set()).add(not negated)
                elif token in negative_cues and not negated:
                    polarities.setdefault(axis, set()).add(False)
        return polarities
