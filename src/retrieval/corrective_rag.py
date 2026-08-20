"""Deterministic corrective-RAG relevance gating."""

import math
from dataclasses import dataclass
from enum import StrEnum

from retrieval.models import SearchResult
from retrieval.sparse import meaningful_terms


class CorrectiveRagSignal(StrEnum):
    """Recommended next step after lexical relevance grading."""

    KEEP = "keep"
    FILTER = "filter"
    RETRY_REWRITE = "retry_rewrite"


@dataclass(frozen=True)
class CorrectiveRagDecision:
    """Filtered retrieval results and the gate's recommended next step."""

    results: list[SearchResult]
    signal: CorrectiveRagSignal
    rewrite_hint: str | None = None


class CorrectiveRagGate:
    """Grade retrieved evidence by query-term coverage without an LLM."""

    def __init__(self, keep_threshold: float = 0.6, filter_threshold: float = 0.2) -> None:
        """Create a lexical corrective-RAG gate.

        Args:
            keep_threshold: Minimum query-term coverage for a strong result.
            filter_threshold: Minimum coverage for a usable borderline result.

        Raises:
            ValueError: If thresholds are non-finite, outside ``[0, 1]``, or
                ``filter_threshold`` is greater than ``keep_threshold``.
        """
        thresholds = {
            "keep_threshold": keep_threshold,
            "filter_threshold": filter_threshold,
        }
        for name, value in thresholds.items():
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a finite number within [0.0, 1.0]")
        if filter_threshold > keep_threshold:
            raise ValueError("filter_threshold must not exceed keep_threshold")
        self._keep_threshold = keep_threshold
        self._filter_threshold = filter_threshold

    def evaluate(
        self,
        query: str,
        results: list[SearchResult],
    ) -> CorrectiveRagDecision:
        """Return relevant results plus a keep, filter, or retry/rewrite signal.

        Coverage is the fraction of distinct, non-stopword query terms found in
        each result's title or text. If any result reaches ``keep_threshold``,
        only strong results are returned with ``KEEP``. Otherwise, borderline
        results reaching ``filter_threshold`` are returned with ``FILTER``.
        When no result qualifies, the empty result list carries
        ``RETRY_REWRITE`` and a deterministic rewrite hint.
        """
        query_terms = meaningful_terms(query)
        if not query_terms:
            return CorrectiveRagDecision(
                results=[],
                signal=CorrectiveRagSignal.RETRY_REWRITE,
                rewrite_hint="Add specific content terms before retrying retrieval.",
            )

        graded = [(self._coverage(query_terms, result), result) for result in results]
        strong = [result for coverage, result in graded if coverage >= self._keep_threshold]
        if strong:
            return CorrectiveRagDecision(results=strong, signal=CorrectiveRagSignal.KEEP)

        borderline = [result for coverage, result in graded if coverage >= self._filter_threshold]
        if borderline:
            return CorrectiveRagDecision(
                results=borderline,
                signal=CorrectiveRagSignal.FILTER,
            )

        terms = ", ".join(sorted(query_terms))
        return CorrectiveRagDecision(
            results=[],
            signal=CorrectiveRagSignal.RETRY_REWRITE,
            rewrite_hint=f"Try synonyms or broader terminology for: {terms}.",
        )

    @staticmethod
    def _coverage(query_terms: set[str], result: SearchResult) -> float:
        result_terms = meaningful_terms(f"{result.chunk.title} {result.chunk.text}")
        return len(query_terms & result_terms) / len(query_terms)
