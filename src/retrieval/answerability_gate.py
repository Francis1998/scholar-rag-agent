"""Deterministic lexical answerability gate for retrieved evidence."""

import math
from dataclasses import dataclass

from retrieval.models import SearchResult
from retrieval.sparse import meaningful_terms


@dataclass(frozen=True)
class AnswerabilityReport:
    """Aggregate and per-result answerability scores for a retrieval set.

    ``answerability`` is the mean of ``result_scores`` (``0.0`` when there are
    no results or the query has no meaningful terms). ``kept_count`` /
    ``dropped_count`` reflect the outcome of applying ``min_score`` to those
    per-result scores; they do not encode the aggregate threshold check used
    by :meth:`AnswerabilityGate.filter`.
    """

    answerability: float
    result_scores: tuple[float, ...]
    kept_count: int
    dropped_count: int


class AnswerabilityGate:
    """Decide whether retrieved chunks can lexically support answering a query.

    Inspired by LlamaIndex answerability checks and Self-RAG "can answer?"
    gates, but fully deterministic: each result is scored by the fraction of
    distinct non-stopword query terms that appear in the chunk title or text.
    Aggregate answerability is the mean of those per-result scores. No network
    or LLM call is required.
    """

    def __init__(
        self,
        min_score: float = 0.2,
        answerability_threshold: float = 0.3,
    ) -> None:
        """Create a lexical answerability gate.

        Args:
            min_score: Inclusive per-result coverage below which a chunk is
                dropped by :meth:`filter` when the aggregate gate passes.
            answerability_threshold: Inclusive mean coverage below which
                :meth:`filter` returns an empty list for the whole batch.

        Raises:
            ValueError: If either threshold is non-finite or outside
                ``[0.0, 1.0]``.
        """
        for name, value in {
            "min_score": min_score,
            "answerability_threshold": answerability_threshold,
        }.items():
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a finite number within [0.0, 1.0]")
        self._min_score = min_score
        self._answerability_threshold = answerability_threshold

    def score(self, query: str, results: list[SearchResult]) -> AnswerabilityReport:
        """Return per-result coverage scores and aggregate answerability.

        Coverage for one result is ``|query_terms ∩ chunk_terms| /
        |query_terms|`` using :func:`retrieval.sparse.meaningful_terms`. When
        the query has no meaningful terms, every result scores ``0.0`` and
        aggregate answerability is ``0.0``.
        """
        query_terms = meaningful_terms(query)
        if not results:
            return AnswerabilityReport(
                answerability=0.0,
                result_scores=(),
                kept_count=0,
                dropped_count=0,
            )
        if not query_terms:
            scores = tuple(0.0 for _ in results)
            return AnswerabilityReport(
                answerability=0.0,
                result_scores=scores,
                kept_count=0,
                dropped_count=len(results),
            )

        scores = tuple(self._coverage(query_terms, result) for result in results)
        kept_count = sum(1 for score in scores if score >= self._min_score)
        return AnswerabilityReport(
            answerability=sum(scores) / len(scores),
            result_scores=scores,
            kept_count=kept_count,
            dropped_count=len(scores) - kept_count,
        )

    def filter(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        """Drop weak chunks, or return ``[]`` when the batch is unanswerable.

        First compute :meth:`score`. If aggregate ``answerability`` is below
        ``answerability_threshold``, return an empty list even if some
        individual chunks would pass ``min_score``. Otherwise return the
        original result objects whose per-result score is at least
        ``min_score``, preserving input order and identity.
        """
        report = self.score(query, results)
        if report.answerability < self._answerability_threshold:
            return []
        return [
            result
            for result, score in zip(results, report.result_scores, strict=True)
            if score >= self._min_score
        ]

    @staticmethod
    def _coverage(query_terms: set[str], result: SearchResult) -> float:
        chunk_terms = meaningful_terms(f"{result.chunk.title} {result.chunk.text}")
        return len(query_terms & chunk_terms) / len(query_terms)
