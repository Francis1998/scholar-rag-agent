"""Gate retrieval hits by a minimum relevance score threshold."""

import math

from retrieval.models import SearchResult


class ScoreThresholdGate:
    """Drop hits whose score falls below ``min_score``.

    Inspired by Haystack ``ScoreThreshold`` and LlamaIndex
    ``SimilarityPostprocessor`` score floors. Results at or above the
    threshold are kept with rewritten provenance; weaker hits are dropped.
    Distinct from :class:`~retrieval.cross_encoder_gate.CrossEncoderGate`
    (proxy rescoring) and soft boosters. Inputs are not mutated. Local
    postprocessor for GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2
    pipelines (not a DOI connector).
    """

    def __init__(self, min_score: float = 0.0) -> None:
        """Create a score-threshold gate.

        Args:
            min_score: Inclusive keep threshold (finite).

        Raises:
            ValueError: If ``min_score`` is non-finite.
        """
        if not math.isfinite(min_score):
            raise ValueError("min_score must be a finite number")
        self._min_score = min_score

    def gate(
        self,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Return results whose score meets ``min_score``."""
        if not results:
            return []
        limit = len(results) if top_k is None else min(top_k, len(results))
        if limit <= 0:
            return []

        kept: list[SearchResult] = []
        for result in results:
            if result.score < self._min_score:
                continue
            kept.append(
                SearchResult(
                    chunk=result.chunk,
                    score=result.score,
                    retriever="score_threshold_gate",
                    path=[*result.path, result.retriever],
                )
            )
            if len(kept) >= limit:
                break
        return kept
