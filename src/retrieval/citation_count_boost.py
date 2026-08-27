"""Deterministic citation-count re-scoring for hybrid retrieval results."""

import math

from retrieval.models import SearchResult

_COUNT_KEYS = ("citation_count", "cited_by_count")


class CitationCountBooster:
    """Re-score results by blending relevance with normalized log1p citations.

    Inspired by LlamaIndex metadata boost / Haystack score-boost postprocessors
    and scholarly ranking that prefers highly cited evidence. Each result keeps
    its prior relevance score ``old`` and receives a batch-normalized
    ``log1p(citation_count)`` signal read from ``chunk.metadata`` keys
    ``citation_count`` or ``cited_by_count`` (missing or invalid → ``0``).
    The blended score is:

    ```text
    new_score = (1 - alpha) * old + alpha * normalized_log1p
    ```

    Results are re-sorted by ``new_score`` descending; equal scores retain
    input order (stable sort). Input objects are not mutated.
    """

    def __init__(self, alpha: float = 0.3) -> None:
        """Create a citation-count booster.

        Args:
            alpha: Weight assigned to the normalized citation signal in
                ``[0.0, 1.0]``. The complementary weight ``1 - alpha`` is
                applied to the previous score.

        Raises:
            ValueError: If ``alpha`` is non-finite or outside ``[0.0, 1.0]``.
        """
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be a finite number within [0.0, 1.0]")
        self._alpha = alpha

    def boost(
        self,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Return results re-scored and ordered by blended citation signal.

        Citation counts are transformed with ``log1p``, then divided by the
        maximum ``log1p`` value in the batch so the signal lies in ``[0, 1]``.
        When every count is missing or zero, the citation signal is ``0.0`` for
        all rows. ``top_k`` truncates after sorting; ``None`` keeps every result.
        """
        limit = len(results) if top_k is None else min(top_k, len(results))
        if not results or limit <= 0:
            return []

        log_counts = [math.log1p(self._citation_count(result.chunk.metadata)) for result in results]
        peak = max(log_counts)
        signals = [value / peak if peak > 0.0 else 0.0 for value in log_counts]

        rescored: list[SearchResult] = []
        for result, signal in zip(results, signals, strict=True):
            score = (1.0 - self._alpha) * result.score + self._alpha * signal
            rescored.append(
                SearchResult(
                    chunk=result.chunk,
                    score=score,
                    retriever="citation_count_boost",
                    path=[*result.path, result.retriever],
                )
            )
        # Stable sort: Python's sorted is stable, so equal scores keep input order.
        return sorted(rescored, key=lambda result: result.score, reverse=True)[:limit]

    @staticmethod
    def _citation_count(metadata: dict[str, str]) -> float:
        for key in _COUNT_KEYS:
            raw = metadata.get(key, "").strip()
            if not raw:
                continue
            try:
                value = float(raw)
            except ValueError:
                continue
            if math.isfinite(value) and value >= 0.0:
                return value
        return 0.0
