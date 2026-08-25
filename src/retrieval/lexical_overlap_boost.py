"""Deterministic lexical-overlap re-scoring for hybrid retrieval results."""

import math

from retrieval.models import SearchResult
from retrieval.sparse import meaningful_terms


class LexicalOverlapBooster:
    """Re-score results by blending relevance with query-chunk term overlap.

    Inspired by hybrid BM25 / Haystack keyword boost stages: each result keeps
    its prior relevance score ``old`` and receives a Jaccard overlap signal
    between query and chunk terms (via :func:`retrieval.sparse.meaningful_terms`).
    The blended score is:

    ```text
    new_score = (1 - alpha) * old + alpha * overlap
    ```

    Results are re-sorted by ``new_score`` descending; equal scores retain
    input order (stable sort). Input objects are not mutated.
    """

    def __init__(self, alpha: float = 0.3) -> None:
        """Create a lexical overlap booster.

        Args:
            alpha: Weight assigned to Jaccard overlap in ``[0.0, 1.0]``. The
                complementary weight ``1 - alpha`` is applied to the previous
                score.

        Raises:
            ValueError: If ``alpha`` is non-finite or outside ``[0.0, 1.0]``.
        """
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be a finite number within [0.0, 1.0]")
        self._alpha = alpha

    def boost(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Return results re-scored and ordered by blended lexical overlap.

        Jaccard overlap is ``|query ∩ chunk| / |query U chunk|`` over
        title+text terms via :func:`meaningful_terms`. When both sets are
        empty, overlap is ``0.0``. ``top_k`` truncates after sorting; ``None``
        keeps every result.
        """
        limit = len(results) if top_k is None else min(top_k, len(results))
        if not results or limit <= 0:
            return []

        query_terms = meaningful_terms(query)
        rescored: list[SearchResult] = []
        for result in results:
            overlap = self._jaccard(query_terms, result)
            score = (1.0 - self._alpha) * result.score + self._alpha * overlap
            rescored.append(
                SearchResult(
                    chunk=result.chunk,
                    score=score,
                    retriever="lexical_overlap_boost",
                    path=[*result.path, result.retriever],
                )
            )
        # Stable sort: Python's sorted is stable, so equal scores keep input order.
        return sorted(rescored, key=lambda result: result.score, reverse=True)[:limit]

    @staticmethod
    def _jaccard(query_terms: set[str], result: SearchResult) -> float:
        chunk_terms = meaningful_terms(f"{result.chunk.title} {result.chunk.text}")
        if not query_terms and not chunk_terms:
            return 0.0
        union = query_terms | chunk_terms
        if not union:
            return 0.0
        return len(query_terms & chunk_terms) / len(union)
