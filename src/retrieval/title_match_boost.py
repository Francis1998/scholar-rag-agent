"""Deterministic title-only lexical re-scoring for hybrid retrieval results."""

import math

from retrieval.models import SearchResult
from retrieval.sparse import meaningful_terms


class TitleMatchBooster:
    """Re-score results by blending relevance with query-title term overlap.

    Inspired by LlamaIndex title-metadata boost and Haystack keyword-in-title
    stages. Unlike :class:`~retrieval.lexical_overlap_boost.LexicalOverlapBooster`,
    which measures Jaccard overlap over chunk **title + text**, this booster
    uses ``chunk.title`` only. Each result keeps its prior relevance score
    ``old`` and receives a Jaccard overlap signal between query and title
    terms (via :func:`retrieval.sparse.meaningful_terms`). The blended score
    is:

    ```text
    new_score = (1 - alpha) * old + alpha * overlap
    ```

    Results are re-sorted by ``new_score`` descending; equal scores retain
    input order (stable sort). Input objects are not mutated.
    """

    def __init__(self, alpha: float = 0.3) -> None:
        """Create a title-match booster.

        Args:
            alpha: Weight assigned to Jaccard title overlap in ``[0.0, 1.0]``.
                The complementary weight ``1 - alpha`` is applied to the
                previous score.

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
        """Return results re-scored and ordered by blended title overlap.

        Jaccard overlap is ``|query ∩ title| / |query U title|`` over terms
        via :func:`meaningful_terms` on the query and ``chunk.title`` only.
        When both sets are empty, overlap is ``0.0``. ``top_k`` truncates
        after sorting; ``None`` keeps every result.
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
                    retriever="title_match_boost",
                    path=[*result.path, result.retriever],
                )
            )
        # Stable sort: Python's sorted is stable, so equal scores keep input order.
        return sorted(rescored, key=lambda result: result.score, reverse=True)[:limit]

    @staticmethod
    def _jaccard(query_terms: set[str], result: SearchResult) -> float:
        title_terms = meaningful_terms(result.chunk.title)
        if not query_terms and not title_terms:
            return 0.0
        union = query_terms | title_terms
        if not union:
            return 0.0
        return len(query_terms & title_terms) / len(union)
