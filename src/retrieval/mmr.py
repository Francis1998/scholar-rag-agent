"""Maximal Marginal Relevance (MMR) diversity re-ranking.

Hybrid retrieval frequently returns several near-duplicate chunks (e.g. the same
passage repeated across paper versions or overlapping sections). Feeding those
into a limited context window wastes budget on redundant text and starves the
answer of complementary evidence. MMR re-orders results to balance query
relevance against novelty relative to already-selected chunks, using the classic
Carbonell & Goldstein (1998) objective:

    score(d) = lambda * relevance(d) - (1 - lambda) * max_{s in selected} sim(d, s)

Similarity here is a dependency-free lexical Jaccard overlap over the shared
tokenizer, so the re-ranking is deterministic and requires no embedding model.
"""

from retrieval.models import SearchResult
from retrieval.sparse import tokenize


class MMRDiversifier:
    """Re-rank search results for relevance/diversity balance via MMR."""

    def __init__(self, lambda_param: float = 0.7) -> None:
        """Create a diversifier.

        Args:
            lambda_param: Trade-off in ``[0.0, 1.0]``. ``1.0`` is pure relevance
                (no diversification); ``0.0`` is pure novelty.

        Raises:
            ValueError: If ``lambda_param`` is outside ``[0.0, 1.0]``.
        """
        if not 0.0 <= lambda_param <= 1.0:
            raise ValueError(f"lambda_param must be within [0.0, 1.0], got {lambda_param}")
        self._lambda = lambda_param

    def diversify(
        self, results: list[SearchResult], top_k: int | None = None
    ) -> list[SearchResult]:
        """Return results re-ordered by Maximal Marginal Relevance.

        Args:
            results: Candidate results, assumed already scored for relevance.
            top_k: Maximum number of results to return; defaults to all.

        Returns:
            Results ordered greedily by the MMR objective. Ties preserve the
            input order for determinism.
        """
        if not results:
            return []
        limit = len(results) if top_k is None else min(top_k, len(results))
        if limit <= 0:
            return []

        relevance = self._normalized_relevance([result.score for result in results])
        token_sets = [set(tokenize(result.chunk.text)) for result in results]

        selected_indices: list[int] = []
        candidate_indices = list(range(len(results)))
        while candidate_indices and len(selected_indices) < limit:
            best_index = candidate_indices[0]
            best_mmr: float | None = None
            for index in candidate_indices:
                if not selected_indices:
                    mmr = relevance[index]
                else:
                    max_similarity = max(
                        self._jaccard(token_sets[index], token_sets[selected])
                        for selected in selected_indices
                    )
                    mmr = self._lambda * relevance[index] - (1.0 - self._lambda) * max_similarity
                if best_mmr is None or mmr > best_mmr:
                    best_mmr = mmr
                    best_index = index
            selected_indices.append(best_index)
            candidate_indices.remove(best_index)

        return [results[index] for index in selected_indices]

    @staticmethod
    def _normalized_relevance(scores: list[float]) -> list[float]:
        """Min-max normalize scores into ``[0, 1]`` for the MMR objective.

        When every score is equal there is no relevance signal to separate the
        candidates, so all relevances are treated as ``1.0`` and diversity alone
        drives selection.

        Args:
            scores: Raw relevance scores.

        Returns:
            Normalized relevance values aligned with ``scores``.
        """
        lowest = min(scores)
        highest = max(scores)
        span = highest - lowest
        if span == 0:
            return [1.0 for _ in scores]
        return [(score - lowest) / span for score in scores]

    @staticmethod
    def _jaccard(left: set[str], right: set[str]) -> float:
        """Return Jaccard similarity between two token sets.

        Args:
            left: First token set.
            right: Second token set.

        Returns:
            Overlap in ``[0.0, 1.0]``; two empty sets are treated as identical.
        """
        if not left and not right:
            return 1.0
        union = left | right
        if not union:
            return 0.0
        return len(left & right) / len(union)
