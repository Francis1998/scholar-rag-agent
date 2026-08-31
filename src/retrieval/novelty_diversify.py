"""Diversify retrieval hits by demoting near-duplicate chunk text."""

import math

from retrieval.models import SearchResult
from retrieval.sparse import tokenize


class NoveltyDiversifier:
    """Greedy novelty re-ranking that demotes near-duplicate chunks.

    Inspired by LlamaIndex diversity postprocessors and Maximal Marginal
    Relevance (Carbonell & Goldstein, 1998). Unlike
    :class:`~retrieval.mmr.MMRDiversifier` (reorder-only with a lambda trade-off)
    and :class:`~retrieval.near_duplicate_collapse.NearDuplicateCollapser`
    (hard drop), this stage soft-demotes redundancy via an alpha blend:

    ```text
    novelty = 1 - max Jaccard(tokens(d), tokens(s))  over selected s
    new_score = (1 - alpha) * old + alpha * novelty
    ```

    Selection is greedy: at each step the remaining candidate with the highest
    blended score is chosen and appended. When nothing is selected yet,
    ``novelty = 1.0``. Empty token sets only match other empty sets at
    similarity ``1.0``. Results receive updated scores and
    ``retriever="novelty_diversify"``. Inputs are not mutated. Local retrieval
    postprocessor for GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2
    pipelines (not a DOI connector).
    """

    def __init__(self, alpha: float = 0.5) -> None:
        """Create a novelty diversifier.

        Args:
            alpha: Weight for the novelty signal in ``[0.0, 1.0]``.
                ``0.0`` preserves relevance order (still rewrites provenance);
                ``1.0`` ranks purely by novelty vs already-selected rows.

        Raises:
            ValueError: If ``alpha`` is non-finite or outside ``[0.0, 1.0]``.
        """
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be a finite number within [0.0, 1.0]")
        self._alpha = alpha

    def diversify(
        self,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Return results greedily re-ranked with a novelty penalty."""
        limit = len(results) if top_k is None else min(top_k, len(results))
        if not results or limit <= 0:
            return []

        token_sets = [set(tokenize(result.chunk.text)) for result in results]
        selected_indices: list[int] = []
        remaining = list(range(len(results)))
        output: list[SearchResult] = []

        while remaining and len(output) < limit:
            best_index = remaining[0]
            best_score = float("-inf")
            for index in remaining:
                novelty = self._novelty(token_sets[index], selected_indices, token_sets)
                score = (1.0 - self._alpha) * results[index].score + self._alpha * novelty
                if score > best_score:
                    best_score = score
                    best_index = index
            prior = results[best_index]
            output.append(
                SearchResult(
                    chunk=prior.chunk,
                    score=best_score,
                    retriever="novelty_diversify",
                    path=[*prior.path, prior.retriever],
                )
            )
            selected_indices.append(best_index)
            remaining.remove(best_index)
        return output

    @staticmethod
    def _novelty(
        candidate_tokens: set[str],
        selected_indices: list[int],
        token_sets: list[set[str]],
    ) -> float:
        if not selected_indices:
            return 1.0
        max_similarity = max(
            NoveltyDiversifier._jaccard(candidate_tokens, token_sets[selected])
            for selected in selected_indices
        )
        return 1.0 - max_similarity

    @staticmethod
    def _jaccard(left: set[str], right: set[str]) -> float:
        if not left and not right:
            return 1.0
        union = left | right
        if not union:
            return 0.0
        return len(left & right) / len(union)
