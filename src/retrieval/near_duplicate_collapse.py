"""Collapse near-duplicate retrieval hits by lexical Jaccard similarity."""

import math

from retrieval.models import SearchResult
from retrieval.sparse import meaningful_terms


class NearDuplicateCollapser:
    """Keep one highest-scoring representative per near-duplicate cluster.

    Inspired by LlamaIndex ``SimilarityPostprocessor`` / dedupe stages and
    Haystack near-duplicate filters. Chunks whose ``meaningful_terms`` Jaccard
    similarity on **text** (not title) meets or exceeds ``threshold`` are
    treated as near-duplicates. Within each cluster only the highest-scoring
    result is kept; survivors retain their relative score order. Unlike
    :class:`~retrieval.mmr.MMRDiversifier`, which re-ranks for novelty without
    dropping rows, this stage hard-collapses redundant hits. Input objects are
    never mutated.
    """

    def __init__(self, threshold: float = 0.9) -> None:
        """Create a near-duplicate collapser.

        Args:
            threshold: Inclusive Jaccard similarity in ``[0.0, 1.0]`` at which
                two chunk texts are considered near-duplicates. Defaults to
                ``0.9``.

        Raises:
            ValueError: If ``threshold`` is non-finite or outside ``[0.0, 1.0]``.
        """
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be a finite number within [0.0, 1.0]")
        self._threshold = threshold

    def collapse(
        self,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Return deduplicated results ordered by descending score.

        Candidates are considered in stable score-descending order. A candidate
        is kept when its text-term Jaccard similarity to every already-kept
        representative is strictly below ``threshold``; otherwise it is dropped
        as a near-duplicate of a higher-scoring hit. Empty term sets only match
        other empty term sets at similarity ``1.0`` (or ``0.0`` when comparing
        empty vs non-empty). ``top_k`` truncates after collapse; ``None`` keeps
        every survivor.
        """
        if not results:
            return []
        limit = len(results) if top_k is None else min(top_k, len(results))
        if limit <= 0:
            return []

        # Stable sort by score so the highest-scoring member of each cluster
        # is considered first and becomes the representative.
        ordered = sorted(results, key=lambda result: result.score, reverse=True)
        term_sets = [meaningful_terms(result.chunk.text) for result in ordered]

        kept: list[SearchResult] = []
        kept_terms: list[set[str]] = []
        for result, terms in zip(ordered, term_sets, strict=True):
            if any(self._jaccard(terms, prior) >= self._threshold for prior in kept_terms):
                continue
            kept.append(result)
            kept_terms.append(terms)
            if len(kept) >= limit:
                break
        return kept

    @staticmethod
    def _jaccard(left: set[str], right: set[str]) -> float:
        if not left and not right:
            return 1.0
        union = left | right
        if not union:
            return 1.0
        return len(left & right) / len(union)
