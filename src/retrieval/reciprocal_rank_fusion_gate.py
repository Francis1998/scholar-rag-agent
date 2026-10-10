"""Gate that fuses multiple ranked result lists via Reciprocal Rank Fusion."""

from retrieval.models import SearchResult
from retrieval.rrf import reciprocal_rank_fusion


class ReciprocalRankFusionGate:
    """Merge rankings from multiple retrieval sources using RRF.

    Each result's fused score is ``sum(1 / (k + rank_i))`` across all lists
    it appears in, using its first original 1-based rank in each list.
    Results are sorted descending by fused score and optionally capped
    by *top_k*.

    Inspired by Cormack, Clarke & Buettcher (2009) reciprocal rank fusion
    and LlamaIndex/Haystack RRF postprocessors.  Inputs are not mutated.
    Provider-independent local postprocessor (not a DOI connector); see
    ``docs/guides/PROVIDER_MODELS_GUIDE.md`` for supported model adapters.
    """

    def __init__(self, k: int = 60) -> None:
        """Create an RRF gate.

        Args:
            k: Rank smoothing constant (positive integer).

        Raises:
            ValueError: If *k* is not a positive integer.
        """
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        self._k = k

    def gate(
        self,
        result_sets: list[list[SearchResult]],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Fuse *result_sets* into a single ranked list.

        Args:
            result_sets: One list of ``SearchResult`` per retrieval source.
            top_k: Optional cap; ``None`` returns all, non-positive values return none.

        Returns:
            Fused results sorted by descending RRF score.
        """
        if not result_sets:
            return []

        fused = reciprocal_rank_fusion(
            result_sets,
            limit=None if top_k is None else max(top_k, 0),
            rank_constant=self._k,
        )
        return [
            result.model_copy(update={"retriever": "reciprocal_rank_fusion_gate"})
            for result in fused
        ]
