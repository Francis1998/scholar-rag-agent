"""Gate that fuses multiple ranked result lists via Reciprocal Rank Fusion."""

from retrieval.models import SearchResult


class ReciprocalRankFusionGate:
    """Merge rankings from multiple retrieval sources using RRF.

    Each result's fused score is ``sum(1 / (k + rank_i))`` across all lists
    it appears in.  Results are sorted descending by fused score and
    optionally capped by *top_k*.

    Inspired by Cormack, Clarke & Buettcher (2009) reciprocal rank fusion
    and LlamaIndex/Haystack RRF postprocessors.  Inputs are not mutated.
    Local postprocessor for GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
    Kimi K2 pipelines (not a DOI connector).
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
            top_k: Optional cap on returned results.

        Returns:
            Fused results sorted by descending RRF score.
        """
        if not result_sets:
            return []

        scores: dict[str, float] = {}
        best: dict[str, SearchResult] = {}
        sources: dict[str, list[str]] = {}

        for results in result_sets:
            for rank, result in enumerate(results, start=1):
                cid = result.chunk.chunk_id
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (self._k + rank)
                best.setdefault(cid, result)
                sources.setdefault(cid, []).append(result.retriever)

        fused = [
            SearchResult(
                chunk=best[cid].chunk,
                score=score,
                retriever="reciprocal_rank_fusion_gate",
                path=sources[cid],
            )
            for cid, score in scores.items()
        ]
        fused.sort(key=lambda r: r.score, reverse=True)

        if top_k is not None:
            fused = fused[: max(top_k, 0)]
        return fused
