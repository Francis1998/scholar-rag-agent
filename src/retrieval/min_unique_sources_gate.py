"""Gate that requires a minimum number of unique source documents."""

from retrieval.models import SearchResult


class MinUniqueSourcesGate:
    """Reject result sets lacking source diversity.

    Counts distinct ``document_id`` values across hits.  If the count is
    below *min_sources* the entire batch is rejected (returns empty).
    Otherwise all results pass through with rewritten provenance.

    Inspired by LlamaIndex/Haystack diversity-aware postprocessors.
    Inputs are not mutated.  Local postprocessor for GPT-5.5 /
    Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 pipelines (not a DOI
    connector).
    """

    def __init__(self, min_sources: int = 2) -> None:
        """Create a minimum-unique-sources gate.

        Args:
            min_sources: Required distinct ``document_id`` count (positive).

        Raises:
            ValueError: If *min_sources* is not a positive integer.
        """
        if not isinstance(min_sources, int) or min_sources <= 0:
            raise ValueError("min_sources must be a positive integer")
        self._min_sources = min_sources

    def gate(
        self,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Return *results* only when enough unique sources are present."""
        if not results:
            return []

        unique = {r.chunk.document_id for r in results}
        if len(unique) < self._min_sources:
            return []

        limit = len(results) if top_k is None else min(top_k, len(results))
        if limit <= 0:
            return []

        return [
            SearchResult(
                chunk=r.chunk,
                score=r.score,
                retriever="min_unique_sources_gate",
                path=[*r.path, r.retriever],
            )
            for r in results[:limit]
        ]
