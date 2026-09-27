"""Gate that requires a minimum number of unique source documents."""

from collections.abc import Iterable

from retrieval.models import SearchResult


def count_evidence_documents(results: Iterable[SearchResult]) -> int:
    """Count actual document IDs without discarding evidence or rewriting provenance."""
    return len({result.chunk.document_id for result in results})


class MinUniqueSourcesGate:
    """Reject result sets lacking source diversity.

    Counts distinct ``document_id`` values across hits.  If the count is
    below *min_sources* the entire batch is rejected (returns empty).
    Otherwise all results pass through with rewritten provenance.

    Inputs are not mutated. This offline, library-only postprocessor is not
    invoked by the API; gate() alone does not prevent answer generation.
    The integrated min_evidence_documents requirement shares the counting
    helper, not these discard/provenance-rewrite semantics.
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

        if count_evidence_documents(results) < self._min_sources:
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
