"""Gate that caps results from any single source to promote diversity."""

from collections import Counter

from retrieval.models import SearchResult


class DiversityCapGate:
    """Limit how many results any single source document contributes.

    Iterates results in their existing order and keeps each hit only while
    its ``document_id`` has not yet reached *max_per_source*.  This
    promotes source diversity without re-scoring.

    Inspired by LlamaIndex/Haystack diversity-aware postprocessors.
    Inputs are not mutated.  Local postprocessor for GPT-5.5 /
    Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 pipelines (not a DOI
    connector).
    """

    def __init__(self, max_per_source: int = 3) -> None:
        """Create a diversity-cap gate.

        Args:
            max_per_source: Maximum hits per ``document_id`` (positive).

        Raises:
            ValueError: If *max_per_source* is not a positive integer.
        """
        if not isinstance(max_per_source, int) or max_per_source <= 0:
            raise ValueError("max_per_source must be a positive integer")
        self._max_per_source = max_per_source

    def gate(
        self,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Return results with per-source caps applied."""
        if not results:
            return []

        counts: Counter[str] = Counter()
        kept: list[SearchResult] = []
        limit = len(results) if top_k is None else min(top_k, len(results))
        if limit <= 0:
            return []

        for r in results:
            doc_id = r.chunk.document_id
            if counts[doc_id] >= self._max_per_source:
                continue
            counts[doc_id] += 1
            kept.append(
                SearchResult(
                    chunk=r.chunk,
                    score=r.score,
                    retriever="diversity_cap_gate",
                    path=[*r.path, r.retriever],
                )
            )
            if len(kept) >= limit:
                break
        return kept
