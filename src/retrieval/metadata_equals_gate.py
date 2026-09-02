"""Gate retrieval hits by exact metadata key/value equality filters."""

from retrieval.models import SearchResult


class MetadataEqualsGate:
    """Keep hits whose chunk metadata equals all required key/value pairs.

    Inspired by LlamaIndex ``MetadataFilters`` equality filters and Haystack
    ``MetadataRouter``. Empty ``required`` is a pass-through. Missing keys or
    mismatched values drop the hit. Distinct from
    :class:`~retrieval.required_metadata_gate.RequiredMetadataGate` (presence
    only). Inputs are not mutated. Local postprocessor for GPT-5.5 /
    Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 pipelines (not a DOI connector).
    """

    def __init__(self, required: dict[str, str] | None = None) -> None:
        """Create a metadata-equals gate.

        Args:
            required: Mapping of metadata key → required string value.
        """
        self._required = dict(required or {})

    def gate(
        self,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Return results matching all required metadata equalities."""
        if not results:
            return []
        limit = len(results) if top_k is None else min(top_k, len(results))
        if limit <= 0:
            return []
        if not self._required:
            return [
                SearchResult(
                    chunk=result.chunk,
                    score=result.score,
                    retriever="metadata_equals_gate",
                    path=[*result.path, result.retriever],
                )
                for result in results[:limit]
            ]

        kept: list[SearchResult] = []
        for result in results:
            meta = result.chunk.metadata or {}
            if not all(str(meta.get(key, "")) == value for key, value in self._required.items()):
                continue
            kept.append(
                SearchResult(
                    chunk=result.chunk,
                    score=result.score,
                    retriever="metadata_equals_gate",
                    path=[*result.path, result.retriever],
                )
            )
            if len(kept) >= limit:
                break
        return kept
