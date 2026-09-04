"""Gate that drops results whose abstract/text is shorter than a minimum."""

from retrieval.models import SearchResult


class MinAbstractLengthGate:
    """Keep results whose chunk text meets a minimum character length.

    Uses ``chunk.text`` length (scholarly abstract or passage body) and
    drops stubs that are too short to support grounded generation.

    Inspired by LlamaIndex/Haystack length filters that drop stub abstracts before generation.
    Inputs are not mutated.  Local postprocessor for GPT-5.5 /
    Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 pipelines (not a DOI
    connector).
    """

    def __init__(self, min_chars: int = 80) -> None:
        """Create a min-abstract-length gate.

        Args:
            min_chars: Minimum ``len(chunk.text)`` required (non-negative).

        Raises:
            ValueError: If *min_chars* is not a non-negative integer.
        """
        if not isinstance(min_chars, int) or min_chars < 0:
            raise ValueError("min_chars must be a non-negative integer")
        self._min_chars = min_chars

    def gate(
        self,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Return results whose text length is at least *min_chars*."""
        if not results:
            return []

        kept: list[SearchResult] = []
        limit = len(results) if top_k is None else min(top_k, len(results))
        if limit <= 0:
            return []

        for r in results:
            if len(r.chunk.text) < self._min_chars:
                continue
            kept.append(
                SearchResult(
                    chunk=r.chunk,
                    score=r.score,
                    retriever="min_abstract_length_gate",
                    path=[*r.path, r.retriever],
                )
            )
            if len(kept) >= limit:
                break
        return kept
