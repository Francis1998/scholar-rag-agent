"""Deterministic lexical compression of retrieved chunk context."""

import re

from retrieval.models import SearchResult
from retrieval.sparse import meaningful_terms

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")


class ContextualCompressor:
    """Extract bounded query-relevant sentence spans from retrieved chunks."""

    def __init__(
        self,
        max_sentences_per_chunk: int = 3,
        max_chars_per_chunk: int = 1_200,
        min_overlap: int = 1,
    ) -> None:
        """Create a lexical contextual compressor.

        Args:
            max_sentences_per_chunk: Maximum selected spans per returned chunk.
            max_chars_per_chunk: Hard character limit for each compressed chunk.
            min_overlap: Minimum distinct query-term overlap for a sentence.

        Raises:
            ValueError: If any bound is not a positive integer.
        """
        bounds = {
            "max_sentences_per_chunk": max_sentences_per_chunk,
            "max_chars_per_chunk": max_chars_per_chunk,
            "min_overlap": min_overlap,
        }
        for name, value in bounds.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self._max_sentences = max_sentences_per_chunk
        self._max_chars = max_chars_per_chunk
        self._min_overlap = min_overlap

    def compress(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Return results whose text contains only relevant sentence spans.

        Chunks without a sentence meeting ``min_overlap`` are filtered out.
        Selected sentences are restored to source order, while ties during
        selection preserve their original order. Input objects are not mutated.
        """
        limit = len(results) if top_k is None else min(top_k, len(results))
        query_terms = meaningful_terms(query)
        if not query_terms or limit <= 0:
            return []

        compressed_results: list[SearchResult] = []
        for result in results:
            compressed_text = self._compress_text(query_terms, result.chunk.text)
            if not compressed_text:
                continue
            compressed_results.append(
                SearchResult(
                    chunk=result.chunk.model_copy(update={"text": compressed_text}),
                    score=result.score,
                    retriever="contextual_compression",
                    path=[*result.path, result.retriever],
                )
            )
            if len(compressed_results) == limit:
                break
        return compressed_results

    def _compress_text(self, query_terms: set[str], text: str) -> str:
        sentences = [
            sentence.strip()
            for sentence in _SENTENCE_BOUNDARY.split(text.strip())
            if sentence.strip()
        ]
        ranked: list[tuple[int, float, int, str]] = []
        for index, sentence in enumerate(sentences):
            sentence_terms = meaningful_terms(sentence)
            overlap = len(query_terms & sentence_terms)
            if overlap < self._min_overlap:
                continue
            density = overlap / max(len(sentence_terms), 1)
            ranked.append((overlap, density, index, sentence))

        selected = sorted(ranked, key=lambda item: (-item[0], -item[1], item[2]))[
            : self._max_sentences
        ]
        selected.sort(key=lambda item: item[2])
        compressed = " ".join(item[3] for item in selected)
        return compressed[: self._max_chars].rstrip()
