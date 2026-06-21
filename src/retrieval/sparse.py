"""BM25 sparse retrieval over local chunks."""

import math
from collections import Counter

from retrieval.models import Chunk, SearchResult


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric terms."""
    return [token.strip(".,;:()[]{}!?\"'").lower() for token in text.split() if token.strip()]


class BM25Retriever:
    """Minimal BM25 retriever for scientific text chunks."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        """Create an empty BM25 index."""
        self._k1 = k1
        self._b = b
        self._chunks: list[Chunk] = []
        self._term_frequencies: dict[str, Counter[str]] = {}
        self._document_frequencies: Counter[str] = Counter()
        self._document_lengths: dict[str, int] = {}
        self._average_length = 0.0

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Index chunks for BM25 retrieval."""
        for chunk in chunks:
            terms = tokenize(chunk.text)
            frequencies: Counter[str] = Counter(terms)
            self._chunks.append(chunk)
            self._term_frequencies[chunk.chunk_id] = frequencies
            self._document_lengths[chunk.chunk_id] = len(terms)
            self._document_frequencies.update(frequencies.keys())
        total_length = sum(self._document_lengths.values())
        self._average_length = total_length / max(len(self._document_lengths), 1)

    async def retrieve(self, query: str, limit: int = 10) -> list[SearchResult]:
        """Retrieve chunks ranked by BM25 score."""
        query_terms = tokenize(query)
        scored_results = [
            SearchResult(
                chunk=chunk, score=self._score(chunk.chunk_id, query_terms), retriever="bm25"
            )
            for chunk in self._chunks
        ]
        return sorted(scored_results, key=lambda result: result.score, reverse=True)[:limit]

    def _score(self, chunk_id: str, query_terms: list[str]) -> float:
        """Compute BM25 score for a chunk."""
        score = 0.0
        frequencies = self._term_frequencies[chunk_id]
        document_length = self._document_lengths[chunk_id]
        corpus_size = max(len(self._chunks), 1)
        for term in query_terms:
            term_frequency = frequencies.get(term, 0)
            if term_frequency == 0:
                continue
            document_frequency = self._document_frequencies.get(term, 0)
            idf = math.log(
                1 + (corpus_size - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            denominator = term_frequency + self._k1 * (
                1 - self._b + self._b * document_length / max(self._average_length, 1.0)
            )
            score += idf * (term_frequency * (self._k1 + 1)) / denominator
        return score
