"""BM25 sparse retrieval over local chunks."""

import math
from collections import Counter

from retrieval.models import Chunk, SearchResult
from retrieval.scope import DocumentIdsInput, normalize_document_ids

STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "with",
    }
)


def tokenize(text: str) -> list[str]:
    """Split on whitespace, lowercase, and strip selected surrounding punctuation.

    Drop tokens emptied by stripping, but retain other nonempty tokens for
    BM25 and hash embeddings. ``meaningful_terms`` applies the stricter
    lexical-overlap filter without changing retrieval tokenization.
    """
    tokens = [token.strip(".,;:()[]{}!?\"'").lower() for token in text.split()]
    return [token for token in tokens if token]


def meaningful_terms(text: str) -> set[str]:
    """Return non-stopword tokens containing at least one Unicode letter or number."""
    return {
        term
        for term in tokenize(text)
        if term not in STOPWORDS and any(char.isalnum() for char in term)
    }


class BM25Retriever:
    """Minimal BM25 retriever for scientific text chunks."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        """Create an empty BM25 index."""
        self._k1 = k1
        self._b = b
        self._chunks: dict[str, Chunk] = {}
        self._term_frequencies: dict[str, Counter[str]] = {}
        self._document_frequencies: Counter[str] = Counter()
        self._document_lengths: dict[str, int] = {}
        self._average_length = 0.0

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Snapshot the batch, then upsert each chunk's contribution to corpus statistics."""
        indexed_chunks = [chunk.model_copy(deep=True) for chunk in chunks]
        for indexed_chunk in indexed_chunks:
            terms = tokenize(indexed_chunk.text)
            frequencies: Counter[str] = Counter(terms)
            previous = self._term_frequencies.get(indexed_chunk.chunk_id)
            if previous is not None:
                for term in previous:
                    self._document_frequencies[term] -= 1
                    if self._document_frequencies[term] == 0:
                        del self._document_frequencies[term]
            self._chunks[indexed_chunk.chunk_id] = indexed_chunk
            self._term_frequencies[indexed_chunk.chunk_id] = frequencies
            self._document_lengths[indexed_chunk.chunk_id] = len(terms)
            self._document_frequencies.update(frequencies.keys())
        total_length = sum(self._document_lengths.values())
        self._average_length = total_length / max(len(self._document_lengths), 1)

    async def retrieve(
        self, query: str, limit: int = 10, *, document_ids: DocumentIdsInput | None = None
    ) -> list[SearchResult]:
        """Return detached hits, ranking scoped candidates with global BM25 statistics."""
        scope = normalize_document_ids(document_ids)
        allowed = None if scope is None else frozenset(scope)
        query_terms = tokenize(query)
        scored_chunks = [
            (chunk, self._score(chunk.chunk_id, query_terms))
            for chunk in self._chunks.values()
            if allowed is None or chunk.document_id in allowed
        ]
        ranked_chunks = sorted(scored_chunks, key=lambda item: (-item[1], item[0].chunk_id))[:limit]
        return [
            SearchResult(chunk=chunk.model_copy(deep=True), score=score, retriever="bm25")
            for chunk, score in ranked_chunks
        ]

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
