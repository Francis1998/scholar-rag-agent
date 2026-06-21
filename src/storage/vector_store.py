"""Simple in-memory vector store abstraction for local retrieval."""

from retrieval.embeddings import cosine_similarity
from retrieval.models import Chunk, SearchResult


class InMemoryVectorStore:
    """Store vectors and chunks for cosine search."""

    def __init__(self) -> None:
        """Create an empty vector store."""
        self._vectors: dict[str, list[float]] = {}
        self._chunks: dict[str, Chunk] = {}

    def upsert(self, chunk: Chunk, vector: list[float]) -> None:
        """Insert or replace a chunk vector."""
        self._chunks[chunk.chunk_id] = chunk
        self._vectors[chunk.chunk_id] = vector

    def search(self, vector: list[float], limit: int = 10) -> list[SearchResult]:
        """Return nearest chunks by cosine similarity."""
        results = [
            SearchResult(
                chunk=self._chunks[chunk_id],
                score=cosine_similarity(vector, chunk_vector),
                retriever="vector_store",
            )
            for chunk_id, chunk_vector in self._vectors.items()
        ]
        return sorted(results, key=lambda result: result.score, reverse=True)[:limit]
