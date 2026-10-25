"""Simple in-memory vector store abstraction for local retrieval."""

from retrieval.counts import validate_result_count
from retrieval.embeddings import cosine_similarity
from retrieval.models import Chunk, SearchResult


class InMemoryVectorStore:
    """Store owned chunk/vector snapshots for in-memory cosine search.

    Vectors must already be normalized. Inputs and returned chunks do not share
    mutable state with indexed data; call upsert again to replace an entry.
    """

    def __init__(self) -> None:
        """Create an empty vector store."""
        self._vectors: dict[str, list[float]] = {}
        self._chunks: dict[str, Chunk] = {}

    def upsert(self, chunk: Chunk, vector: list[float]) -> None:
        """Snapshot both inputs before replacing the entry for the chunk ID."""
        owned_chunk = chunk.model_copy(deep=True)
        owned_vector = list(vector)
        self._chunks[owned_chunk.chunk_id] = owned_chunk
        self._vectors[owned_chunk.chunk_id] = owned_vector

    def search(self, vector: list[float], limit: int = 10) -> list[SearchResult]:
        """Return detached chunks by cosine similarity, with insertion-order ties."""
        validate_result_count(limit)
        if limit == 0:
            return []
        results = [
            SearchResult(
                chunk=self._chunks[chunk_id].model_copy(deep=True),
                score=cosine_similarity(vector, chunk_vector),
                retriever="vector_store",
            )
            for chunk_id, chunk_vector in self._vectors.items()
        ]
        return sorted(results, key=lambda result: result.score, reverse=True)[:limit]
