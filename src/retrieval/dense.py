"""Dense cosine retrieval over local chunk embeddings."""

from retrieval.embeddings import HashEmbeddingModel, cosine_similarity
from retrieval.models import Chunk, SearchResult
from retrieval.scope import DocumentIdsInput, normalize_document_ids


class DenseRetriever:
    """Retrieve chunks by deterministic dense cosine similarity."""

    def __init__(self, embedder: HashEmbeddingModel | None = None) -> None:
        """Create an empty dense retriever."""
        self._embedder = embedder or HashEmbeddingModel()
        self._chunks: dict[str, Chunk] = {}
        self._vectors: dict[str, list[float]] = {}

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Upsert snapshots by chunk ID, keeping the last successfully embedded value."""
        for chunk in chunks:
            indexed_chunk = chunk.model_copy(deep=True)
            vector = self._embedder.embed(indexed_chunk.text).copy()
            self._chunks[indexed_chunk.chunk_id] = indexed_chunk
            self._vectors[indexed_chunk.chunk_id] = vector

    async def retrieve(
        self, query: str, limit: int = 10, *, document_ids: DocumentIdsInput | None = None
    ) -> list[SearchResult]:
        """Return detached snapshots of the most similar chunks for a query."""
        scope = normalize_document_ids(document_ids)
        allowed = None if scope is None else frozenset(scope)
        query_vector = self._embedder.embed(query)
        scored_chunks = [
            (chunk, cosine_similarity(query_vector, self._vectors[chunk.chunk_id]))
            for chunk in self._chunks.values()
            if allowed is None or chunk.document_id in allowed
        ]
        ranked_chunks = sorted(scored_chunks, key=lambda item: (-item[1], item[0].chunk_id))[:limit]
        return [
            SearchResult(chunk=chunk.model_copy(deep=True), score=score, retriever="dense")
            for chunk, score in ranked_chunks
        ]
