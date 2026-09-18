"""Dense cosine retrieval over local chunk embeddings."""

from retrieval.embeddings import HashEmbeddingModel, cosine_similarity
from retrieval.models import Chunk, SearchResult
from retrieval.scope import DocumentIdsInput, normalize_document_ids


class DenseRetriever:
    """Retrieve chunks by deterministic dense cosine similarity."""

    def __init__(self, embedder: HashEmbeddingModel | None = None) -> None:
        """Create an empty dense retriever."""
        self._embedder = embedder or HashEmbeddingModel()
        self._chunks: list[Chunk] = []
        self._vectors: dict[str, list[float]] = {}

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Index chunks for dense retrieval."""
        for chunk in chunks:
            self._chunks.append(chunk)
            self._vectors[chunk.chunk_id] = self._embedder.embed(chunk.text)

    async def retrieve(
        self, query: str, limit: int = 10, *, document_ids: DocumentIdsInput | None = None
    ) -> list[SearchResult]:
        """Retrieve the most similar chunks for a query."""
        scope = normalize_document_ids(document_ids)
        allowed = None if scope is None else frozenset(scope)
        query_vector = self._embedder.embed(query)
        results = [
            SearchResult(
                chunk=chunk,
                score=cosine_similarity(query_vector, self._vectors[chunk.chunk_id]),
                retriever="dense",
            )
            for chunk in self._chunks
            if allowed is None or chunk.document_id in allowed
        ]
        return sorted(results, key=lambda result: result.score, reverse=True)[:limit]
