"""Dense cosine retrieval over local chunk embeddings."""

from collections.abc import Iterator
from contextlib import contextmanager
from threading import RLock

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
        self._lock = RLock()

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Snapshot the batch, then upsert the last successfully embedded value per ID."""
        indexed_chunks = [chunk.model_copy(deep=True) for chunk in chunks]
        with self._lock:
            for indexed_chunk in indexed_chunks:
                vector = self._embedder.embed(indexed_chunk.text).copy()
                self._chunks[indexed_chunk.chunk_id] = indexed_chunk
                self._vectors[indexed_chunk.chunk_id] = vector

    @contextmanager
    def replacing_documents(
        self, document_ids: frozenset[str], chunks: list[Chunk]
    ) -> Iterator[None]:
        """Stage an owned replacement and publish only after the caller's writes succeed."""
        if (
            getattr(self.add_chunks, "__func__", None) is not DenseRetriever.add_chunks
            and getattr(self.replacing_documents, "__func__", None)
            is DenseRetriever.replacing_documents
        ):
            raise TypeError(
                "Custom dense add_chunks requires a replacing_documents implementation."
            )
        with self._lock:
            replacement = DenseRetriever(self._embedder)
            replacement._chunks = {
                key: chunk
                for key, chunk in self._chunks.items()
                if chunk.document_id not in document_ids
            }
            replacement._vectors = {key: self._vectors[key] for key in replacement._chunks}
            replacement.add_chunks(chunks)
            yield
            self._chunks = replacement._chunks
            self._vectors = replacement._vectors

    async def retrieve(
        self, query: str, limit: int = 10, *, document_ids: DocumentIdsInput | None = None
    ) -> list[SearchResult]:
        """Return detached snapshots of the most similar chunks for a query."""
        scope = normalize_document_ids(document_ids)
        allowed = None if scope is None else frozenset(scope)
        with self._lock:
            query_vector = self._embedder.embed(query)
            scored_chunks = [
                (chunk, cosine_similarity(query_vector, self._vectors[chunk.chunk_id]))
                for chunk in self._chunks.values()
                if allowed is None or chunk.document_id in allowed
            ]
            ranked_chunks = sorted(scored_chunks, key=lambda item: (-item[1], item[0].chunk_id))[
                :limit
            ]
            return [
                SearchResult(chunk=chunk.model_copy(deep=True), score=score, retriever="dense")
                for chunk, score in ranked_chunks
            ]
