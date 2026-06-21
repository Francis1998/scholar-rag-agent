"""End-to-end ingestion pipeline for normalized scientific documents."""

from ingestion.chunking import TextChunker
from retrieval.graph import GraphRAGBuilder
from retrieval.hybrid import HybridRetriever
from retrieval.models import Chunk, Document
from storage.document_store import SQLiteDocumentStore


class IngestionPipeline:
    """Normalize, chunk, persist, and index scientific documents."""

    def __init__(
        self,
        document_store: SQLiteDocumentStore,
        hybrid_retriever: HybridRetriever,
        graph_builder: GraphRAGBuilder,
        chunker: TextChunker | None = None,
    ) -> None:
        """Create an ingestion pipeline."""
        self._document_store = document_store
        self._hybrid_retriever = hybrid_retriever
        self._graph_builder = graph_builder
        self._chunker = chunker or TextChunker()

    def ingest_documents(self, documents: list[Document]) -> list[Chunk]:
        """Persist documents and index their chunks."""
        chunks: list[Chunk] = []
        for document in documents:
            chunks.extend(self._chunker.chunk(document))
        self._document_store.add_documents(documents, chunks)
        self._hybrid_retriever.add_chunks(chunks)
        self._graph_builder.index_chunks(chunks)
        return chunks
