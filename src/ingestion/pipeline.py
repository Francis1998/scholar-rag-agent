"""End-to-end ingestion pipeline for normalized scientific documents."""

from ingestion.chunking import TextChunker
from retrieval.graph import GraphRAGBuilder
from retrieval.hybrid import HybridRetriever
from retrieval.models import Chunk, Document
from storage.document_store import SQLiteDocumentStore
from storage.graph_store import SQLiteGraphStore


def _require_replacement_hook(
    component: object, base: type[object], legacy: str, replacement: str
) -> None:
    if getattr(getattr(component, legacy), "__func__", None) is not getattr(
        base, legacy
    ) and getattr(getattr(component, replacement), "__func__", None) is getattr(base, replacement):
        raise TypeError(
            f"Custom {base.__name__}.{legacy} requires an explicit {replacement} implementation."
        )


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
        """Replace complete documents; repeated document IDs use their last supplied value."""
        batch = {document.document_id: document.model_copy(deep=True) for document in documents}
        if not batch:
            return []
        graph_store = self._graph_builder.graph_store
        _require_replacement_hook(
            self._document_store, SQLiteDocumentStore, "add_documents", "replacing_documents"
        )
        _require_replacement_hook(
            self._graph_builder, GraphRAGBuilder, "index_chunks", "prepare_chunks"
        )
        _require_replacement_hook(
            graph_store, SQLiteGraphStore, "replace_chunk", "replace_documents"
        )
        current_documents = list(batch.values())
        chunks: list[Chunk] = []
        for document in current_documents:
            generated = self._chunker.chunk(document.model_copy(deep=True))
            if any(chunk.document_id != document.document_id for chunk in generated):
                raise ValueError("Each chunk must belong to the document being chunked.")
            chunks.extend(chunk.model_copy(deep=True) for chunk in generated)
        if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
            raise ValueError("Replacement chunks must have unique chunk_id values.")
        document_ids = frozenset(batch)
        graph_chunks = self._graph_builder.prepare_chunks(chunks)
        shared_database = graph_store.database_path == self._document_store.database_path
        with (
            self._hybrid_retriever.replacing_documents(document_ids, chunks),
            self._document_store.replacing_documents(current_documents, chunks) as connection,
        ):
            graph_store.replace_documents(
                document_ids, graph_chunks, connection=connection if shared_database else None
            )
        return chunks
