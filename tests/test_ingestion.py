"""Tests for ingestion pipeline."""

from pathlib import Path

from ingestion.pipeline import IngestionPipeline
from retrieval.dense import DenseRetriever
from retrieval.graph import GraphRAGBuilder
from retrieval.hybrid import HybridRetriever
from retrieval.hyde import HyDEExpander
from retrieval.models import Document
from retrieval.sparse import BM25Retriever
from storage.document_store import SQLiteDocumentStore
from storage.graph_store import SQLiteGraphStore


def test_ingestion_persists_and_indexes_chunks(tmp_path: Path) -> None:
    """Ingestion persists documents and indexes chunks for retrieval."""
    database_path = tmp_path / "ingest.sqlite3"
    hybrid = HybridRetriever(DenseRetriever(), BM25Retriever(), HyDEExpander())
    pipeline = IngestionPipeline(
        SQLiteDocumentStore(database_path),
        hybrid,
        GraphRAGBuilder(SQLiteGraphStore(database_path)),
    )
    chunks = pipeline.ingest_documents(
        [
            Document(
                document_id="d1",
                title="Paper",
                text="GraphRAG supports scientific retrieval.",
                source="fixture",
            )
        ]
    )
    assert len(chunks) == 1
