"""Tests for ingestion pipeline."""

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from ingestion.chunking import TextChunker
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


@pytest.mark.parametrize(("chunk_size", "overlap"), [(0, 0), (-1, 0), (4, -2), (4, 4), (4, 5)])
@pytest.mark.parametrize(
    "text",
    ["abcdefghij", "", " \t\n "],
    ids=["nonempty", "empty", "whitespace"],
)
async def test_invalid_chunker_cannot_persist_or_index_documents(
    tmp_path: Path, chunk_size: int, overlap: int, text: str
) -> None:
    database_path = tmp_path / "invalid-geometry.sqlite3"
    document_store = SQLiteDocumentStore(database_path)
    dense = DenseRetriever()
    sparse = BM25Retriever()
    hybrid = HybridRetriever(dense, sparse, HyDEExpander())
    graph_builder = GraphRAGBuilder(SQLiteGraphStore(database_path))
    document = Document(document_id="d1", title="Paper", text=text, source="fixture")

    with pytest.raises(ValueError, match=r"chunk_size|overlap"):
        chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)
        pipeline = IngestionPipeline(document_store, hybrid, graph_builder, chunker=chunker)
        pipeline.ingest_documents([document])

    with closing(sqlite3.connect(database_path)) as connection:
        assert connection.execute(
            "SELECT (SELECT COUNT(*) FROM documents), (SELECT COUNT(*) FROM chunks), "
            "(SELECT COUNT(*) FROM graph_chunks), (SELECT COUNT(*) FROM entity_mentions), "
            "(SELECT COUNT(*) FROM entity_edges)"
        ).fetchone() == (0, 0, 0, 0, 0)
    assert await dense.retrieve("GraphRAG") == []
    assert await sparse.retrieve("GraphRAG") == []
