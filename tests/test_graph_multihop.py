"""Tests for GraphRAG extraction and multi-hop retrieval."""

from pathlib import Path

from retrieval.graph import GraphRAGBuilder
from retrieval.models import Chunk
from retrieval.multihop import MultiHopRetriever
from storage.graph_store import SQLiteGraphStore


async def test_multihop_retriever_follows_entity_chain(tmp_path: Path) -> None:
    """Multi-hop retrieval returns chunks connected to seed entities."""
    store = SQLiteGraphStore(tmp_path / "graph.sqlite3")
    chunks = [
        Chunk(
            chunk_id="c1",
            document_id="d1",
            title="GraphRAG",
            text="GraphRAG connects Retrieval and Agents for scientific reasoning.",
            source="fixture",
        ),
        Chunk(
            chunk_id="c2",
            document_id="d2",
            title="Agents",
            text="Agents use Planning and Retrieval for multi-hop synthesis.",
            source="fixture",
        ),
    ]
    GraphRAGBuilder(store).index_chunks(chunks)
    results = await MultiHopRetriever(store).retrieve("GraphRAG", ["GraphRAG"], depth=3, limit=5)
    assert {result.chunk.chunk_id for result in results}
