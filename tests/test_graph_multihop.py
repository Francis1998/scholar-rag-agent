"""Tests for GraphRAG extraction and multi-hop retrieval."""

from pathlib import Path

from retrieval.graph import GraphRAGBuilder
from retrieval.models import Chunk, Entity, EntityEdge
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


async def test_multihop_max_depth_bounds_traversal(tmp_path: Path) -> None:
    """``max_depth`` must bound traversal so the configured hop cap is enforced.

    This is the cap wired from ``settings.max_hops`` in the API container.
    A second-hop chunk should only be reachable when ``max_depth`` allows it.
    """
    store = SQLiteGraphStore(tmp_path / "graph.sqlite3")
    chunk_a = Chunk(chunk_id="a", document_id="d", title="A", text="Alpha", source="fixture")
    chunk_b = Chunk(chunk_id="b", document_id="d", title="B", text="Beta", source="fixture")
    store.add_mentions(chunk_a, [Entity(name="Alpha", label="TERM")])
    store.add_mentions(chunk_b, [Entity(name="Beta", label="TERM")])
    store.add_edges([EntityEdge(source="Alpha", target="Beta", chunk_id="a")])

    shallow = await MultiHopRetriever(store, max_depth=1).retrieve(
        "q", ["Alpha"], depth=5, limit=10
    )
    deep = await MultiHopRetriever(store, max_depth=3).retrieve("q", ["Alpha"], depth=5, limit=10)

    assert {result.chunk.chunk_id for result in shallow} == {"a"}
    assert {result.chunk.chunk_id for result in deep} == {"a", "b"}
