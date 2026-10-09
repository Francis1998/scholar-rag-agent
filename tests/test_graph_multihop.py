"""Tests for GraphRAG extraction and multi-hop retrieval."""

from pathlib import Path

import pytest

from retrieval.graph import GraphRAGBuilder
from retrieval.models import Chunk, Entity, EntityEdge
from retrieval.multihop import MultiHopRetriever
from retrieval.scope import DocumentIdsInput
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


async def test_multihop_max_depth_can_exceed_five(tmp_path: Path) -> None:
    """A ``max_depth`` above five must traverse beyond five hops.

    ``__init__`` and ``retrieve`` previously applied a hardcoded literal ceiling
    of ``5`` alongside the configurable ``max_depth``, so a chain longer than
    five hops was silently truncated even when ``max_depth`` (wired from
    ``settings.max_hops``) permitted deeper traversal. The configured
    ``max_depth`` must be the sole traversal bound.
    """
    store = SQLiteGraphStore(tmp_path / "graph.sqlite3")
    chain_length = 7
    for index in range(1, chain_length + 1):
        chunk = Chunk(
            chunk_id=f"c{index}",
            document_id="d",
            title=f"N{index}",
            text=f"Node {index}",
            source="fixture",
        )
        store.add_mentions(chunk, [Entity(name=f"E{index}", label="TERM")])
    store.add_edges(
        [
            EntityEdge(source=f"E{index}", target=f"E{index + 1}", chunk_id=f"c{index}")
            for index in range(1, chain_length)
        ]
    )

    results = await MultiHopRetriever(store, max_depth=chain_length).retrieve(
        "q", ["E1"], depth=chain_length, limit=20
    )
    reached = {result.chunk.chunk_id for result in results}

    # c6 and c7 are only reachable at hop six and seven respectively.
    assert {"c6", "c7"}.issubset(reached)


@pytest.mark.parametrize("scope", [None, ("selected",)], ids=["unscoped", "scoped"])
async def test_multihop_frontier_budget_is_not_spent_on_case_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scope: tuple[str, ...] | None
) -> None:
    store = SQLiteGraphStore(tmp_path / "graph.sqlite3")
    for index, name in enumerate(["ALPHA", "Alpha", "Beta"]):
        chunk = Chunk(
            chunk_id=f"c{index}",
            document_id="selected",
            title=name,
            text=f"Seed {name}",
            source="fixture",
        )
        store.add_mentions(chunk, [Entity(name="Seed"), Entity(name=name)])
        store.add_edges([EntityEdge(source="Seed", target=name, chunk_id=chunk.chunk_id)])
    tail = Chunk(
        chunk_id="tail",
        document_id="selected",
        title="Beta",
        text="Beta evidence reachable only at the second hop.",
        source="fixture",
    )
    store.add_mentions(tail, [Entity(name="Beta")])
    frontiers: list[list[str]] = []
    requested_limits: list[int] = []
    neighbours = store.neighbours

    def bounded_neighbours(
        entities: list[str],
        limit: int = 20,
        *,
        document_ids: DocumentIdsInput | None = None,
    ) -> list[str]:
        """Cap real SQL fan-out while leaving room for all fixture chunks."""
        frontiers.append(list(entities))
        requested_limits.append(limit)
        return neighbours(entities, limit=min(limit, 2), document_ids=document_ids)

    monkeypatch.setattr(store, "neighbours", bounded_neighbours)

    results = await MultiHopRetriever(store).retrieve(
        "Seed", ["sEeD"], depth=2, limit=10, document_ids=scope
    )

    assert frontiers == [["seed"], ["alpha", "beta"]]
    assert requested_limits == [40, 40]
    assert {result.chunk.chunk_id: result.score for result in results} == {
        "c0": 1.0,
        "c1": 1.0,
        "c2": 1.0,
        "tail": 0.5,
    }
    assert results[-1].chunk == tail
    assert results[-1].path == ["alpha", "beta"]
