"""Graph counts are validated before SQL, slicing, or custom-store calls."""

import math
import sqlite3
from pathlib import Path
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from retrieval.graph import GraphRetriever
from retrieval.models import Chunk, Entity, EntityEdge, SearchResult
from retrieval.multihop import MultiHopRetriever
from storage.graph_store import SQLiteGraphStore

INVALID_LIMITS = [
    pytest.param(-1, id="negative"),
    pytest.param(-(10**400), id="huge-negative"),
    pytest.param(True, id="true"),
    pytest.param(False, id="false"),
    pytest.param("1", id="numeric-string"),
    pytest.param("-1", id="negative-string"),
    pytest.param(0.0, id="zero-float"),
    pytest.param(1.0, id="integral-float"),
    pytest.param(1.5, id="fractional-float"),
    pytest.param(math.nan, id="nan"),
    pytest.param(math.inf, id="infinity"),
    pytest.param(-math.inf, id="negative-infinity"),
    pytest.param(None, id="none"),
]
VALID_LIMITS = [
    0,
    1,
    2,
    10,
    20,
    pytest.param(2**61, id="fanout-overflow"),
    pytest.param(2**63 - 1, id="sqlite-max"),
    pytest.param(2**63, id="above-sqlite-max"),
    pytest.param(10**400, id="huge"),
]
SCOPES = [None, ("selected",), ("missing",)]


@pytest.fixture
def graph_chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"c{index:02d}",
            document_id="selected" if index % 2 else "excluded",
            title=f"Paper {index}",
            text=f"Seed Node{index:02d}",
            source="fixture",
        )
        for index in range(24)
    ]


@pytest.fixture
def graph_store(tmp_path: Path, graph_chunks: list[Chunk]) -> SQLiteGraphStore:
    store = SQLiteGraphStore(tmp_path / "graph.sqlite3")
    for chunk in reversed(graph_chunks):
        neighbour = f"Node{chunk.chunk_id[1:]}"
        store.replace_chunk(
            chunk,
            [Entity(name="Seed"), Entity(name=neighbour)],
            [EntityEdge(source="Seed", target=neighbour, chunk_id=chunk.chunk_id)],
        )
    return store


def eligible_chunks(chunks: list[Chunk], scope: tuple[str, ...] | None) -> list[Chunk]:
    return [chunk for chunk in chunks if scope is None or chunk.document_id in scope]


@pytest.mark.parametrize("lookup", ["chunks_for_entities", "neighbours"])
@pytest.mark.parametrize("seeds", [[], ["Seed"]], ids=["empty-seeds", "seeds"])
@pytest.mark.parametrize("scope", [None, ("missing",)], ids=["unscoped", "empty-matches"])
@pytest.mark.parametrize("limit", INVALID_LIMITS)
def test_graph_store_rejects_invalid_limits_before_sql(
    graph_store: SQLiteGraphStore,
    monkeypatch: pytest.MonkeyPatch,
    lookup: str,
    seeds: list[str],
    scope: tuple[str, ...] | None,
    limit: object,
) -> None:
    connect = Mock(wraps=sqlite3.connect)
    monkeypatch.setattr("storage.graph_store.sqlite3.connect", connect)
    error = ValueError if isinstance(limit, int) and not isinstance(limit, bool) else TypeError
    message = (
        r"^limit must be >= 0$"
        if error is ValueError
        else r"^limit must be an integer \(excluding booleans\)$"
    )

    with pytest.raises(error, match=message):
        rows = getattr(graph_store, lookup)(seeds, limit=limit, document_ids=scope)
        pytest.fail(f"{lookup} accepted {limit!r} and returned {len(rows)} rows")

    connect.assert_not_called()


@pytest.mark.parametrize("seeds", [[], ["Seed"]], ids=["empty-seeds", "seeds"])
@pytest.mark.parametrize("scope", [None, ("missing",)], ids=["unscoped", "empty-matches"])
@pytest.mark.parametrize("limit", INVALID_LIMITS)
async def test_graph_retriever_rejects_invalid_limits_before_custom_store(
    graph_chunks: list[Chunk],
    seeds: list[str],
    scope: tuple[str, ...] | None,
    limit: object,
) -> None:
    store = Mock(spec=SQLiteGraphStore)
    store.chunks_for_entities.return_value = eligible_chunks(graph_chunks, scope)
    error = ValueError if isinstance(limit, int) and not isinstance(limit, bool) else TypeError
    message = (
        r"^limit must be >= 0$"
        if error is ValueError
        else r"^limit must be an integer \(excluding booleans\)$"
    )

    with pytest.raises(error, match=message):
        results = await GraphRetriever(store).retrieve(
            seeds,
            limit=limit,  # type: ignore[arg-type]
            document_ids=scope,
        )
        pytest.fail(f"GraphRetriever accepted {limit!r} and returned {len(results)} results")

    store.chunks_for_entities.assert_not_called()


@pytest.mark.parametrize("lookup", ["chunks_for_entities", "neighbours"])
@pytest.mark.parametrize("scope", SCOPES)
def test_zero_graph_store_limit_skips_sql(
    graph_store: SQLiteGraphStore,
    monkeypatch: pytest.MonkeyPatch,
    lookup: str,
    scope: tuple[str, ...] | None,
) -> None:
    connect = Mock(wraps=sqlite3.connect)
    monkeypatch.setattr("storage.graph_store.sqlite3.connect", connect)

    assert getattr(graph_store, lookup)(["Seed"], limit=0, document_ids=scope) == []

    connect.assert_not_called()


@pytest.mark.parametrize("seeds", [[], ["Seed"]], ids=["empty-seeds", "seeds"])
@pytest.mark.parametrize("scope", SCOPES)
async def test_zero_graph_retriever_limit_skips_custom_store(
    seeds: list[str], scope: tuple[str, ...] | None
) -> None:
    store = Mock(spec=SQLiteGraphStore)
    store.chunks_for_entities.return_value = []

    assert await GraphRetriever(store).retrieve(seeds, limit=0, document_ids=scope) == []

    store.chunks_for_entities.assert_not_called()


@pytest.mark.parametrize("lookup", ["chunks_for_entities", "neighbours"])
@pytest.mark.parametrize("limit", [0, -1, pytest.param(10**400, id="huge")])
def test_graph_store_empty_scope_still_fails_before_sql(
    graph_store: SQLiteGraphStore,
    monkeypatch: pytest.MonkeyPatch,
    lookup: str,
    limit: int,
) -> None:
    connect = Mock(wraps=sqlite3.connect)
    monkeypatch.setattr("storage.graph_store.sqlite3.connect", connect)

    with pytest.raises(ValidationError) as error:
        getattr(graph_store, lookup)([], limit=limit, document_ids=[])

    assert error.value.errors()[0]["type"] == "too_short"
    connect.assert_not_called()


@pytest.mark.parametrize("kind", ["graph", "multihop"])
@pytest.mark.parametrize("limit", [0, -1, pytest.param(10**400, id="huge")])
async def test_graph_retriever_empty_scope_still_fails_before_custom_store(
    kind: str, limit: int
) -> None:
    store = Mock(spec=SQLiteGraphStore)

    with pytest.raises(ValidationError) as error:
        if kind == "graph":
            await GraphRetriever(store).retrieve([], limit=limit, document_ids=[])
        else:
            await MultiHopRetriever(store).retrieve("Seed", [], limit=limit, document_ids=[])

    assert error.value.errors()[0]["type"] == "too_short"
    store.chunks_for_entities.assert_not_called()
    store.neighbours.assert_not_called()


@pytest.mark.parametrize("lookup", ["chunks_for_entities", "neighbours"])
@pytest.mark.parametrize("limit", VALID_LIMITS)
@pytest.mark.parametrize("scope", SCOPES)
def test_graph_store_valid_limits_preserve_order_and_scope(
    graph_store: SQLiteGraphStore,
    graph_chunks: list[Chunk],
    lookup: str,
    limit: int,
    scope: tuple[str, ...] | None,
) -> None:
    expected = eligible_chunks(graph_chunks, scope)[:limit]
    if lookup == "chunks_for_entities":
        assert graph_store.chunks_for_entities(["sEeD"], limit=limit, document_ids=scope) == (
            expected
        )
    else:
        assert graph_store.neighbours(["sEeD"], limit=limit, document_ids=scope) == [
            f"Node{chunk.chunk_id[1:]}" for chunk in expected
        ]


@pytest.mark.parametrize("kind", ["graph", "multihop"])
@pytest.mark.parametrize("limit", VALID_LIMITS)
@pytest.mark.parametrize("scope", SCOPES)
async def test_graph_retrievers_valid_limits_with_real_storage(
    graph_store: SQLiteGraphStore,
    graph_chunks: list[Chunk],
    kind: str,
    limit: int,
    scope: tuple[str, ...] | None,
) -> None:
    if kind == "graph":
        results = await GraphRetriever(graph_store).retrieve(
            ["sEeD"], limit=limit, document_ids=scope
        )
    else:
        results = await MultiHopRetriever(graph_store).retrieve(
            "Seed", ["sEeD"], depth=2, limit=limit, document_ids=scope
        )

    assert results == [
        SearchResult(
            chunk=chunk,
            score=1.0,
            retriever=kind,
            path=["sEeD"] if kind == "graph" else ["seed"],
        )
        for chunk in eligible_chunks(graph_chunks, scope)[:limit]
    ]


@pytest.mark.parametrize("limit", [1, 2, 10, pytest.param(10**400, id="huge")])
@pytest.mark.parametrize("scope", [None, ("selected",)])
async def test_graph_retriever_preserves_custom_store_count_and_final_slice(
    graph_chunks: list[Chunk], limit: int, scope: tuple[str, ...] | None
) -> None:
    store = Mock(spec=SQLiteGraphStore)
    chunks = eligible_chunks(graph_chunks, scope)
    store.chunks_for_entities.return_value = chunks
    seeds = ["sEeD", "SEED"]

    results = await GraphRetriever(store).retrieve(seeds, limit=limit, document_ids=scope)

    assert results == [
        SearchResult(chunk=chunk, score=1.0, retriever="graph", path=seeds)
        for chunk in chunks[:limit]
    ]
    if scope is None:
        store.chunks_for_entities.assert_called_once_with(seeds, limit=limit)
    else:
        store.chunks_for_entities.assert_called_once_with(seeds, limit=limit, document_ids=scope)


@pytest.mark.parametrize("scope", SCOPES)
async def test_graph_lookup_defaults_are_unchanged(
    graph_store: SQLiteGraphStore,
    graph_chunks: list[Chunk],
    scope: tuple[str, ...] | None,
) -> None:
    expected = eligible_chunks(graph_chunks, scope)

    assert graph_store.chunks_for_entities(["Seed"], document_ids=scope) == expected[:10]
    assert graph_store.neighbours(["Seed"], document_ids=scope) == [
        f"Node{chunk.chunk_id[1:]}" for chunk in expected[:20]
    ]
    direct = await GraphRetriever(graph_store).retrieve(["Seed"], document_ids=scope)
    multihop = await MultiHopRetriever(graph_store).retrieve("Seed", ["Seed"], document_ids=scope)
    assert [result.chunk for result in direct] == expected[:10]
    assert [result.chunk for result in multihop] == expected[:10]
