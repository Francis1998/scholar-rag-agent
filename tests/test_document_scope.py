"""Candidate-level regressions for opt-in, fail-closed document selection."""

import asyncio
from pathlib import Path

import pytest
from pydantic import ValidationError

from api.schemas import QueryRequest
from retrieval.dense import DenseRetriever
from retrieval.graph import GraphRetriever
from retrieval.hybrid import HybridRetriever
from retrieval.hyde import HyDEExpander
from retrieval.models import Chunk, Entity, EntityEdge
from retrieval.multihop import MultiHopRetriever
from retrieval.sparse import BM25Retriever
from storage.graph_store import SQLiteGraphStore


def chunk(chunk_id: str, document_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        title=f"Synthetic {chunk_id}",
        text=text,
        source="synthetic:scope-test",
    )


def test_scope_normalizes_without_changing_caller_input() -> None:
    supplied = [" doc-b ", "doc-a", "doc-b", "Doc-A", "paper:10.1/example"]
    request = QueryRequest.model_validate({"query": "retrieval", "document_ids": supplied})
    assert request.document_ids == ("doc-b", "doc-a", "Doc-A", "paper:10.1/example")
    assert supplied == [" doc-b ", "doc-a", "doc-b", "Doc-A", "paper:10.1/example"]
    supplied[:] = ["different"]
    assert request.document_ids == ("doc-b", "doc-a", "Doc-A", "paper:10.1/example")
    assert QueryRequest(query="retrieval").document_ids is None


@pytest.mark.parametrize(
    "scope",
    [
        None,
        [],
        "",
        "doc-a",
        1,
        True,
        {"doc-a": True},
        [None],
        [1],
        [False],
        [""],
        [" \t "],
        ["x" * 129],
        ["doc-a"] * 101,
    ],
)
def test_invalid_explicit_api_scope_is_rejected(scope: object) -> None:
    with pytest.raises(ValidationError):
        QueryRequest.model_validate({"query": "retrieval", "document_ids": scope})


def test_scope_bounds_apply_before_deduplication() -> None:
    assert QueryRequest.model_validate(
        {"query": "retrieval", "document_ids": ["x" * 128]}
    ).document_ids == ("x" * 128,)
    distinct = [f"doc-{index}" for index in range(100)]
    assert (
        len(
            QueryRequest.model_validate(
                {"query": "retrieval", "document_ids": distinct}
            ).document_ids
        )
        == 100
    )
    assert QueryRequest.model_validate(
        {"query": "retrieval", "document_ids": ["doc-a"] * 100}
    ).document_ids == ("doc-a",)


def test_imported_ids_keep_unicode_quotes_and_internal_whitespace() -> None:
    ids = ["paper-\u03b2", "https://doi.org/10.1/'quoted'", "paper with spaces", "line\nbreak"]
    request = QueryRequest.model_validate({"query": "retrieval", "document_ids": ids})
    assert request.document_ids == tuple(ids)


@pytest.mark.parametrize("kind", ["dense", "bm25", "hybrid"])
async def test_scope_filters_before_top_k_and_rrf(kind: str) -> None:
    query = "retrieval"
    expanded = await HyDEExpander().expand(query)
    excluded = [chunk(f"excluded-{index}", "excluded", expanded) for index in range(6)]
    allowed = chunk("allowed", "allowed", "retrieval " + "unrelated " * 80)
    retriever = (
        DenseRetriever()
        if kind == "dense"
        else BM25Retriever()
        if kind == "bm25"
        else HybridRetriever(DenseRetriever(), BM25Retriever(), HyDEExpander())
    )
    retriever.add_chunks([*excluded, allowed])
    search_query = query if kind == "hybrid" else expanded
    baseline = await retriever.retrieve(search_query, limit=2)
    assert len(baseline) == 2
    assert all(result.chunk.document_id == "excluded" for result in baseline)

    scoped = await retriever.retrieve(search_query, limit=2, document_ids=["allowed"])
    assert [result.chunk.chunk_id for result in scoped] == ["allowed"]
    if kind == "hybrid":
        assert scoped[0].retriever == "rrf"
        assert scoped[0].path == ["dense", "bm25"]
        assert scoped[0].score == pytest.approx(2 / 61)
    else:
        all_results = await retriever.retrieve(search_query, limit=10)
        allowed_score = next(r.score for r in all_results if r.chunk.document_id == "allowed")
        assert scoped[0].score == allowed_score
    assert await retriever.retrieve(search_query, document_ids=["missing"]) == []
    assert await retriever.retrieve(search_query, limit=2) == baseline


@pytest.mark.parametrize("kind", ["dense", "bm25", "hybrid"])
@pytest.mark.parametrize("scope", [[], [""], ["doc-a"] * 101])
async def test_retriever_invalid_scope_never_means_global(kind: str, scope: list[str]) -> None:
    retriever = (
        DenseRetriever()
        if kind == "dense"
        else BM25Retriever()
        if kind == "bm25"
        else HybridRetriever(DenseRetriever(), BM25Retriever(), HyDEExpander())
    )
    retriever.add_chunks([chunk("a", "doc-a", "retrieval")])
    with pytest.raises(ValueError, match=r"document_ids|Tuple|tuple"):
        await retriever.retrieve("retrieval", document_ids=scope)


def test_graph_chunk_and_edge_scope_is_applied_before_sql_limit(tmp_path: Path) -> None:
    store = SQLiteGraphStore(tmp_path / "graph.sqlite3")
    for index in range(8):
        excluded = chunk(f"x-{index}", "excluded", f"Alpha A{index}")
        store.add_mentions(excluded, [Entity(name="Alpha"), Entity(name=f"A{index}")])
        store.add_edges(
            [EntityEdge(source="Alpha", target=f"A{index}", chunk_id=excluded.chunk_id)]
        )
    allowed = chunk("z-allowed", "paper:10.1/'allowed-\u03b2'", "Alpha Zulu")
    store.add_mentions(allowed, [Entity(name="Alpha"), Entity(name="Zulu")])
    store.add_edges([EntityEdge(source="Alpha", target="Zulu", chunk_id=allowed.chunk_id)])
    assert store.chunks_for_entities(["Alpha"], limit=1)[0].document_id == "excluded"
    assert store.neighbours(["Alpha"], limit=1) == ["A0"]

    scope = [allowed.document_id]
    assert store.chunks_for_entities(["Alpha"], limit=1, document_ids=scope) == [allowed]
    assert store.neighbours(["Alpha"], limit=1, document_ids=scope) == ["Zulu"]
    assert store.neighbours(["Zulu"], limit=1, document_ids=scope) == ["Alpha"]
    assert store.chunks_for_entities(["Alpha"], document_ids=["missing"]) == []
    assert store.neighbours(["Alpha"], document_ids=["missing"]) == []
    assert store.chunks_for_entities(["Alpha') OR 1=1 --"], document_ids=scope) == []
    assert store.neighbours(["Zulu') OR 1=1 --"], document_ids=scope) == []


async def test_multihop_cannot_use_an_excluded_document_as_a_bridge(tmp_path: Path) -> None:
    store = SQLiteGraphStore(tmp_path / "bridge.sqlite3")
    start = chunk("start", "selected", "Alpha evidence")
    bridge = chunk("bridge", "excluded", "Alpha Beta excluded bridge")
    tail = chunk("tail", "selected", "Beta Gamma otherwise allowed evidence")
    store.add_mentions(start, [Entity(name="Alpha")])
    store.add_mentions(bridge, [Entity(name="Alpha"), Entity(name="Beta")])
    store.add_mentions(tail, [Entity(name="Beta"), Entity(name="Gamma")])
    store.add_edges(
        [
            EntityEdge(source="Alpha", target="Beta", chunk_id="bridge"),
            EntityEdge(source="Beta", target="Gamma", chunk_id="tail"),
            EntityEdge(source="Alpha", target="Gamma", chunk_id="not-in-graph-chunks"),
        ]
    )
    retriever = MultiHopRetriever(store)
    all_results = await retriever.retrieve("Alpha", ["Alpha"], depth=3)
    assert {r.chunk.chunk_id for r in all_results} == {"start", "bridge", "tail"}

    scoped = await retriever.retrieve("Alpha", ["Alpha"], depth=3, document_ids=["selected"])
    assert [r.chunk.chunk_id for r in scoped] == ["start"]
    assert scoped[0].path == ["alpha"]
    assert await retriever.retrieve("Alpha", ["Alpha"], document_ids=["missing"]) == []
    direct = await GraphRetriever(store).retrieve(["Alpha"], document_ids=["selected"])
    assert [r.chunk.chunk_id for r in direct] == ["start"]


def test_graph_validates_scope_even_without_entities(tmp_path: Path) -> None:
    store = SQLiteGraphStore(tmp_path / "empty.sqlite3")
    with pytest.raises(ValueError):
        store.chunks_for_entities([], document_ids=[])
    with pytest.raises(ValueError):
        store.neighbours([], document_ids=[])


async def test_multihop_validates_scope_even_at_zero_depth(tmp_path: Path) -> None:
    retriever = MultiHopRetriever(SQLiteGraphStore(tmp_path / "empty.sqlite3"))
    with pytest.raises(ValueError):
        await retriever.retrieve("q", [], depth=0, document_ids=[])


async def test_hybrid_snapshots_scope_before_hyde_await(monkeypatch: pytest.MonkeyPatch) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    hyde = HyDEExpander()
    original_expand = hyde.expand

    async def delayed_expand(query: str) -> str:
        entered.set()
        await release.wait()
        return await original_expand(query)

    monkeypatch.setattr(hyde, "expand", delayed_expand)
    retriever = HybridRetriever(DenseRetriever(), BM25Retriever(), hyde)
    retriever.add_chunks([chunk("a", "a", "retrieval"), chunk("b", "b", "retrieval")])
    supplied = ["a"]
    pending = asyncio.create_task(retriever.retrieve("retrieval", document_ids=supplied))
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        supplied[:] = ["b"]
    finally:
        release.set()
    results = await asyncio.wait_for(pending, timeout=2)
    assert [r.chunk.document_id for r in results] == ["a"]
