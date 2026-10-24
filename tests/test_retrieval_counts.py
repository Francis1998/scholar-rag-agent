"""Result counts fail before work, without weakening explicit document scopes."""

import math
from collections.abc import Awaitable, Callable
from pathlib import Path
from unittest.mock import AsyncMock, Mock, call

import pytest

from agent.executor import Executor
from agent.models import QueryIntent, QueryObservation, QueryPlan, RetrievalTask
from llm.fake import FakeLLMAdapter
from retrieval.citations import CitationGrounder
from retrieval.dense import DenseRetriever
from retrieval.embeddings import HashEmbeddingModel
from retrieval.hybrid import HybridRetriever
from retrieval.hyde import HyDEExpander
from retrieval.mmr import MMRDiversifier
from retrieval.models import Chunk, Entity, SearchResult
from retrieval.multihop import MultiHopRetriever
from retrieval.rerank import AdaptiveReranker
from retrieval.rrf import reciprocal_rank_fusion
from retrieval.scope import DocumentIdsInput
from retrieval.sparse import BM25Retriever, tokenize
from storage.graph_store import SQLiteGraphStore

Retrieve = Callable[[int, DocumentIdsInput | None], Awaitable[list[SearchResult]]]
RetrievalBoundary = tuple[Retrieve, str, list[Mock]]
INVALID_COUNTS = [
    pytest.param(-1, id="negative"),
    pytest.param(-(10**400), id="huge-negative"),
    pytest.param(True, id="true"),
    pytest.param(False, id="false"),
    pytest.param(0.0, id="zero-float"),
    pytest.param(1.0, id="integral-float"),
    pytest.param(1.5, id="fractional-float"),
    pytest.param(math.nan, id="nan"),
    pytest.param(math.inf, id="infinity"),
    pytest.param(-math.inf, id="negative-infinity"),
    pytest.param("1", id="numeric-string"),
    pytest.param("private-count-value", id="private-string"),
    pytest.param(None, id="none"),
    pytest.param([], id="list"),
    pytest.param({}, id="dict"),
]


def _chunks(count: int = 3) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"c{index:02d}",
            document_id="excluded" if index == 2 else "selected",
            title=f"Paper {index}",
            text="retrieval",
            source="fixture",
            metadata={"index": str(index)},
        )
        for index in reversed(range(count))
    ]


def _plan(*, tasks: int = 1, scoped: bool = True) -> QueryPlan:
    return QueryPlan(
        run_id="count-contract",
        observation=QueryObservation(
            original_query="retrieval",
            intent=QueryIntent.SYNTHESIS,
            document_ids=("selected",) if scoped else None,
        ),
        tasks=[
            RetrievalTask(
                task_id=f"task-{index}",
                query=f"retrieval {index}",
                rationale="Synthetic count fixture",
                target_entities=["Seed"],
                max_hops=2,
            )
            for index in range(tasks)
        ],
        rationale_trace=[],
    )


def _executor(hybrid: HybridRetriever, multihop: MultiHopRetriever) -> Executor:
    return Executor(hybrid, multihop, AdaptiveReranker(), FakeLLMAdapter(), CitationGrounder())


@pytest.fixture(params=[False, True], ids=["empty", "populated"])
def populated(request: pytest.FixtureRequest) -> bool:
    return bool(request.param)


@pytest.fixture(params=["dense", "bm25", "hybrid", "multihop", "executor"])
def boundary(
    request: pytest.FixtureRequest, populated: bool, monkeypatch: pytest.MonkeyPatch
) -> RetrievalBoundary:
    chunks = _chunks() if populated else []
    embedder = HashEmbeddingModel()
    dense = DenseRetriever(embedder)
    sparse = BM25Retriever()
    llm = FakeLLMAdapter()
    hyde = HyDEExpander(llm)
    diversifier = MMRDiversifier()
    hybrid = HybridRetriever(dense, sparse, hyde, diversifier)
    hybrid.add_chunks(chunks)
    store = Mock(spec=SQLiteGraphStore)
    store.chunks_for_entities.return_value = chunks
    store.neighbours.return_value = []
    multihop = MultiHopRetriever(store)
    executor = _executor(hybrid, multihop)

    embed = Mock(wraps=embedder.embed)
    tokenize_spy = Mock(wraps=tokenize)
    score = Mock(wraps=sparse._score)
    monkeypatch.setattr(embedder, "embed", embed)
    monkeypatch.setattr("retrieval.sparse.tokenize", tokenize_spy)
    monkeypatch.setattr(sparse, "_score", score)
    work = [embed, tokenize_spy, score]
    if request.param in ("hybrid", "executor"):
        for component, method in (
            (llm, "generate"),
            (hyde, "expand"),
            (dense, "retrieve"),
            (sparse, "retrieve"),
        ):
            spy = AsyncMock(wraps=getattr(component, method))
            monkeypatch.setattr(component, method, spy)
            work.append(spy)
        diversify = Mock(wraps=diversifier.diversify)
        fusion = Mock(wraps=reciprocal_rank_fusion)
        monkeypatch.setattr(diversifier, "diversify", diversify)
        monkeypatch.setattr("retrieval.hybrid.reciprocal_rank_fusion", fusion)
        work.extend([diversify, fusion])
    if request.param == "executor":
        for component in (hybrid, multihop):
            spy = AsyncMock(wraps=component.retrieve)
            monkeypatch.setattr(component, "retrieve", spy)
            work.append(spy)
    work.extend([store.chunks_for_entities, store.neighbours])

    async def retrieve(limit: int, scope: DocumentIdsInput | None) -> list[SearchResult]:
        if request.param == "executor":
            return await executor.retrieve(_plan(), max_results=limit, document_ids=scope)
        if request.param == "multihop":
            return await multihop.retrieve("retrieval", ["Seed"], limit=limit, document_ids=scope)
        retriever = {"dense": dense, "bm25": sparse, "hybrid": hybrid}[request.param]
        return await retriever.retrieve("retrieval", limit=limit, document_ids=scope)

    return retrieve, "max_results" if request.param == "executor" else "limit", work


@pytest.mark.parametrize("count", INVALID_COUNTS)
async def test_invalid_counts_fail_before_work(boundary: RetrievalBoundary, count: object) -> None:
    retrieve, field, work = boundary
    error = ValueError if isinstance(count, int) and not isinstance(count, bool) else TypeError
    message = (
        f"{field} must be >= 0"
        if error is ValueError
        else rf"{field} must be an integer \(excluding booleans\)"
    )

    with pytest.raises(error, match=f"^{message}$"):
        await retrieve(count, None)  # type: ignore[arg-type]

    for spy in work:
        spy.assert_not_called()


@pytest.mark.parametrize("scope", [None, ["selected"]], ids=["unscoped", "scoped"])
async def test_zero_returns_without_work(
    boundary: RetrievalBoundary, scope: DocumentIdsInput | None
) -> None:
    retrieve, _, work = boundary

    assert await retrieve(0, scope) == []

    for spy in work:
        spy.assert_not_called()


@pytest.mark.parametrize("scope", [[], [""], ["selected"] * 101])
async def test_zero_still_validates_scope(
    boundary: RetrievalBoundary, scope: DocumentIdsInput
) -> None:
    retrieve, _, work = boundary

    with pytest.raises(ValueError):
        await retrieve(0, scope)

    for spy in work:
        spy.assert_not_called()


@pytest.mark.parametrize("retriever_type", [DenseRetriever, BM25Retriever])
async def test_negative_limit_cannot_leak_all_but_one_chunk(
    retriever_type: type[DenseRetriever] | type[BM25Retriever],
) -> None:
    retriever = retriever_type()
    retriever.add_chunks(_chunks())

    with pytest.raises(ValueError, match=r"^limit must be >= 0$"):
        leaked = await retriever.retrieve("retrieval", limit=-1)
        pytest.fail(f"negative limit leaked {[result.chunk.chunk_id for result in leaked]}")


@pytest.mark.parametrize("count", INVALID_COUNTS)
@pytest.mark.parametrize("depth", [0, 2])
async def test_multihop_rejects_counts_even_without_seeds(count: object, depth: int) -> None:
    store = Mock(spec=SQLiteGraphStore)
    retriever = MultiHopRetriever(store)
    error = ValueError if isinstance(count, int) and not isinstance(count, bool) else TypeError

    with pytest.raises(error, match=r"^limit must be "):
        await retriever.retrieve("retrieval", [], depth=depth, limit=count)  # type: ignore[arg-type]

    store.chunks_for_entities.assert_not_called()
    store.neighbours.assert_not_called()


@pytest.mark.parametrize("count", INVALID_COUNTS)
async def test_executor_rejects_counts_even_without_tasks(count: object) -> None:
    hybrid = Mock(spec=HybridRetriever)
    multihop = Mock(spec=MultiHopRetriever)
    error = ValueError if isinstance(count, int) and not isinstance(count, bool) else TypeError

    with pytest.raises(error, match=r"^max_results must be "):
        await _executor(hybrid, multihop).retrieve(
            _plan(tasks=0),
            max_results=count,  # type: ignore[arg-type]
        )

    hybrid.retrieve.assert_not_called()
    multihop.retrieve.assert_not_called()


@pytest.mark.parametrize("tasks", [0, 1])
async def test_zero_executor_preserves_plan_scope_equality(tasks: int) -> None:
    hybrid = Mock(spec=HybridRetriever)
    multihop = Mock(spec=MultiHopRetriever)

    with pytest.raises(ValueError, match="Retrieval document_ids do not match the plan's scope"):
        await _executor(hybrid, multihop).retrieve(
            _plan(tasks=tasks), max_results=0, document_ids=["excluded"]
        )

    hybrid.retrieve.assert_not_called()
    multihop.retrieve.assert_not_called()


@pytest.mark.parametrize("kind", ["dense", "bm25", "hybrid", "hybrid-mmr"])
@pytest.mark.parametrize("scope", [None, ("selected",), ("missing",)])
@pytest.mark.parametrize("limit", [1, 2, 10, 10**400])
async def test_positive_counts_preserve_rankings_provenance_and_scope(
    kind: str, scope: tuple[str, ...] | None, limit: int
) -> None:
    chunks = _chunks(12)
    retriever = (
        DenseRetriever()
        if kind == "dense"
        else BM25Retriever()
        if kind == "bm25"
        else HybridRetriever(
            DenseRetriever(),
            BM25Retriever(),
            HyDEExpander(FakeLLMAdapter()),
            MMRDiversifier() if kind == "hybrid-mmr" else None,
        )
    )
    retriever.add_chunks(chunks)
    eligible = sorted(
        [chunk for chunk in chunks if scope is None or chunk.document_id in scope],
        key=lambda chunk: chunk.chunk_id,
    )
    expected = [
        SearchResult(
            chunk=chunk,
            score=(
                1.0
                if kind == "dense"
                else math.log(1 + 0.5 / 12.5)
                if kind == "bm25"
                else 2 / (60 + rank)
            ),
            retriever=kind if kind in ("dense", "bm25") else "rrf",
            path=[] if kind in ("dense", "bm25") else ["dense", "bm25"],
        )
        for rank, chunk in enumerate(eligible[:limit], start=1)
    ]

    assert await retriever.retrieve("retrieval", limit=limit, document_ids=scope) == expected
    assert len(await retriever.retrieve("retrieval")) == 10
    assert chunks == _chunks(12)


@pytest.mark.parametrize("limit", [1, 2, 10, 10**400])
async def test_multihop_preserves_forwarded_counts_and_paths(limit: int) -> None:
    chunks = sorted(_chunks(), key=lambda chunk: chunk.chunk_id)[:limit]
    store = Mock(spec=SQLiteGraphStore)
    store.chunks_for_entities.return_value = chunks
    store.neighbours.return_value = []

    results = await MultiHopRetriever(store).retrieve("retrieval", ["Seed"], limit=limit)

    assert results == [
        SearchResult(chunk=chunk, score=1.0, retriever="multihop", path=["seed"])
        for chunk in chunks
    ]
    store.chunks_for_entities.assert_called_once_with(["seed"], limit=limit)
    store.neighbours.assert_called_once_with(["seed"], limit=limit * 4)


@pytest.mark.parametrize("limit", [0, 1, 2, 10])
@pytest.mark.parametrize("scope", [None, ("selected",), ("missing",)])
async def test_multihop_real_storage_preserves_scoped_counts(
    tmp_path: Path, populated: bool, limit: int, scope: tuple[str, ...] | None
) -> None:
    store = SQLiteGraphStore(tmp_path / "counts.sqlite3")
    chunks = _chunks() if populated else []
    for chunk in chunks:
        store.add_mentions(chunk, [Entity(name="Seed")])
    eligible = sorted(
        [chunk for chunk in chunks if scope is None or chunk.document_id in scope],
        key=lambda chunk: chunk.chunk_id,
    )

    results = await MultiHopRetriever(store).retrieve(
        "retrieval", ["Seed"], limit=limit, document_ids=scope
    )

    assert results == [
        SearchResult(chunk=chunk, score=1.0, retriever="multihop", path=["seed"])
        for chunk in eligible[:limit]
    ]


@pytest.mark.parametrize("max_results", [1, 2, 8, 10**400])
async def test_executor_preserves_global_cap_best_score_and_forwarding(max_results: int) -> None:
    first, second, third = sorted(_chunks(), key=lambda chunk: chunk.chunk_id)
    hybrid = Mock(spec=HybridRetriever)
    multihop = Mock(spec=MultiHopRetriever)
    best = SearchResult(chunk=first, score=0.9, retriever="multihop", path=["seed"])
    next_best = SearchResult(chunk=third, score=0.7, retriever="rrf", path=["dense", "bm25"])
    last = SearchResult(chunk=second, score=0.4, retriever="rrf", path=["dense", "bm25"])
    hybrid.retrieve.side_effect = [
        [SearchResult(chunk=first, score=0.5, retriever="rrf"), last],
        [next_best, SearchResult(chunk=first, score=0.3, retriever="rrf")],
    ]
    multihop.retrieve.side_effect = [[best], []]
    plan = _plan(tasks=2, scoped=False)

    results = await _executor(hybrid, multihop).retrieve(plan, max_results=max_results)

    assert results == [best, next_best, last][:max_results]
    assert results[0] is best
    assert hybrid.retrieve.await_args_list == [
        call(task.query, limit=max_results) for task in plan.tasks
    ]
    assert multihop.retrieve.await_args_list == [
        call(query=task.query, seed_entities=["Seed"], depth=2, limit=max_results)
        for task in plan.tasks
    ]
