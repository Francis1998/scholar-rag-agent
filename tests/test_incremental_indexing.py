"""Exact-ID upserts keep incremental retrieval equivalent to a fresh index."""

from unittest.mock import Mock

import pytest

from retrieval.dense import DenseRetriever
from retrieval.embeddings import HashEmbeddingModel
from retrieval.models import Chunk
from retrieval.sparse import BM25Retriever


def make_chunk(chunk_id: str, text: str = "retrieval", document_id: str = "selected") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        title=f"Synthetic {chunk_id}",
        text=text,
        source="synthetic:index-upsert",
    )


@pytest.mark.parametrize("retriever_type", [DenseRetriever, BM25Retriever])
@pytest.mark.parametrize("same_batch", [False, True])
async def test_replaying_ids_does_not_fill_top_k_with_duplicates(
    retriever_type: type[DenseRetriever] | type[BM25Retriever], same_batch: bool
) -> None:
    chunks = [make_chunk(chunk_id) for chunk_id in ("c", "a", "b")]
    retriever = retriever_type()
    if same_batch:
        retriever.add_chunks([*chunks, chunks[1], chunks[1]])
    else:
        retriever.add_chunks(chunks)
        retriever.add_chunks([chunks[1]])
        retriever.add_chunks([chunks[1]])

    results = await retriever.retrieve("retrieval", limit=2)
    assert [result.chunk.chunk_id for result in results] == ["a", "b"]
    all_results = await retriever.retrieve("retrieval")
    assert [result.chunk.chunk_id for result in all_results] == ["a", "b", "c"]


@pytest.mark.parametrize("retriever_type", [DenseRetriever, BM25Retriever])
async def test_new_distinct_ids_with_identical_text_remain_distinct(
    retriever_type: type[DenseRetriever] | type[BM25Retriever],
) -> None:
    retriever = retriever_type()
    retriever.add_chunks([])
    assert await retriever.retrieve("retrieval") == []
    retriever.add_chunks([make_chunk("c"), make_chunk("b")])
    retriever.add_chunks([make_chunk("a")])
    retriever.add_chunks([])

    results = await retriever.retrieve("retrieval")
    assert [result.chunk.chunk_id for result in results] == ["a", "b", "c"]
    assert len({result.score for result in results}) == 1
    assert results[0].score > 0


@pytest.mark.parametrize("retriever_type", [DenseRetriever, BM25Retriever])
@pytest.mark.parametrize("same_batch", [False, True])
async def test_replacement_keeps_current_payload_scores_and_document_scope(
    retriever_type: type[DenseRetriever] | type[BM25Retriever], same_batch: bool
) -> None:
    original = make_chunk("a", "obsolete obsolete shared", "old-paper")
    other = make_chunk("b", "current shared", "other-paper")
    replacement = Chunk(
        chunk_id=original.chunk_id,
        document_id="new-paper",
        title="Updated synthetic title",
        text="current current evidence",
        source="synthetic:updated",
        metadata={"revision": "2"},
    )
    retriever = retriever_type()
    if same_batch:
        retriever.add_chunks([original, other, replacement])
    else:
        retriever.add_chunks([original, other])
        retriever.add_chunks([replacement])
    fresh = retriever_type()
    fresh.add_chunks([other, replacement])

    results = await retriever.retrieve("current")
    assert [result.chunk for result in results if result.chunk.chunk_id == "a"] == [replacement]
    assert results == await fresh.retrieve("current")
    assert await retriever.retrieve("current", document_ids=["old-paper"]) == []
    scoped = await retriever.retrieve("current", limit=1, document_ids=["new-paper"])
    assert [result.chunk for result in scoped] == [replacement]
    assert scoped == await fresh.retrieve("current", limit=1, document_ids=["new-paper"])


@pytest.mark.parametrize("retriever_type", [DenseRetriever, BM25Retriever])
async def test_scope_precedes_unique_top_k_after_replay(
    retriever_type: type[DenseRetriever] | type[BM25Retriever],
) -> None:
    excluded = [make_chunk(f"a-excluded-{index}", document_id="excluded") for index in range(4)]
    selected = [
        make_chunk(chunk_id, "retrieval " + "background " * 10)
        for chunk_id in ("b-selected", "c-selected")
    ]
    retriever = retriever_type()
    retriever.add_chunks([*excluded, *selected, selected[0], selected[0]])

    unscoped = await retriever.retrieve("retrieval", limit=2)
    assert [result.chunk.chunk_id for result in unscoped] == [
        "a-excluded-0",
        "a-excluded-1",
    ]
    scoped = await retriever.retrieve("retrieval", limit=2, document_ids=["selected"])
    assert [result.chunk.chunk_id for result in scoped] == ["b-selected", "c-selected"]
    all_results = await retriever.retrieve("retrieval", limit=20)
    assert scoped == [result for result in all_results if result.chunk.document_id == "selected"]
    assert await retriever.retrieve("retrieval", document_ids=["missing"]) == []


@pytest.mark.parametrize("same_batch", [False, True])
@pytest.mark.parametrize(
    "replacement_text",
    ["old old shared", "current current current shared evidence", "", "!!!"],
    ids=["replay", "replacement", "empty-replacement", "empty-tokens-replacement"],
)
async def test_bm25_scores_match_the_unique_final_corpus(
    same_batch: bool, replacement_text: str
) -> None:
    original = make_chunk("a", "old old shared")
    other = make_chunk("b", "old current shared")
    background = make_chunk("c", "background")
    replacement = make_chunk("a", replacement_text)
    added = make_chunk("d", "old shared current new")
    retriever = BM25Retriever()
    if same_batch:
        retriever.add_chunks([original, other, background, replacement, added, replacement])
    else:
        retriever.add_chunks([original, other, background])
        retriever.add_chunks([replacement])
        retriever.add_chunks([added])
        retriever.add_chunks([replacement])
        retriever.add_chunks([])
    fresh = BM25Retriever()
    fresh.add_chunks([replacement, other, background, added])

    for query in ("old", "shared", "current", "new", "old current shared", "background", "missing"):
        actual = await retriever.retrieve(query, limit=20)
        expected = await fresh.retrieve(query, limit=20)
        # Check statistics independently of the separate duplicate-result regressions.
        assert {result.chunk.chunk_id: result.score for result in actual} == pytest.approx(
            {result.chunk.chunk_id: result.score for result in expected}
        ), query


@pytest.mark.parametrize("chunk_id", ["a", "new"])
async def test_failed_embedding_leaves_the_previous_index_usable(
    monkeypatch: pytest.MonkeyPatch, chunk_id: str
) -> None:
    embedder = HashEmbeddingModel()
    retriever = DenseRetriever(embedder)
    retriever.add_chunks([make_chunk("a"), make_chunk("b", "background")])
    before = await retriever.retrieve("retrieval")

    with monkeypatch.context() as patch:
        patch.setattr(embedder, "embed", Mock(side_effect=ValueError("embedding failed")))
        with pytest.raises(ValueError, match="embedding failed"):
            retriever.add_chunks([make_chunk(chunk_id, "rejected replacement")])

    assert await retriever.retrieve("retrieval") == before
