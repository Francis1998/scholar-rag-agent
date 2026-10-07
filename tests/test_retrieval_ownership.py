"""Indexed state and returned chunks must not share caller-owned mutable data."""

from unittest.mock import patch

import pytest

from retrieval.dense import DenseRetriever
from retrieval.embeddings import HashEmbeddingModel
from retrieval.hybrid import HybridRetriever
from retrieval.hyde import HyDEExpander
from retrieval.models import Chunk
from retrieval.sparse import BM25Retriever

Retriever = DenseRetriever | BM25Retriever | HybridRetriever


def make_chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id=chunk_id,
            document_id=document_id,
            title=f"Synthetic {chunk_id}",
            text=text,
            source=f"synthetic:{chunk_id}",
            metadata={"revision": "1"},
        )
        for chunk_id, document_id, text in [
            ("a", "paper", "alpha alpha evidence"),
            ("b", "paper", "beta evidence"),
            ("c", "other", "gamma evidence"),
        ]
    ]


async def assert_matches_fresh(actual: Retriever, fresh: Retriever) -> None:
    for query in ("alpha", "beta", "gamma", "unindexed"):
        for document_ids in (None, ["paper"], ["other"], ["unindexed-paper"]):
            for limit in (0, 1, -1, 10):
                assert await actual.retrieve(
                    query, limit=limit, document_ids=document_ids
                ) == await fresh.retrieve(query, limit=limit, document_ids=document_ids), (
                    query,
                    document_ids,
                    limit,
                )


def change_payload(chunk: Chunk) -> None:
    chunk.document_id = "unindexed-paper"
    chunk.text = "unindexed unindexed findings"
    chunk.title = "Changed title"
    chunk.source = "synthetic:changed"
    chunk.metadata["revision"] = "2"
    chunk.metadata["added"] = "caller-owned"


@pytest.mark.parametrize("retriever_type", [DenseRetriever, BM25Retriever])
@pytest.mark.parametrize("boundary", ["input", "result"])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("chunk_id", "unindexed-chunk"),
        ("document_id", "unindexed-paper"),
        ("text", "unindexed unindexed findings"),
        ("title", "Changed title"),
        ("source", "synthetic:changed"),
        ("metadata", "2"),
    ],
)
async def test_caller_mutations_leave_scores_scope_and_payload_unchanged(
    retriever_type: type[DenseRetriever] | type[BM25Retriever],
    boundary: str,
    field: str,
    value: str,
) -> None:
    chunks = make_chunks()
    retriever = retriever_type()
    retriever.add_chunks(chunks)
    fresh = retriever_type()
    fresh.add_chunks(make_chunks())

    owned = chunks[0]
    if boundary == "result":
        owned = (await retriever.retrieve("alpha", limit=1))[0].chunk
        assert owned.chunk_id == "a"
    if field == "metadata":
        owned.metadata["revision"] = value
        owned.metadata["added"] = "caller-owned"
    else:
        setattr(owned, field, value)

    await assert_matches_fresh(retriever, fresh)
    if boundary == "result":
        assert chunks == make_chunks()


@pytest.mark.parametrize("retriever_type", [DenseRetriever, BM25Retriever])
async def test_separate_results_own_distinct_chunks_and_metadata(
    retriever_type: type[DenseRetriever] | type[BM25Retriever],
) -> None:
    chunks = make_chunks()
    retriever = retriever_type()
    retriever.add_chunks(chunks)
    first = await retriever.retrieve("alpha")
    second = await retriever.retrieve("alpha")
    pristine = [result.model_copy(deep=True) for result in second]

    for left, right in zip(first, second, strict=True):
        assert left is not right
        assert left.chunk is not right.chunk
        assert left.chunk.metadata is not right.chunk.metadata
    change_payload(first[0].chunk)
    first[0].chunk.chunk_id = "unindexed-chunk"
    first[0].score = -999.0
    first[0].retriever = "caller"
    first[0].path.append("caller")
    first.clear()

    assert second == pristine
    assert chunks == make_chunks()
    assert await retriever.retrieve("alpha") == pristine


@pytest.mark.parametrize("retriever_type", [DenseRetriever, BM25Retriever])
@pytest.mark.parametrize("boundary", ["input", "result"])
async def test_explicit_reindexing_updates_the_snapshot(
    retriever_type: type[DenseRetriever] | type[BM25Retriever], boundary: str
) -> None:
    chunks = make_chunks()
    retriever = retriever_type()
    retriever.add_chunks(chunks)
    replacement = chunks[0]
    if boundary == "result":
        replacement = (await retriever.retrieve("alpha", limit=1))[0].chunk
    change_payload(replacement)
    expected = replacement.model_copy(deep=True)
    retriever.add_chunks([replacement])
    fresh = retriever_type()
    fresh.add_chunks([expected, *make_chunks()[1:]])

    await assert_matches_fresh(retriever, fresh)
    updated = await retriever.retrieve("unindexed", limit=1, document_ids=["unindexed-paper"])
    assert updated[0].chunk == expected
    assert updated[0].score > 0
    old_scope = await retriever.retrieve("alpha", document_ids=["paper"])
    assert [result.chunk.chunk_id for result in old_scope] == ["b"]

    replacement.chunk_id = "unindexed-chunk"
    replacement.metadata.clear()
    updated[0].chunk.text = "another caller edit"
    updated[0].chunk.metadata.clear()
    await assert_matches_fresh(retriever, fresh)


async def test_dense_snapshots_input_before_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    chunks = make_chunks()
    embedder = HashEmbeddingModel()
    embed = embedder.embed
    retriever = DenseRetriever(embedder)
    fresh = DenseRetriever()
    fresh.add_chunks(make_chunks())

    def embed_and_mutate_caller(text: str) -> list[float]:
        change_payload(chunks[0])
        chunks[0].chunk_id = "unindexed-chunk"
        return embed(text)

    with monkeypatch.context() as context:
        context.setattr(embedder, "embed", embed_and_mutate_caller)
        retriever.add_chunks(chunks)

    await assert_matches_fresh(retriever, fresh)


class ReusingEmbedder(HashEmbeddingModel):
    def __init__(self) -> None:
        super().__init__()
        self.buffer: list[float] = []

    def embed(self, text: str) -> list[float]:
        self.buffer[:] = super().embed(text)
        return self.buffer


async def test_dense_owns_embeddings_from_a_reusable_buffer() -> None:
    embedder = ReusingEmbedder()
    retriever = DenseRetriever(embedder)
    retriever.add_chunks(make_chunks())
    fresh = DenseRetriever()
    fresh.add_chunks(make_chunks())
    embedder.buffer[:] = [0.0] * embedder.dimensions

    await assert_matches_fresh(retriever, fresh)


@pytest.mark.parametrize("boundary", ["input", "dense", "bm25", "hybrid"])
async def test_hybrid_mutations_do_not_contaminate_either_component(boundary: str) -> None:
    dense, sparse = DenseRetriever(), BM25Retriever()
    hybrid = HybridRetriever(dense, sparse, HyDEExpander())
    chunks = make_chunks()
    hybrid.add_chunks(chunks)
    fresh_dense, fresh_sparse = DenseRetriever(), BM25Retriever()
    fresh_hybrid = HybridRetriever(fresh_dense, fresh_sparse, HyDEExpander())
    fresh_hybrid.add_chunks(make_chunks())
    retrievers: dict[str, Retriever] = {"dense": dense, "bm25": sparse, "hybrid": hybrid}

    owned = chunks[0]
    if boundary != "input":
        results = await retrievers[boundary].retrieve("alpha")
        owned = next(result.chunk for result in results if result.chunk.chunk_id == "a")
    change_payload(owned)
    owned.chunk_id = "unindexed-chunk"

    await assert_matches_fresh(dense, fresh_dense)
    await assert_matches_fresh(sparse, fresh_sparse)
    await assert_matches_fresh(hybrid, fresh_hybrid)
    if boundary != "input":
        assert chunks == make_chunks()


@pytest.mark.parametrize("retriever_type", [DenseRetriever, BM25Retriever])
async def test_snapshots_copy_only_indexed_inputs_and_returned_chunks(
    retriever_type: type[DenseRetriever] | type[BM25Retriever],
) -> None:
    chunks = make_chunks()
    retriever = retriever_type()
    with patch.object(
        Chunk, "model_copy", autospec=True, side_effect=Chunk.model_copy
    ) as copy_chunk:
        retriever.add_chunks(chunks)
        assert copy_chunk.call_count == len(chunks)
        assert all(call.kwargs == {"deep": True} for call in copy_chunk.call_args_list)
        copy_chunk.reset_mock()
        retriever.add_chunks([chunks[0]])
        assert copy_chunk.call_count == 1
        copy_chunk.reset_mock()
        retriever.add_chunks([])
        assert copy_chunk.call_count == 0

        for document_ids in (None, ["paper"], ["missing"]):
            for limit in (0, 1, -1, 10):
                copy_chunk.reset_mock()
                results = await retriever.retrieve("alpha", limit=limit, document_ids=document_ids)
                assert copy_chunk.call_count == len(results), (document_ids, limit)
                assert all(call.kwargs == {"deep": True} for call in copy_chunk.call_args_list)
                assert [call.args[0].chunk_id for call in copy_chunk.call_args_list] == [
                    result.chunk.chunk_id for result in results
                ]
