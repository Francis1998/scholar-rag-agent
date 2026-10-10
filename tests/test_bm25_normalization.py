"""BM25 length normalization must use the true positive mean of indexed chunks."""

import math
from pathlib import Path

import httpx
import pytest

from ingestion.chunking import TextChunker
from ingestion.pipeline import IngestionPipeline
from retrieval.dense import DenseRetriever
from retrieval.graph import GraphRAGBuilder
from retrieval.hybrid import HybridRetriever
from retrieval.hyde import HyDEExpander
from retrieval.models import Chunk, Document
from retrieval.sparse import BM25Retriever
from storage.document_store import SQLiteDocumentStore
from storage.graph_store import SQLiteGraphStore
from tests.test_graph_reindexing import FixtureExtractor, deny_network


def make_chunk(chunk_id: str, text: str, document_id: str = "selected") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        title="Synthetic normalization fixture",
        text=text,
        source="synthetic:bm25-normalization",
    )


def reference_score(
    *,
    term_frequency: int,
    document_length: int,
    corpus_size: int,
    total_length: int,
    document_frequency: int = 1,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """Calculate one matched term's BM25 score from explicit, independent fixture counts."""
    average_length = total_length / corpus_size
    idf = math.log((corpus_size + 1) / (document_frequency + 0.5))
    return (
        idf * (k1 + 1) / (1 + k1 / term_frequency * (1 - b + b * document_length / average_length))
    )


async def test_one_token_and_one_empty_chunk_use_half_token_average() -> None:
    retriever = BM25Retriever()
    retriever.add_chunks([make_chunk("a", "alpha"), make_chunk("empty", "")])

    results = await retriever.retrieve("alpha")

    assert [result.chunk.chunk_id for result in results] == ["a", "empty"]
    assert results[0].score == pytest.approx(0.47803253831720366)
    assert results[1].score == 0.0


@pytest.mark.parametrize("empty_text", ["", " \t\n ", ".,;:()[]{}!?\"'"])
@pytest.mark.parametrize(
    ("text", "term_frequency", "document_length", "empty_count"),
    [
        ("alpha", 1, 1, 1),
        ("alpha", 1, 1, 3),
        ("alpha beta", 1, 2, 2),
        ("alpha alpha beta", 2, 3, 3),
    ],
    ids=["half", "quarter", "two-thirds", "three-quarters"],
)
@pytest.mark.parametrize(("k1", "b"), [(1.5, 0.75), (0.6, 0.2), (1.5, 0.0)])
async def test_fractional_average_matches_reference(
    empty_text: str,
    text: str,
    term_frequency: int,
    document_length: int,
    empty_count: int,
    k1: float,
    b: float,
) -> None:
    retriever = BM25Retriever(k1=k1, b=b)
    empty_chunks = [make_chunk(f"empty-{index}", empty_text) for index in range(empty_count)]
    retriever.add_chunks([*reversed(empty_chunks), make_chunk("match", text)])

    results = await retriever.retrieve("ALPHA,")

    assert [result.chunk.chunk_id for result in results] == [
        "match",
        *sorted(chunk.chunk_id for chunk in empty_chunks),
    ]
    assert results[0].score == pytest.approx(
        reference_score(
            term_frequency=term_frequency,
            document_length=document_length,
            corpus_size=empty_count + 1,
            total_length=document_length,
            k1=k1,
            b=b,
        )
    )
    assert all(result.score == 0.0 for result in results[1:])


@pytest.mark.parametrize(
    ("text", "term_frequency", "total_length"),
    [("alpha", 1, 2), ("alpha alpha beta", 2, 4)],
    ids=["average-one", "average-two"],
)
async def test_common_average_at_least_one_matches_reference(
    text: str, term_frequency: int, total_length: int
) -> None:
    retriever = BM25Retriever()
    retriever.add_chunks([make_chunk("match", text), make_chunk("other", "beta")])

    results = await retriever.retrieve("alpha")

    assert results[0].chunk.chunk_id == "match"
    assert results[0].score == pytest.approx(
        reference_score(
            term_frequency=term_frequency,
            document_length=total_length - 1,
            corpus_size=2,
            total_length=total_length,
        )
    )
    assert results[1].score == 0.0


@pytest.mark.parametrize("texts", [(), ("", " \t\n ", "!!!")], ids=["empty-index", "all-empty"])
@pytest.mark.parametrize("query", ["alpha", "", " \t\n ", "?!"])
async def test_empty_corpus_returns_only_finite_zero_scores(
    texts: tuple[str, ...], query: str
) -> None:
    retriever = BM25Retriever()
    chunks = [make_chunk(str(index), text) for index, text in enumerate(texts)]
    retriever.add_chunks(list(reversed(chunks)))

    results = await retriever.retrieve(query)

    assert [result.chunk for result in results] == chunks
    assert all(math.isfinite(result.score) and result.score == 0.0 for result in results)


@pytest.mark.parametrize("query", ["missing", "", " \t\n ", "?!"])
async def test_no_matching_query_returns_finite_zeros_in_id_order(query: str) -> None:
    retriever = BM25Retriever()
    retriever.add_chunks([make_chunk("c", "alpha"), make_chunk("b", "!!!"), make_chunk("a", "")])

    results = await retriever.retrieve(query)

    assert [result.chunk.chunk_id for result in results] == ["a", "b", "c"]
    assert all(math.isfinite(result.score) and result.score == 0.0 for result in results)


async def test_document_scope_preserves_global_fractional_statistics() -> None:
    selected = make_chunk("b", "alpha")
    retriever = BM25Retriever()
    retriever.add_chunks(
        [
            selected,
            make_chunk("a", "alpha", "excluded"),
            *[make_chunk(key, "", "excluded") for key in ("f", "e", "d", "c")],
        ]
    )

    all_results = await retriever.retrieve("alpha")
    [scoped] = await retriever.retrieve("alpha", limit=1, document_ids=["selected"])

    assert [result.chunk.chunk_id for result in all_results[:2]] == ["a", "b"]
    assert scoped == all_results[1]
    assert scoped.chunk == selected
    assert scoped.score == pytest.approx(
        reference_score(
            term_frequency=1,
            document_length=1,
            corpus_size=6,
            total_length=2,
            document_frequency=2,
        )
    )
    assert await retriever.retrieve("alpha", document_ids=["missing"]) == []


@pytest.mark.parametrize("same_batch", [False, True])
@pytest.mark.parametrize(
    ("original_text", "replacement_text", "total_length"),
    [
        ("alpha beta gamma delta", "alpha", 1),
        ("alpha", "", 0),
        ("", "alpha", 1),
        ("alpha", "alpha beta gamma delta", 4),
    ],
    ids=["one-to-quarter", "quarter-to-zero", "zero-to-quarter", "quarter-to-one"],
)
async def test_upsert_across_normalization_boundary_matches_fresh_index(
    same_batch: bool, original_text: str, replacement_text: str, total_length: int
) -> None:
    empty_chunks = [make_chunk(key, text) for key, text in [("a", ""), ("b", " "), ("c", "!!!")]]
    original = make_chunk("match", original_text)
    replacement = make_chunk("match", replacement_text)
    retriever = BM25Retriever()
    if same_batch:
        retriever.add_chunks([original, *empty_chunks, replacement, replacement])
    else:
        retriever.add_chunks([original, *empty_chunks])
        retriever.add_chunks([replacement])
        retriever.add_chunks([replacement])
        retriever.add_chunks([])
    fresh = BM25Retriever()
    fresh.add_chunks([*empty_chunks, replacement])

    results = await retriever.retrieve("alpha")

    assert results == await fresh.retrieve("alpha")
    assert len(results) == 4
    expected = (
        reference_score(
            term_frequency=1,
            document_length=total_length,
            corpus_size=4,
            total_length=total_length,
        )
        if total_length
        else 0.0
    )
    assert next(result.score for result in results if result.chunk.chunk_id == "match") == (
        pytest.approx(expected)
    )


async def test_small_window_ingestion_replacement_and_restart_preserve_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", deny_network)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", deny_network)
    database = tmp_path / "normalization.sqlite3"
    store = SQLiteDocumentStore(database)
    sparse = BM25Retriever(k1=0.6, b=0.2)
    hybrid = HybridRetriever(DenseRetriever(), sparse, HyDEExpander())
    pipeline = IngestionPipeline(
        store,
        hybrid,
        GraphRAGBuilder(SQLiteGraphStore(database), extractor=FixtureExtractor()),
        TextChunker(chunk_size=1, overlap=0),
    )
    document = Document(
        document_id="paper", title="Synthetic", text="!", source="synthetic:normalization"
    )
    retained = pipeline.ingest_documents([document.model_copy(update={"document_id": "other"})])

    for text, expected_texts, total_length in [
        ("a b", ["a", " ", "b"], 2),
        ("a !", ["a", " ", "!"], 1),
        ("!", ["!"], 0),
        ("a !", ["a", " ", "!"], 1),
    ]:
        revision = document.model_copy(update={"text": text})
        chunks = pipeline.ingest_documents([revision])
        assert [chunk.text for chunk in chunks] == expected_texts
        assert pipeline.ingest_documents([revision]) == chunks
        persisted = SQLiteDocumentStore(database).list_chunks()
        assert persisted == sorted([*retained, *chunks], key=lambda chunk: chunk.chunk_id)
        restarted_sparse = BM25Retriever(k1=0.6, b=0.2)
        restarted_hybrid = HybridRetriever(DenseRetriever(), restarted_sparse, HyDEExpander())
        restarted_hybrid.add_chunks(persisted)

        results = await sparse.retrieve("a")
        assert results == await restarted_sparse.retrieve("a")
        expected = (
            reference_score(
                term_frequency=1,
                document_length=1,
                corpus_size=len(expected_texts) + 1,
                total_length=total_length,
                k1=0.6,
                b=0.2,
            )
            if total_length
            else 0.0
        )
        for result in results:
            assert math.isfinite(result.score)
            assert result.score == pytest.approx(expected if result.chunk.text == "a" else 0.0)
        assert await hybrid.retrieve("a") == await restarted_hybrid.retrieve("a")
