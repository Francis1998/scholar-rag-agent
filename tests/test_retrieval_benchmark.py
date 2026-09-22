"""Test the benchmark's public Python API and actual retriever wiring."""

import asyncio
import json

import httpx
import pytest
from pydantic import ValidationError

from evaluation.benchmark import (
    BenchmarkDataset,
    BenchmarkOptions,
    dataset_fingerprint,
    load_demo_dataset,
    parse_dataset,
    run_benchmark,
)
from retrieval.dense import DenseRetriever
from retrieval.hybrid import HybridRetriever
from retrieval.hyde import HyDEExpander
from retrieval.models import Chunk, SearchResult
from retrieval.sparse import BM25Retriever


def _tiny_dataset() -> BenchmarkDataset:
    return parse_dataset(
        json.dumps(
            {
                "schema_version": 1,
                "name": "tiny",
                "chunks": [
                    {
                        "chunk_id": "apple",
                        "document_id": "fruit",
                        "title": "Fruit",
                        "text": "apple apple fruit",
                        "source": "synthetic:fruit",
                    },
                    {
                        "chunk_id": "ocean",
                        "document_id": "water",
                        "title": "Water",
                        "text": "ocean blue water",
                        "source": "synthetic:water",
                    },
                ],
                "cases": [
                    {"case_id": "a", "query": "apple", "relevant_chunk_ids": ["apple"]},
                    {"case_id": "b", "query": "ocean", "relevant_chunk_ids": ["ocean"]},
                ],
            }
        ).encode("utf-8")
    )


@pytest.mark.parametrize("name", ["bm25", "hybrid"])
async def test_uses_the_existing_retrievers_with_the_recorded_cutoff(name: str) -> None:
    dataset = load_demo_dataset()
    options = BenchmarkOptions.model_validate({"k": 2, "retrievers": (name,)})
    report = await run_benchmark(dataset, options)
    retriever = (
        BM25Retriever()
        if name == "bm25"
        else HybridRetriever(DenseRetriever(), BM25Retriever(), HyDEExpander())
    )
    retriever.add_chunks([Chunk(**chunk.model_dump()) for chunk in dataset.chunks])
    for case, score in zip(dataset.cases, report.retrievers[0].cases, strict=True):
        expected = await retriever.retrieve(case.query, limit=2)
        assert score.retrieved_chunk_ids == [result.chunk.chunk_id for result in expected]
        overlap = set(score.retrieved_chunk_ids) & set(case.relevant_chunk_ids)
        assert score.hit_at_k == float(bool(overlap))
        assert score.recall_at_k == len(overlap) / len(case.relevant_chunk_ids)


async def test_one_passing_retriever_cannot_mask_another_failing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def no_results(
        _self: HybridRetriever, _query: str, limit: int = 10
    ) -> list[SearchResult]:
        return []

    monkeypatch.setattr(HybridRetriever, "retrieve", no_results)
    report = await run_benchmark(
        _tiny_dataset(), BenchmarkOptions(k=1, min_hit_at_k=1.0, min_recall_at_k=1.0)
    )
    assert report.retrievers[0].passed is True
    assert report.retrievers[1].passed is False
    assert report.passed is False
    assert report.retrievers[1].mean_recall_at_k == 0


async def test_identical_queries_reuse_rankings_but_keep_distinct_labels() -> None:
    data = _tiny_dataset().model_dump()
    data["cases"] = [
        {"case_id": "correct", "query": "apple", "relevant_chunk_ids": ["apple"]},
        {"case_id": "incorrect", "query": "apple", "relevant_chunk_ids": ["ocean"]},
    ]
    report = await run_benchmark(
        BenchmarkDataset.model_validate(data), BenchmarkOptions(k=1, retrievers=("bm25",))
    )
    assert report.retrievers[0].mean_hit_at_k == 0.5
    assert [case.hit_at_k for case in report.retrievers[0].cases] == [1.0, 0.0]


async def test_snapshots_inputs_before_await_and_isolates_repeated_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _tiny_dataset()
    options = BenchmarkOptions(k=1, retrievers=("bm25",))
    fingerprint = dataset_fingerprint(dataset)
    started = asyncio.Event()
    release = asyncio.Event()
    original = BM25Retriever.retrieve

    async def pause(self: BM25Retriever, query: str, limit: int = 10) -> list[SearchResult]:
        started.set()
        await release.wait()
        return await original(self, query, limit=limit)

    monkeypatch.setattr(BM25Retriever, "retrieve", pause)
    task = asyncio.create_task(run_benchmark(dataset, options))
    await asyncio.wait_for(started.wait(), timeout=2)
    dataset.chunks.clear()
    dataset.cases.clear()
    options.k = 100
    release.set()
    report = await asyncio.wait_for(task, timeout=2)
    assert report.dataset_sha256 == fingerprint
    assert report.k == 1
    assert report.chunk_count == report.case_count == 2
    assert report.retrievers[0].mean_recall_at_k == 1
    repeated = await run_benchmark(_tiny_dataset(), BenchmarkOptions(k=1, retrievers=("bm25",)))
    assert repeated == report


async def test_no_http_clients_even_with_provider_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbid_client(*_args: object, **_kwargs: object) -> None:
        pytest.fail("benchmark must not construct HTTP clients")

    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "MOONSHOT_API_KEY"):
        monkeypatch.setenv(key, "unused-test-key")
    monkeypatch.setenv("SCHOLAR_RAG_MAX_HOPS", "invalid-but-irrelevant")
    monkeypatch.setattr(httpx.Client, "__init__", forbid_client)
    monkeypatch.setattr(httpx.AsyncClient, "__init__", forbid_client)
    report = await run_benchmark(load_demo_dataset())
    assert report.passed


async def test_retrieval_errors_propagate_without_success_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail(_self: BM25Retriever, _query: str, limit: int = 10) -> list[SearchResult]:
        raise RuntimeError("synthetic retrieval failure")

    monkeypatch.setattr(BM25Retriever, "retrieve", fail)
    with pytest.raises(RuntimeError, match="synthetic retrieval failure"):
        await run_benchmark(_tiny_dataset())


@pytest.mark.parametrize(
    "options",
    [
        {"k": True},
        {"k": 1.0},
        {"k": "1"},
        {"k": 0},
        {"k": 101},
        {"retrievers": ()},
        {"retrievers": ("bm25", "bm25")},
        {"retrievers": ("unknown",)},
        {"min_hit_at_k": float("nan")},
        {"min_recall_at_k": float("inf")},
        {"min_recall_at_k": True},
    ],
)
def test_python_options_have_the_same_strict_bounds(options: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        BenchmarkOptions.model_validate(options)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("chunk_id", " apple"),
        ("document_id", "fruit "),
        ("text", ""),
        ("title", " "),
        ("source", 42),
        ("text", "x" * 32769),
    ],
    ids=["padded-chunk", "padded-document", "empty-text", "blank-title", "bad-source", "long-text"],
)
def test_chunk_validation_is_strict_and_bounded(field: str, value: object) -> None:
    data = _tiny_dataset().model_dump()
    data["chunks"][0][field] = value
    with pytest.raises(ValidationError):
        BenchmarkDataset.model_validate(data)


def test_non_ascii_ids_and_text_survive_validation_and_fingerprinting() -> None:
    data = _tiny_dataset().model_dump()
    label = "paper-\u00e9"
    data["chunks"][0]["chunk_id"] = label
    data["chunks"][0]["text"] = "caf\u00e9 evidence"
    data["cases"][0]["relevant_chunk_ids"] = [label]
    dataset = parse_dataset(json.dumps(data, ensure_ascii=False).encode("utf-8"))
    assert dataset.chunks[0].chunk_id == label
    assert dataset.chunks[0].text == "caf\u00e9 evidence"
    assert len(dataset_fingerprint(dataset)) == 64
