"""Bounded, model-free retrieval benchmarks over explicitly labeled chunks."""

from __future__ import annotations

import hashlib
import json
from functools import partial
from importlib.resources import files
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from evaluation.harness import EvalCase, EvaluationHarness
from retrieval.dense import DenseRetriever
from retrieval.hybrid import HybridRetriever
from retrieval.hyde import HyDEExpander
from retrieval.models import Chunk, SearchResult
from retrieval.sparse import BM25Retriever

MAX_DATASET_BYTES = 8 * 1024 * 1024
Identifier = Annotated[str, StringConstraints(min_length=1, max_length=128)]
RetrieverName = Literal["bm25", "hybrid"]


class BenchmarkModel(BaseModel):
    """Reject ambiguous fields/types and avoid echoing source text in validation errors."""

    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)


class BenchmarkChunk(BenchmarkModel):
    """An already-chunked passage; its supplied ID is the evaluation unit."""

    chunk_id: Identifier
    document_id: Identifier
    title: str = Field(min_length=1, max_length=512)
    text: str = Field(min_length=1, max_length=32768)
    source: str = Field(min_length=1, max_length=1024)

    @field_validator("chunk_id", "document_id", "title", "text", "source")
    @classmethod
    def nonblank(cls, value: str) -> str:
        """Keep supplied content intact, but reject blank passages and identifiers."""
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("chunk_id", "document_id")
    @classmethod
    def unpadded_identifier(cls, value: str) -> str:
        """Require exact, unambiguous labels rather than silently trimming IDs."""
        if value != value.strip():
            raise ValueError("IDs must not have surrounding whitespace")
        return value


class BenchmarkCase(BenchmarkModel):
    """A query with binary relevance judgments against this dataset's chunk IDs."""

    case_id: Identifier
    query: str = Field(min_length=1, max_length=4096)
    relevant_chunk_ids: list[Identifier] = Field(min_length=1, max_length=10000)

    @field_validator("case_id", "query")
    @classmethod
    def nonblank(cls, value: str) -> str:
        """Reject empty evaluation questions and labels."""
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("case_id")
    @classmethod
    def unpadded_identifier(cls, value: str) -> str:
        """Reject padded case labels."""
        if value != value.strip():
            raise ValueError("IDs must not have surrounding whitespace")
        return value

    @field_validator("relevant_chunk_ids")
    @classmethod
    def unique_relevance(cls, value: list[str]) -> list[str]:
        """Do not silently deduplicate or normalize relevance judgments."""
        if any(not item.strip() or item != item.strip() for item in value):
            raise ValueError("relevant chunk IDs must be nonblank and unpadded")
        if len(set(value)) != len(value):
            raise ValueError("relevant chunk IDs must be unique")
        return value


class BenchmarkDataset(BenchmarkModel):
    """Versioned, bounded local dataset; no URLs are fetched or code executed."""

    schema_version: int = Field(ge=1, le=1)
    name: str = Field(min_length=1, max_length=128)
    chunks: list[BenchmarkChunk] = Field(min_length=1, max_length=10000)
    cases: list[BenchmarkCase] = Field(min_length=1, max_length=1000)

    @field_validator("name")
    @classmethod
    def nonblank_name(cls, value: str) -> str:
        """Reject unnamed datasets."""
        if not value.strip():
            raise ValueError("dataset name must not be blank")
        return value

    @model_validator(mode="after")
    def check_labels(self) -> Self:
        """Fail closed on duplicate labels or gold IDs absent from the corpus."""
        chunk_ids = {chunk.chunk_id for chunk in self.chunks}
        if len(chunk_ids) != len(self.chunks):
            raise ValueError("chunk IDs must be unique")
        if len({case.case_id for case in self.cases}) != len(self.cases):
            raise ValueError("case IDs must be unique")
        if any(not set(case.relevant_chunk_ids) <= chunk_ids for case in self.cases):
            raise ValueError("every relevant chunk ID must exist in the dataset")
        return self


class BenchmarkOptions(BenchmarkModel):
    """Validated options shared by Python and command-line callers."""

    k: int = Field(default=5, ge=1, le=100)
    retrievers: tuple[RetrieverName, ...] = ("bm25", "hybrid")
    min_hit_at_k: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    min_recall_at_k: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)

    @field_validator("retrievers")
    @classmethod
    def nonempty_unique_retrievers(
        cls, value: tuple[RetrieverName, ...]
    ) -> tuple[RetrieverName, ...]:
        """Reject empty runs and repeated configurations."""
        if not value or len(set(value)) != len(value):
            raise ValueError("retrievers must be a nonempty, unique selection")
        return value


class BenchmarkCaseScore(BenchmarkModel):
    """Retrieval-only results without source text, queries, or generated answers."""

    case_id: str
    hit_at_k: float
    recall_at_k: float
    retrieved_chunk_ids: list[str]


class RetrieverScore(BenchmarkModel):
    """One real retriever configuration and its independently applied thresholds."""

    name: RetrieverName
    configuration: str
    mean_hit_at_k: float
    mean_recall_at_k: float
    passed: bool
    cases: list[BenchmarkCaseScore]


class BenchmarkReport(BenchmarkModel):
    """Deterministic, versioned benchmark artifact suitable for a local CI gate."""

    schema_version: int = 1
    dataset_name: str
    dataset_sha256: str
    chunk_count: int
    case_count: int
    k: int
    min_hit_at_k: float | None
    min_recall_at_k: float | None
    passed: bool
    retrievers: list[RetrieverScore]


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("dataset JSON must not contain duplicate object keys")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ValueError(f"dataset JSON cannot contain {value}")


def parse_dataset(data: bytes) -> BenchmarkDataset:
    """Validate UTF-8 JSON, file size, schema, and complete relevance judgments."""
    if len(data) > MAX_DATASET_BYTES:
        raise ValueError(f"dataset exceeds the {MAX_DATASET_BYTES}-byte limit")
    parsed = json.loads(
        data.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_reject_constant
    )
    return BenchmarkDataset.model_validate(parsed)


def load_dataset(path: Path) -> BenchmarkDataset:
    """Read at most the accepted file size plus one byte, without changing the input."""
    with path.open("rb") as source:
        return parse_dataset(source.read(MAX_DATASET_BYTES + 1))


def load_demo_dataset() -> BenchmarkDataset:
    """Load the packaged, explicitly synthetic fixture, including from a wheel."""
    return parse_dataset(files("evaluation").joinpath("data/retrieval_benchmark.json").read_bytes())


def dataset_fingerprint(dataset: BenchmarkDataset) -> str:
    """Hash canonical validated JSON; list order is retained because it breaks ranking ties."""
    canonical = json.dumps(
        dataset.model_dump(mode="json"), sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _cached_results(
    ranked: dict[str, list[SearchResult]], query: str, _limit: int
) -> list[SearchResult]:
    return ranked[query]


async def run_benchmark(
    dataset: BenchmarkDataset, options: BenchmarkOptions | None = None
) -> BenchmarkReport:
    """Benchmark fresh BM25/default-hybrid indexes, never generation or the full agent."""
    dataset = BenchmarkDataset.model_validate(dataset.model_dump())
    selected = BenchmarkOptions.model_validate((options or BenchmarkOptions()).model_dump())
    chunks = [Chunk(**chunk.model_dump()) for chunk in dataset.chunks]
    harness = EvaluationHarness(
        [
            EvalCase(
                case_id=case.case_id,
                query=case.query,
                relevant_chunk_ids=frozenset(case.relevant_chunk_ids),
            )
            for case in dataset.cases
        ]
    )
    scores: list[RetrieverScore] = []
    for name in selected.retrievers:
        retriever: BM25Retriever | HybridRetriever
        if name == "bm25":
            retriever = BM25Retriever()
            configuration = "BM25 k1=1.5 b=0.75; supplied queries; supplied corpus order"
        else:
            retriever = HybridRetriever(DenseRetriever(), BM25Retriever(), HyDEExpander())
            configuration = (
                "64-dimensional lexical hash vectors + BM25 + deterministic HyDE template + RRF; "
                "no graph expansion, reranker, learned embeddings, or LLM"
            )
        retriever.add_chunks(chunks)
        ranked: dict[str, list[SearchResult]] = {}
        for case in dataset.cases:
            if case.query not in ranked:
                ranked[case.query] = await retriever.retrieve(case.query, limit=selected.k)
        report = harness.evaluate(partial(_cached_results, ranked), k=selected.k)
        passed = (
            selected.min_hit_at_k is None or report.mean_hit_at_k >= selected.min_hit_at_k
        ) and (
            selected.min_recall_at_k is None or report.mean_recall_at_k >= selected.min_recall_at_k
        )
        scores.append(
            RetrieverScore(
                name=name,
                configuration=configuration,
                mean_hit_at_k=report.mean_hit_at_k,
                mean_recall_at_k=report.mean_recall_at_k,
                passed=passed,
                cases=[
                    BenchmarkCaseScore(
                        case_id=case.case_id,
                        hit_at_k=case.hit_at_k,
                        recall_at_k=case.recall_at_k,
                        retrieved_chunk_ids=list(case.retrieved_chunk_ids),
                    )
                    for case in report.cases
                ],
            )
        )
    return BenchmarkReport(
        dataset_name=dataset.name,
        dataset_sha256=dataset_fingerprint(dataset),
        chunk_count=len(chunks),
        case_count=len(dataset.cases),
        k=selected.k,
        min_hit_at_k=selected.min_hit_at_k,
        min_recall_at_k=selected.min_recall_at_k,
        passed=all(score.passed for score in scores),
        retrievers=scores,
    )
