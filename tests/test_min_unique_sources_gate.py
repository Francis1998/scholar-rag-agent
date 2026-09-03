"""Tests for MinUniqueSourcesGate postprocessor."""

import pytest

from retrieval.min_unique_sources_gate import MinUniqueSourcesGate
from retrieval.models import Chunk, SearchResult


def _result(chunk_id: str, doc_id: str = "doc-a", score: float = 1.0) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id=doc_id,
            title=chunk_id,
            text="",
            source="test",
            metadata={},
        ),
        score=score,
        retriever="bm25",
        path=["hybrid"],
    )


def test_rejects_non_positive_min_sources() -> None:
    with pytest.raises(ValueError, match="min_sources"):
        MinUniqueSourcesGate(min_sources=0)


def test_rejects_non_integer_min_sources() -> None:
    with pytest.raises(ValueError, match="min_sources"):
        MinUniqueSourcesGate(min_sources=1.5)  # type: ignore[arg-type]


def test_empty_results() -> None:
    assert MinUniqueSourcesGate().gate([]) == []


def test_rejects_insufficient_sources() -> None:
    hits = [_result("a", "doc-1"), _result("b", "doc-1")]
    assert MinUniqueSourcesGate(min_sources=2).gate(hits) == []


def test_passes_sufficient_sources() -> None:
    hits = [_result("a", "doc-1"), _result("b", "doc-2")]
    kept = MinUniqueSourcesGate(min_sources=2).gate(hits)
    assert len(kept) == 2
    assert kept[0].retriever == "min_unique_sources_gate"
    assert kept[0].path == ["hybrid", "bm25"]


def test_top_k_limits_output() -> None:
    hits = [_result("a", "doc-1"), _result("b", "doc-2"), _result("c", "doc-3")]
    kept = MinUniqueSourcesGate(min_sources=2).gate(hits, top_k=2)
    assert len(kept) == 2


def test_top_k_zero_returns_empty() -> None:
    hits = [_result("a", "doc-1"), _result("b", "doc-2")]
    assert MinUniqueSourcesGate(min_sources=1).gate(hits, top_k=0) == []


def test_does_not_mutate_inputs() -> None:
    original = [_result("a", "doc-1"), _result("b", "doc-2")]
    path_before = list(original[0].path)
    MinUniqueSourcesGate(min_sources=2).gate(original)
    assert original[0].path == path_before
    assert original[0].retriever == "bm25"


def test_docstring_mentions_frontier_models() -> None:
    assert "GPT-5.5" in (MinUniqueSourcesGate.__doc__ or "")
