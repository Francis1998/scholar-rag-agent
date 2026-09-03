"""Tests for DiversityCapGate postprocessor."""

import pytest

from retrieval.diversity_cap_gate import DiversityCapGate
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


def test_rejects_non_positive_max_per_source() -> None:
    with pytest.raises(ValueError, match="max_per_source"):
        DiversityCapGate(max_per_source=0)


def test_rejects_non_integer_max_per_source() -> None:
    with pytest.raises(ValueError, match="max_per_source"):
        DiversityCapGate(max_per_source=2.5)  # type: ignore[arg-type]


def test_empty_results() -> None:
    assert DiversityCapGate().gate([]) == []


def test_caps_single_source() -> None:
    hits = [_result("a", "doc-1"), _result("b", "doc-1"), _result("c", "doc-1")]
    kept = DiversityCapGate(max_per_source=2).gate(hits)
    assert len(kept) == 2
    assert [r.chunk.chunk_id for r in kept] == ["a", "b"]


def test_diverse_sources_unaffected() -> None:
    hits = [_result("a", "doc-1"), _result("b", "doc-2"), _result("c", "doc-3")]
    kept = DiversityCapGate(max_per_source=1).gate(hits)
    assert len(kept) == 3


def test_provenance_rewritten() -> None:
    kept = DiversityCapGate().gate([_result("a")])
    assert kept[0].retriever == "diversity_cap_gate"
    assert kept[0].path == ["hybrid", "bm25"]


def test_top_k_limits_output() -> None:
    hits = [_result("a", "doc-1"), _result("b", "doc-2"), _result("c", "doc-3")]
    kept = DiversityCapGate().gate(hits, top_k=2)
    assert len(kept) == 2


def test_top_k_zero_returns_empty() -> None:
    assert DiversityCapGate().gate([_result("a")], top_k=0) == []


def test_does_not_mutate_inputs() -> None:
    original = [_result("a")]
    path_before = list(original[0].path)
    DiversityCapGate().gate(original)
    assert original[0].path == path_before
    assert original[0].retriever == "bm25"


def test_docstring_mentions_frontier_models() -> None:
    assert "GPT-5.5" in (DiversityCapGate.__doc__ or "")
