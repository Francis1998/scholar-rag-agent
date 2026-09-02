"""Tests for keyword-match gate postprocessing."""

import pytest

from retrieval.keyword_match_gate import KeywordMatchGate
from retrieval.models import Chunk, SearchResult


def _result(chunk_id: str, score: float, text: str = "") -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id=f"doc-{chunk_id}",
            title=chunk_id,
            text=text,
            source="test",
            metadata={},
        ),
        score=score,
        retriever="bm25",
        path=["hybrid"],
    )


@pytest.mark.parametrize("min_coverage", [-0.1, 1.1, float("nan")])
def test_rejects_invalid_min_coverage(min_coverage: float) -> None:
    with pytest.raises(ValueError, match="min_coverage"):
        KeywordMatchGate(min_coverage=min_coverage)


def test_keeps_sufficient_coverage() -> None:
    results = [
        _result("full", 1.0, "alpha beta gamma"),
        _result("half", 0.9, "alpha only"),
        _result("none", 0.8, "unrelated"),
    ]
    kept = KeywordMatchGate(min_coverage=0.5).gate(results, query="alpha beta")
    assert [item.chunk.chunk_id for item in kept] == ["full", "half"]
    assert kept[0].retriever == "keyword_match_gate"
    assert kept[0].score == 1.0
    assert kept[1].score == 0.5
    assert kept[0].path == ["hybrid", "bm25"]


def test_empty_query_keeps_all_until_top_k() -> None:
    results = [_result("a", 1.0, "x"), _result("b", 0.9, "y")]
    kept = KeywordMatchGate(min_coverage=1.0).gate(results, query="")
    assert len(kept) == 2


def test_empty_and_non_positive_top_k() -> None:
    gate = KeywordMatchGate()
    assert gate.gate([], query="x") == []
    assert gate.gate([_result("a", 1.0, "x")], query="x", top_k=0) == []


def test_does_not_mutate_inputs() -> None:
    original = [_result("a", 1.0, "alpha")]
    path_before = list(original[0].path)
    KeywordMatchGate(min_coverage=0.0).gate(original, query="alpha")
    assert original[0].path == path_before


def test_docstring_mentions_frontier_models() -> None:
    assert "Claude Sonnet 4.6" in (KeywordMatchGate.__doc__ or "")
