"""Tests for score-threshold gate postprocessing."""

import pytest

from retrieval.models import Chunk, SearchResult
from retrieval.score_threshold_gate import ScoreThresholdGate


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


def test_rejects_non_finite_min_score() -> None:
    with pytest.raises(ValueError, match="min_score"):
        ScoreThresholdGate(min_score=float("nan"))


def test_drops_below_threshold() -> None:
    kept = ScoreThresholdGate(min_score=0.5).gate(
        [_result("a", 0.4), _result("b", 0.6), _result("c", 0.5)]
    )
    assert [item.chunk.chunk_id for item in kept] == ["b", "c"]
    assert kept[0].retriever == "score_threshold_gate"
    assert kept[0].path == ["hybrid", "bm25"]


def test_empty_and_non_positive_top_k() -> None:
    gate = ScoreThresholdGate(min_score=0.0)
    assert gate.gate([]) == []
    assert gate.gate([_result("a", 1.0)], top_k=0) == []


def test_top_k_limits() -> None:
    hits = [_result("a", 0.9), _result("b", 0.8), _result("c", 0.7)]
    assert len(ScoreThresholdGate(min_score=0.0).gate(hits, top_k=2)) == 2


def test_does_not_mutate_inputs() -> None:
    original = [_result("a", 0.9)]
    path_before = list(original[0].path)
    ScoreThresholdGate(min_score=0.5).gate(original)
    assert original[0].path == path_before
    assert original[0].retriever == "bm25"


def test_docstring_mentions_frontier_models() -> None:
    assert "GPT-5.5" in (ScoreThresholdGate.__doc__ or "")
