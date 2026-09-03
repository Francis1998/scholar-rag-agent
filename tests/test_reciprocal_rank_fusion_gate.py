"""Tests for ReciprocalRankFusionGate postprocessor."""

import pytest

from retrieval.models import Chunk, SearchResult
from retrieval.reciprocal_rank_fusion_gate import ReciprocalRankFusionGate


def _result(chunk_id: str, score: float = 1.0, retriever: str = "bm25") -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id=f"doc-{chunk_id}",
            title=chunk_id,
            text="",
            source="test",
            metadata={},
        ),
        score=score,
        retriever=retriever,
        path=["hybrid"],
    )


def test_rejects_non_positive_k() -> None:
    with pytest.raises(ValueError, match="k must be a positive integer"):
        ReciprocalRankFusionGate(k=0)


def test_rejects_non_integer_k() -> None:
    with pytest.raises(ValueError, match="k must be a positive integer"):
        ReciprocalRankFusionGate(k=1.5)  # type: ignore[arg-type]


def test_empty_result_sets() -> None:
    assert ReciprocalRankFusionGate().gate([]) == []


def test_single_list_preserves_order() -> None:
    results = [_result("a"), _result("b"), _result("c")]
    fused = ReciprocalRankFusionGate(k=60).gate([results])
    ids = [r.chunk.chunk_id for r in fused]
    assert ids == ["a", "b", "c"]
    assert fused[0].retriever == "reciprocal_rank_fusion_gate"


def test_shared_hits_rank_higher() -> None:
    list1 = [_result("a", retriever="dense"), _result("b", retriever="dense")]
    list2 = [_result("b", retriever="sparse"), _result("c", retriever="sparse")]
    fused = ReciprocalRankFusionGate(k=60).gate([list1, list2])
    assert fused[0].chunk.chunk_id == "b"


def test_rrf_score_formula() -> None:
    k = 60
    list1 = [_result("x", retriever="r1")]
    list2 = [_result("y", retriever="r2"), _result("x", retriever="r2")]
    fused = ReciprocalRankFusionGate(k=k).gate([list1, list2])
    scores = {r.chunk.chunk_id: r.score for r in fused}
    expected_x = 1.0 / (k + 1) + 1.0 / (k + 2)
    expected_y = 1.0 / (k + 1)
    assert abs(scores["x"] - expected_x) < 1e-12
    assert abs(scores["y"] - expected_y) < 1e-12


def test_top_k_limits_output() -> None:
    results = [_result("a"), _result("b"), _result("c")]
    fused = ReciprocalRankFusionGate().gate([results], top_k=2)
    assert len(fused) == 2


def test_top_k_zero_returns_empty() -> None:
    fused = ReciprocalRankFusionGate().gate([[_result("a")]], top_k=0)
    assert fused == []


def test_does_not_mutate_inputs() -> None:
    original = [_result("a")]
    path_before = list(original[0].path)
    ReciprocalRankFusionGate().gate([original])
    assert original[0].path == path_before
    assert original[0].retriever == "bm25"


def test_docstring_mentions_frontier_models() -> None:
    assert "GPT-5.5" in (ReciprocalRankFusionGate.__doc__ or "")
