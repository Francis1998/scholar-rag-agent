"""Tests for term coverage booster postprocessing."""

import pytest

from retrieval.models import Chunk, SearchResult
from retrieval.term_coverage_boost import TermCoverageBooster


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


@pytest.mark.parametrize("alpha", [-0.1, 1.1, float("nan")])
def test_rejects_invalid_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        TermCoverageBooster(alpha=alpha)


def test_boosts_higher_coverage_above_higher_prior() -> None:
    results = [
        _result("low_cov", 1.0, "transformer only"),
        _result("high_cov", 0.5, "transformer attention language models"),
    ]
    ordered = TermCoverageBooster(alpha=0.8).boost(
        results, query="transformer attention language models"
    )
    assert ordered[0].chunk.chunk_id == "high_cov"
    assert ordered[0].retriever == "term_coverage_boost"


def test_alpha_zero_preserves_relevance_order() -> None:
    results = [
        _result("low", 0.2, "full coverage transformer attention language"),
        _result("high", 0.9, "unrelated gardening soil"),
    ]
    ordered = TermCoverageBooster(alpha=0.0).boost(results, query="transformer attention language")
    assert [item.chunk.chunk_id for item in ordered] == ["high", "low"]


def test_does_not_mutate_inputs() -> None:
    original = _result("a", 0.9, "one two three")
    snapshot = original.score
    TermCoverageBooster().boost([original], query="one")
    assert original.score == snapshot
    assert original.retriever == "bm25"


def test_empty_and_top_k() -> None:
    booster = TermCoverageBooster()
    assert booster.boost([], query="q") == []
    assert booster.boost([_result("a", 0.5, "hello")], query="hello", top_k=0) == []
    rows = [
        _result("a", 1.0, "cats dogs"),
        _result("b", 0.9, "cats birds"),
        _result("c", 0.8, "cats fish"),
    ]
    assert len(booster.boost(rows, query="cats", top_k=2)) == 2


def test_empty_query_coverage_is_one() -> None:
    result = _result("a", 0.4, "anything")
    ordered = TermCoverageBooster(alpha=0.5).boost([result], query="")
    # (1-0.5)*0.4 + 0.5*1.0 = 0.7
    assert ordered[0].score == pytest.approx(0.7)


def test_path_rewritten() -> None:
    result = _result("a", 0.5, "hello world")
    ordered = TermCoverageBooster(alpha=0.0).boost([result], query="hello")
    assert ordered[0].path == ["hybrid", "bm25"]


def test_full_coverage_formula() -> None:
    result = _result("a", 0.0, "alpha beta")
    ordered = TermCoverageBooster(alpha=1.0).boost([result], query="alpha beta")
    assert ordered[0].score == pytest.approx(1.0)
