"""Tests for recency half-life boost postprocessing."""

import pytest

from retrieval.models import Chunk, SearchResult
from retrieval.recency_half_life import RecencyHalfLifeBooster


def _result(chunk_id: str, score: float, year: str = "") -> SearchResult:
    metadata = {"year": year} if year else {}
    return SearchResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id=f"doc-{chunk_id}",
            title=chunk_id,
            text=f"text for {chunk_id}",
            source="test",
            metadata=metadata,
        ),
        score=score,
        retriever="bm25",
        path=["hybrid"],
    )


@pytest.mark.parametrize("alpha", [-0.1, 1.1, float("nan")])
def test_rejects_invalid_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        RecencyHalfLifeBooster(alpha=alpha)


def test_rejects_invalid_half_life() -> None:
    with pytest.raises(ValueError, match="half_life_years"):
        RecencyHalfLifeBooster(half_life_years=0.0)


def test_prefers_recent_publication() -> None:
    recent = _result("recent", score=0.0, year="2026")
    old = _result("old", score=1.0, year="2016")
    boosted = RecencyHalfLifeBooster(alpha=1.0, ref_year=2026, half_life_years=5.0).boost(
        [old, recent], query="ignored"
    )
    assert [item.chunk.chunk_id for item in boosted] == ["recent", "old"]
    assert boosted[0].retriever == "recency_half_life"
    assert old.score == 1.0


def test_missing_year_yields_zero_recency() -> None:
    result = _result("only", score=0.8)
    boosted = RecencyHalfLifeBooster(alpha=1.0).boost([result], query="q")
    assert boosted[0].score == pytest.approx(0.0)


def test_blend_formula() -> None:
    result = _result("only", score=0.4, year="2026")
    boosted = RecencyHalfLifeBooster(alpha=0.5, ref_year=2026).boost([result], query="q")
    assert boosted[0].score == pytest.approx(0.7)


def test_future_year_is_clamped_to_one() -> None:
    result = _result("future", score=0.0, year="2030")
    boosted = RecencyHalfLifeBooster(alpha=1.0, ref_year=2026).boost([result], query="q")
    assert boosted[0].score == pytest.approx(1.0)


def test_top_k_and_empty() -> None:
    booster = RecencyHalfLifeBooster(alpha=1.0, ref_year=2026)
    assert booster.boost([], query="q") == []
    rows = [_result("a", 0.0, year="2026"), _result("b", 0.0, year="2000")]
    assert len(booster.boost(rows, query="q", top_k=1)) == 1
