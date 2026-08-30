"""Tests for claim density boost postprocessing."""

import pytest

from retrieval.claim_density_boost import ClaimDensityBooster
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


@pytest.mark.parametrize("alpha", [-0.1, 1.1, float("nan")])
def test_rejects_invalid_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        ClaimDensityBooster(alpha=alpha)


def test_prefers_claim_dense_chunk() -> None:
    dense = _result(
        "dense",
        score=0.0,
        text="We show strong gains. Our results indicate robustness. Background only.",
    )
    sparse = _result(
        "sparse",
        score=1.0,
        text="Background on datasets. Related work overview.",
    )
    boosted = ClaimDensityBooster(alpha=1.0).boost([sparse, dense], query="ignored")
    assert [item.chunk.chunk_id for item in boosted] == ["dense", "sparse"]
    assert boosted[0].retriever == "claim_density_boost"
    assert sparse.score == 1.0


def test_empty_text_yields_zero_density() -> None:
    result = _result("only", score=0.8, text="")
    boosted = ClaimDensityBooster(alpha=1.0).boost([result], query="q")
    assert boosted[0].score == pytest.approx(0.0)


def test_blend_formula() -> None:
    result = _result("only", score=0.4, text="We find a clear effect.")
    boosted = ClaimDensityBooster(alpha=0.5).boost([result], query="q")
    assert boosted[0].score == pytest.approx(0.7)


def test_top_k_and_empty() -> None:
    booster = ClaimDensityBooster(alpha=1.0)
    assert booster.boost([], query="q") == []
    rows = [
        _result("a", 0.0, text="We conclude success."),
        _result("b", 0.0, text="Neutral background."),
    ]
    assert len(booster.boost(rows, query="q", top_k=1)) == 1
