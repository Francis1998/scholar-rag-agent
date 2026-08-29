"""Tests for abstract overlap boost postprocessing."""

import pytest

from retrieval.abstract_overlap_boost import AbstractOverlapBooster
from retrieval.models import Chunk, SearchResult


def _result(chunk_id: str, score: float, abstract: str = "") -> SearchResult:
    metadata = {"abstract": abstract} if abstract else {}
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
        AbstractOverlapBooster(alpha=alpha)


def test_prefers_overlapping_abstract() -> None:
    match = _result("m", score=0.0, abstract="transformer attention language model")
    miss = _result("x", score=1.0, abstract="photosynthesis in green plants")
    boosted = AbstractOverlapBooster(alpha=1.0).boost(
        [miss, match], query="transformer language model"
    )
    assert [item.chunk.chunk_id for item in boosted] == ["m", "x"]
    assert boosted[0].retriever == "abstract_overlap_boost"
    assert miss.score == 1.0


def test_empty_query_or_abstract_yields_zero_overlap() -> None:
    result = _result("only", score=0.8, abstract="alpha beta")
    boosted = AbstractOverlapBooster(alpha=1.0).boost([result], query="")
    assert boosted[0].score == pytest.approx(0.0)


def test_blend_formula() -> None:
    result = _result("only", score=0.5, abstract="neural network")
    boosted = AbstractOverlapBooster(alpha=0.5).boost([result], query="neural network")
    # identical token sets -> jaccard 1.0
    assert boosted[0].score == pytest.approx(0.75)


def test_top_k_and_empty() -> None:
    booster = AbstractOverlapBooster(alpha=1.0)
    assert booster.boost([], query="q") == []
    rows = [_result("a", 0.0, "alpha"), _result("b", 0.0, "beta gamma")]
    assert len(booster.boost(rows, query="alpha", top_k=1)) == 1
