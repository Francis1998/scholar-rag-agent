"""Tests for entity overlap boost postprocessing."""

import pytest

from retrieval.entity_overlap_boost import EntityOverlapBooster
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
        EntityOverlapBooster(alpha=alpha)


def test_prefers_overlapping_entities() -> None:
    match = _result(
        "m",
        score=0.0,
        text="Neural Network training improves benchmark scores.",
    )
    miss = _result(
        "x",
        score=1.0,
        text="Photosynthesis in green plants uses chlorophyll.",
    )
    boosted = EntityOverlapBooster(alpha=1.0).boost([miss, match], query="Neural Network")
    assert [item.chunk.chunk_id for item in boosted] == ["m", "x"]
    assert boosted[0].retriever == "entity_overlap_boost"
    assert miss.score == 1.0


def test_empty_query_or_text_yields_zero_overlap() -> None:
    result = _result("only", score=0.8, text="Alpha Beta method")
    boosted = EntityOverlapBooster(alpha=1.0).boost([result], query="")
    assert boosted[0].score == pytest.approx(0.0)


def test_blend_formula() -> None:
    result = _result("only", score=0.5, text="Neural Network training")
    boosted = EntityOverlapBooster(alpha=0.5).boost([result], query="Neural Network")
    assert boosted[0].score == pytest.approx(0.75)


def test_acronym_entities_match() -> None:
    result = _result("only", score=0.0, text="RAG pipelines use LLM retrieval.")
    boosted = EntityOverlapBooster(alpha=1.0).boost([result], query="LLM RAG")
    assert boosted[0].score == pytest.approx(1.0)


def test_top_k_and_empty() -> None:
    booster = EntityOverlapBooster(alpha=1.0)
    assert booster.boost([], query="q") == []
    rows = [
        _result("a", 0.0, text="Alpha Method"),
        _result("b", 0.0, text="Beta Gamma"),
    ]
    assert len(booster.boost(rows, query="Alpha", top_k=1)) == 1
