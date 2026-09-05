"""Tests for AbstractKeywordBoost postprocessor."""

import pytest

from retrieval.abstract_keyword_boost import AbstractKeywordBoost
from retrieval.models import Chunk, SearchResult


def _result(
    chunk_id: str,
    score: float = 1.0,
    text: str = "body",
    metadata: dict[str, str] | None = None,
) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id=f"doc-{chunk_id}",
            title=chunk_id,
            text=text,
            source="test",
            metadata=metadata or {},
        ),
        score=score,
        retriever="bm25",
        path=["hybrid"],
    )


def test_rejects_invalid_alpha() -> None:
    with pytest.raises(ValueError, match="alpha"):
        AbstractKeywordBoost(alpha=-0.1)


def test_empty_results() -> None:
    assert AbstractKeywordBoost().boost([], query="cancer") == []


def test_top_k_zero_returns_empty() -> None:
    hits = [_result("a", metadata={"abstract": "cancer therapy"})]
    assert AbstractKeywordBoost().boost(hits, query="cancer", top_k=0) == []


def test_boosts_abstract_keyword_hits() -> None:
    hits = [
        _result("miss", 1.0, metadata={"abstract": "unrelated methods"}),
        _result("hit", 0.5, metadata={"abstract": "cancer therapy outcomes"}),
    ]
    kept = AbstractKeywordBoost(alpha=0.8).boost(hits, query="cancer therapy")
    assert [r.chunk.chunk_id for r in kept] == ["hit", "miss"]


def test_falls_back_to_chunk_text() -> None:
    hits = [_result("a", 0.2, text="glucose metabolism pathway")]
    kept = AbstractKeywordBoost(alpha=1.0).boost(hits, query="glucose pathway")
    assert kept[0].score == pytest.approx(1.0)


def test_accepts_explicit_query_terms() -> None:
    hits = [_result("a", 0.0, metadata={"abstract": "CRISPR gene editing"})]
    kept = AbstractKeywordBoost(alpha=1.0).boost(
        hits,
        query="",
        query_terms=["CRISPR", "editing"],
    )
    assert kept[0].score == pytest.approx(1.0)


def test_uses_metadata_keywords_when_query_empty() -> None:
    hits = [
        _result(
            "a",
            0.0,
            metadata={
                "abstract": "neural network transformers",
                "keywords": "neural, transformers",
            },
        )
    ]
    kept = AbstractKeywordBoost(alpha=1.0).boost(hits, query="")
    assert kept[0].score == pytest.approx(1.0)


def test_partial_coverage() -> None:
    hits = [_result("a", 0.0, metadata={"abstract": "cancer research"})]
    kept = AbstractKeywordBoost(alpha=1.0).boost(hits, query="cancer therapy")
    assert kept[0].score == pytest.approx(0.5)


def test_provenance_rewritten() -> None:
    kept = AbstractKeywordBoost().boost(
        [_result("a", metadata={"abstract": "cancer"})],
        query="cancer",
    )
    assert kept[0].retriever == "abstract_keyword_boost"
    assert kept[0].path == ["hybrid", "bm25"]


def test_top_k_limits_output() -> None:
    hits = [
        _result("a", 1.0, metadata={"abstract": "alpha beta"}),
        _result("b", 1.0, metadata={"abstract": "alpha"}),
        _result("c", 1.0, metadata={"abstract": "gamma"}),
    ]
    kept = AbstractKeywordBoost().boost(hits, query="alpha beta", top_k=2)
    assert len(kept) == 2


def test_does_not_mutate_inputs() -> None:
    original = [_result("a", 1.0, metadata={"abstract": "cancer"})]
    path_before = list(original[0].path)
    score_before = original[0].score
    AbstractKeywordBoost().boost(original, query="cancer")
    assert original[0].path == path_before
    assert original[0].retriever == "bm25"
    assert original[0].score == score_before


def test_docstring_mentions_frontier_models() -> None:
    assert "GPT-5.5" in (AbstractKeywordBoost.__doc__ or "")
