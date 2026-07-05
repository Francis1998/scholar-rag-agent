"""Tests for the MMR diversity re-ranker."""

import pytest

from retrieval.mmr import MMRDiversifier
from retrieval.models import Chunk, SearchResult


def _result(chunk_id: str, text: str, score: float) -> SearchResult:
    """Build a SearchResult with a chunk of given text and score."""
    return SearchResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id=f"doc-{chunk_id}",
            title=chunk_id,
            text=text,
            source="test",
        ),
        score=score,
        retriever="test",
    )


def test_mmr_rejects_out_of_range_lambda() -> None:
    """A lambda outside [0.0, 1.0] should fail fast at construction."""
    with pytest.raises(ValueError, match="lambda_param must be within"):
        MMRDiversifier(lambda_param=1.5)


def test_mmr_demotes_near_duplicate_of_top_result() -> None:
    """A near-duplicate of the top result should be demoted below a novel one."""
    results = [
        _result("a", "transformer attention mechanism language models", 1.0),
        _result("b", "transformer attention mechanism language modeling", 0.9),
        _result("c", "graph neural networks molecular property", 0.85),
        _result("d", "unrelated cooking recipe pasta bread", 0.1),
    ]
    diversifier = MMRDiversifier(lambda_param=0.5)

    ordered = [result.chunk.chunk_id for result in diversifier.diversify(results)]

    assert ordered[0] == "a"
    # The novel chunk "c" outranks the near-duplicate "b" under MMR.
    assert ordered == ["a", "c", "b", "d"]


def test_mmr_pure_relevance_preserves_score_order() -> None:
    """With lambda=1.0 MMR must reduce to plain relevance ordering."""
    results = [
        _result("a", "alpha alpha alpha", 0.2),
        _result("b", "beta beta beta", 0.9),
        _result("c", "gamma gamma gamma", 0.5),
    ]
    diversifier = MMRDiversifier(lambda_param=1.0)

    ordered = diversifier.diversify(results)

    assert [result.chunk.chunk_id for result in ordered] == ["b", "c", "a"]


def test_mmr_honors_top_k_and_empty_input() -> None:
    """top_k bounds the output and empty input yields an empty list."""
    results = [
        _result("a", "one two three", 0.9),
        _result("b", "four five six", 0.8),
        _result("c", "seven eight nine", 0.7),
    ]
    diversifier = MMRDiversifier(lambda_param=0.7)

    assert len(diversifier.diversify(results, top_k=2)) == 2
    assert diversifier.diversify([], top_k=5) == []
    assert diversifier.diversify(results, top_k=0) == []
