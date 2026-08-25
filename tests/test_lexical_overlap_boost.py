"""Tests for lexical-overlap retrieval boosting."""

import pytest

from retrieval.lexical_overlap_boost import LexicalOverlapBooster
from retrieval.models import Chunk, SearchResult


def _result(
    chunk_id: str,
    text: str,
    score: float,
    title: str | None = None,
) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id=f"doc-{chunk_id}",
            title=title if title is not None else chunk_id,
            text=text,
            source="test",
            metadata={},
        ),
        score=score,
        retriever="bm25",
        path=["hybrid"],
    )


@pytest.mark.parametrize(
    "alpha",
    [-0.1, 1.1, float("nan"), float("inf")],
)
def test_rejects_invalid_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        LexicalOverlapBooster(alpha=alpha)


def test_boost_blends_old_score_with_jaccard_overlap() -> None:
    high = _result("high", "graph neural networks", score=0.0, title="")
    low = _result("low", "unrelated soil moisture", score=1.0, title="")
    booster = LexicalOverlapBooster(alpha=1.0)

    boosted = booster.boost("graph neural networks", [low, high])

    assert [result.chunk.chunk_id for result in boosted] == ["high", "low"]
    assert boosted[0].score == pytest.approx(1.0)
    assert boosted[1].score == pytest.approx(0.0)
    assert boosted[0].retriever == "lexical_overlap_boost"
    assert boosted[0].path == ["hybrid", "bm25"]
    assert low.score == 1.0  # originals not mutated


def test_formula_matches_one_minus_alpha_old_plus_alpha_overlap() -> None:
    result = _result("only", "graph neural networks", score=0.8, title="")
    booster = LexicalOverlapBooster(alpha=0.3)

    boosted = booster.boost("graph neural networks", [result])

    # Jaccard(|{graph,neural,networks}| / same) = 1.0
    assert boosted[0].score == pytest.approx((1 - 0.3) * 0.8 + 0.3 * 1.0)


def test_stable_ordering_for_tied_scores() -> None:
    first = _result("first", "alpha beta", score=0.5, title="")
    second = _result("second", "alpha beta", score=0.5, title="")
    booster = LexicalOverlapBooster(alpha=0.0)

    boosted = booster.boost("alpha beta", [first, second])

    assert [result.chunk.chunk_id for result in boosted] == ["first", "second"]


def test_top_k_truncates_after_sort() -> None:
    results = [
        _result("a", "unrelated text", score=0.9, title=""),
        _result("b", "graph neural networks", score=0.1, title=""),
        _result("c", "graph networks", score=0.2, title=""),
    ]
    booster = LexicalOverlapBooster(alpha=1.0)

    boosted = booster.boost("graph neural networks", results, top_k=2)

    assert len(boosted) == 2
    assert boosted[0].chunk.chunk_id == "b"


def test_empty_input_and_stopword_query() -> None:
    booster = LexicalOverlapBooster()
    assert booster.boost("anything", []) == []
    boosted = booster.boost("the and of", [_result("x", "the and of", 1.0, title="")])
    # Both term sets empty → overlap 0; alpha=0.3 → 0.7 * 1 + 0.3 * 0
    assert boosted[0].score == pytest.approx(0.7)
