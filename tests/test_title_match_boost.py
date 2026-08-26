"""Tests for title-only lexical retrieval boosting."""

import pytest

from retrieval.models import Chunk, SearchResult
from retrieval.title_match_boost import TitleMatchBooster


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
        TitleMatchBooster(alpha=alpha)


def test_boost_uses_title_not_body() -> None:
    # Body matches the query but title does not → low title overlap.
    body_hit = _result(
        "body",
        "graph neural networks classify molecules",
        score=1.0,
        title="unrelated soil survey",
    )
    # Title matches the query; body does not → high title overlap.
    title_hit = _result(
        "title",
        "unrelated soil moisture measurements",
        score=0.0,
        title="graph neural networks",
    )
    booster = TitleMatchBooster(alpha=1.0)

    boosted = booster.boost("graph neural networks", [body_hit, title_hit])

    assert [result.chunk.chunk_id for result in boosted] == ["title", "body"]
    assert boosted[0].score == pytest.approx(1.0)
    assert boosted[1].score == pytest.approx(0.0)
    assert boosted[0].retriever == "title_match_boost"
    assert boosted[0].path == ["hybrid", "bm25"]
    assert body_hit.score == 1.0  # originals not mutated


def test_distinct_from_body_only_lexical_overlap() -> None:
    """Title-only scoring ignores body terms that LexicalOverlapBooster would use."""
    result = _result(
        "only",
        "graph neural networks appear in the abstract body",
        score=0.8,
        title="completely unrelated heading",
    )
    booster = TitleMatchBooster(alpha=1.0)

    boosted = booster.boost("graph neural networks", [result])

    assert boosted[0].score == pytest.approx(0.0)


def test_formula_matches_one_minus_alpha_old_plus_alpha_overlap() -> None:
    result = _result("only", "body ignored", score=0.8, title="graph neural networks")
    booster = TitleMatchBooster(alpha=0.3)

    boosted = booster.boost("graph neural networks", [result])

    # Jaccard(|{graph,neural,networks}| / same) = 1.0
    assert boosted[0].score == pytest.approx((1 - 0.3) * 0.8 + 0.3 * 1.0)


def test_stable_ordering_for_tied_scores() -> None:
    first = _result("first", "body a", score=0.5, title="alpha beta")
    second = _result("second", "body b", score=0.5, title="alpha beta")
    booster = TitleMatchBooster(alpha=0.0)

    boosted = booster.boost("alpha beta", [first, second])

    assert [result.chunk.chunk_id for result in boosted] == ["first", "second"]


def test_top_k_truncates_after_sort() -> None:
    results = [
        _result("a", "graph neural networks", score=0.9, title="unrelated"),
        _result("b", "unrelated body", score=0.1, title="graph neural networks"),
        _result("c", "partial", score=0.2, title="graph networks"),
    ]
    booster = TitleMatchBooster(alpha=1.0)

    boosted = booster.boost("graph neural networks", results, top_k=2)

    assert len(boosted) == 2
    assert boosted[0].chunk.chunk_id == "b"


def test_empty_input_and_stopword_query() -> None:
    booster = TitleMatchBooster()
    assert booster.boost("anything", []) == []
    boosted = booster.boost("the and of", [_result("x", "body", 1.0, title="the and of")])
    # Both term sets empty → overlap 0; alpha=0.3 → 0.7 * 1 + 0.3 * 0
    assert boosted[0].score == pytest.approx(0.7)


def test_empty_title_yields_zero_overlap_when_query_has_terms() -> None:
    result = _result("empty-title", "graph neural networks", score=1.0, title="")
    booster = TitleMatchBooster(alpha=1.0)

    boosted = booster.boost("graph neural networks", [result])

    assert boosted[0].score == pytest.approx(0.0)
