"""Tests for citation-count retrieval boosting."""

import math

import pytest

from retrieval.citation_count_boost import CitationCountBooster
from retrieval.models import Chunk, SearchResult


def _result(
    chunk_id: str,
    score: float,
    metadata: dict[str, str] | None = None,
) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id=f"doc-{chunk_id}",
            title=chunk_id,
            text=f"text for {chunk_id}",
            source="test",
            metadata={} if metadata is None else metadata,
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
        CitationCountBooster(alpha=alpha)


def test_boost_prefers_higher_citation_counts() -> None:
    low = _result("low", score=1.0, metadata={"citation_count": "0"})
    high = _result("high", score=0.0, metadata={"citation_count": "1000"})
    booster = CitationCountBooster(alpha=1.0)

    boosted = booster.boost([low, high])

    assert [result.chunk.chunk_id for result in boosted] == ["high", "low"]
    assert boosted[0].score == pytest.approx(1.0)
    assert boosted[1].score == pytest.approx(0.0)
    assert boosted[0].retriever == "citation_count_boost"
    assert boosted[0].path == ["hybrid", "bm25"]
    assert low.score == 1.0  # originals not mutated


def test_reads_cited_by_count_fallback() -> None:
    result = _result("only", score=0.0, metadata={"cited_by_count": "99"})
    booster = CitationCountBooster(alpha=1.0)

    boosted = booster.boost([result])

    assert boosted[0].score == pytest.approx(1.0)


def test_prefers_citation_count_over_cited_by_count() -> None:
    result = _result(
        "only",
        score=0.0,
        metadata={"citation_count": "0", "cited_by_count": "999"},
    )
    booster = CitationCountBooster(alpha=1.0)

    boosted = booster.boost([result])

    assert boosted[0].score == pytest.approx(0.0)


def test_missing_count_treated_as_zero() -> None:
    missing = _result("missing", score=0.5, metadata={})
    invalid = _result("invalid", score=0.5, metadata={"citation_count": "n/a"})
    booster = CitationCountBooster(alpha=1.0)

    boosted = booster.boost([missing, invalid])

    assert boosted[0].score == pytest.approx(0.0)
    assert boosted[1].score == pytest.approx(0.0)


def test_formula_matches_one_minus_alpha_old_plus_alpha_normalized_log1p() -> None:
    result = _result("only", score=0.8, metadata={"citation_count": "10"})
    booster = CitationCountBooster(alpha=0.3)

    boosted = booster.boost([result])

    # Single non-zero count → normalized log1p signal is 1.0
    assert boosted[0].score == pytest.approx((1 - 0.3) * 0.8 + 0.3 * 1.0)


def test_normalizes_log1p_across_batch() -> None:
    low = _result("low", score=0.0, metadata={"citation_count": "1"})
    high = _result("high", score=0.0, metadata={"citation_count": "10"})
    booster = CitationCountBooster(alpha=1.0)

    boosted = booster.boost([low, high])

    expected_low = math.log1p(1.0) / math.log1p(10.0)
    by_id = {result.chunk.chunk_id: result.score for result in boosted}
    assert by_id["high"] == pytest.approx(1.0)
    assert by_id["low"] == pytest.approx(expected_low)


def test_stable_ordering_for_tied_scores() -> None:
    first = _result("first", score=0.5, metadata={"citation_count": "5"})
    second = _result("second", score=0.5, metadata={"citation_count": "5"})
    booster = CitationCountBooster(alpha=0.0)

    boosted = booster.boost([first, second])

    assert [result.chunk.chunk_id for result in boosted] == ["first", "second"]


def test_top_k_truncates_after_sort() -> None:
    results = [
        _result("a", score=0.9, metadata={"citation_count": "0"}),
        _result("b", score=0.1, metadata={"citation_count": "100"}),
        _result("c", score=0.2, metadata={"citation_count": "10"}),
    ]
    booster = CitationCountBooster(alpha=1.0)

    boosted = booster.boost(results, top_k=2)

    assert len(boosted) == 2
    assert boosted[0].chunk.chunk_id == "b"


def test_empty_input_returns_empty() -> None:
    assert CitationCountBooster().boost([]) == []
