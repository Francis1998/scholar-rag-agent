"""Tests for open-access preference postprocessing."""

import pytest

from retrieval.models import Chunk, SearchResult
from retrieval.open_access_prefer import OpenAccessPreferencer


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
        OpenAccessPreferencer(alpha=alpha)


@pytest.mark.parametrize("mode", ["drop", "rank", ""])
def test_rejects_invalid_mode(mode: str) -> None:
    with pytest.raises(ValueError, match="mode"):
        OpenAccessPreferencer(mode=mode)


def test_boost_prefers_open_access() -> None:
    closed = _result("closed", score=1.0, metadata={"is_oa": "false"})
    opened = _result("open", score=0.0, metadata={"is_oa": "true"})
    preferencer = OpenAccessPreferencer(alpha=1.0, mode="boost")

    preferred = preferencer.prefer([closed, opened])

    assert [result.chunk.chunk_id for result in preferred] == ["open", "closed"]
    assert preferred[0].score == pytest.approx(1.0)
    assert preferred[1].score == pytest.approx(0.0)
    assert preferred[0].retriever == "open_access_prefer"
    assert preferred[0].path == ["hybrid", "bm25"]
    assert closed.score == 1.0  # originals not mutated


def test_reads_open_access_and_oa_keys() -> None:
    a = _result("a", score=0.0, metadata={"open_access": "yes"})
    b = _result("b", score=0.0, metadata={"oa": "1"})
    preferencer = OpenAccessPreferencer(alpha=1.0)

    preferred = preferencer.prefer([a, b])

    assert preferred[0].score == pytest.approx(1.0)
    assert preferred[1].score == pytest.approx(1.0)


def test_formula_matches_one_minus_alpha_old_plus_alpha_oa() -> None:
    result = _result("only", score=0.8, metadata={"is_oa": "true"})
    preferencer = OpenAccessPreferencer(alpha=0.3)

    preferred = preferencer.prefer([result])

    assert preferred[0].score == pytest.approx((1 - 0.3) * 0.8 + 0.3 * 1.0)


def test_filter_drops_non_oa_when_any_oa_exists() -> None:
    opened = _result("open", score=0.2, metadata={"is_oa": "true"})
    closed = _result("closed", score=0.9, metadata={"is_oa": "false"})
    preferencer = OpenAccessPreferencer(mode="filter")

    preferred = preferencer.prefer([closed, opened])

    assert preferred == [opened]
    assert preferred[0] is opened


def test_filter_keeps_all_when_no_oa() -> None:
    first = _result("first", score=0.4, metadata={"is_oa": "false"})
    second = _result("second", score=0.6, metadata={})
    preferencer = OpenAccessPreferencer(mode="filter")

    preferred = preferencer.prefer([first, second])

    assert preferred == [first, second]
    assert preferred[0] is first


def test_stable_ordering_for_tied_boost_scores() -> None:
    first = _result("first", score=0.5, metadata={"is_oa": "true"})
    second = _result("second", score=0.5, metadata={"is_oa": "true"})
    preferencer = OpenAccessPreferencer(alpha=0.0)

    preferred = preferencer.prefer([first, second])

    assert [result.chunk.chunk_id for result in preferred] == ["first", "second"]


def test_top_k_truncates_after_preference() -> None:
    results = [
        _result("a", score=0.9, metadata={"is_oa": "false"}),
        _result("b", score=0.1, metadata={"is_oa": "true"}),
        _result("c", score=0.2, metadata={"is_oa": "true"}),
    ]
    preferencer = OpenAccessPreferencer(alpha=1.0)

    preferred = preferencer.prefer(results, top_k=2)

    assert len(preferred) == 2
    assert preferred[0].chunk.chunk_id == "b"


def test_empty_input_returns_empty() -> None:
    assert OpenAccessPreferencer().prefer([]) == []
