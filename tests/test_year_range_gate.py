"""Tests for YearRangeGate postprocessor."""

import pytest

from retrieval.models import Chunk, SearchResult
from retrieval.year_range_gate import YearRangeGate


def _result(chunk_id: str, year: str | None, score: float = 1.0) -> SearchResult:
    metadata = {} if year is None else {"year": year}
    return SearchResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id="doc-a",
            title=chunk_id,
            text="body",
            source="test",
            metadata=metadata,
        ),
        score=score,
        retriever="bm25",
        path=["hybrid"],
    )


def test_rejects_non_integer_bounds() -> None:
    with pytest.raises(ValueError, match="min_year"):
        YearRangeGate(min_year=2010.5)  # type: ignore[arg-type]


def test_rejects_inverted_range() -> None:
    with pytest.raises(ValueError, match="min_year"):
        YearRangeGate(min_year=2020, max_year=2010)


def test_empty_results() -> None:
    assert YearRangeGate(min_year=2000).gate([]) == []


def test_filters_outside_range() -> None:
    hits = [_result("a", "1999"), _result("b", "2015"), _result("c", "2025")]
    kept = YearRangeGate(min_year=2000, max_year=2020).gate(hits)
    assert [r.chunk.chunk_id for r in kept] == ["b"]


def test_drops_missing_year() -> None:
    hits = [_result("a", None), _result("b", "2018")]
    kept = YearRangeGate(min_year=2010).gate(hits)
    assert [r.chunk.chunk_id for r in kept] == ["b"]


def test_provenance_rewritten() -> None:
    kept = YearRangeGate(min_year=2000).gate([_result("a", "2010")])
    assert kept[0].retriever == "year_range_gate"
    assert kept[0].path == ["hybrid", "bm25"]


def test_top_k_limits_output() -> None:
    hits = [_result("a", "2011"), _result("b", "2012"), _result("c", "2013")]
    kept = YearRangeGate(min_year=2000).gate(hits, top_k=2)
    assert len(kept) == 2


def test_top_k_zero_returns_empty() -> None:
    assert YearRangeGate().gate([_result("a", "2010")], top_k=0) == []


def test_does_not_mutate_inputs() -> None:
    original = [_result("a", "2010")]
    path_before = list(original[0].path)
    YearRangeGate(min_year=2000).gate(original)
    assert original[0].path == path_before
    assert original[0].retriever == "bm25"


def test_docstring_mentions_frontier_models() -> None:
    assert "GPT-5.5" in (YearRangeGate.__doc__ or "")
