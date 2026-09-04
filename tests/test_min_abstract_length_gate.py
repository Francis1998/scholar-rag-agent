"""Tests for MinAbstractLengthGate postprocessor."""

import pytest

from retrieval.min_abstract_length_gate import MinAbstractLengthGate
from retrieval.models import Chunk, SearchResult


def _result(chunk_id: str, text: str, score: float = 1.0) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id="doc-a",
            title=chunk_id,
            text=text,
            source="test",
            metadata={},
        ),
        score=score,
        retriever="bm25",
        path=["hybrid"],
    )


def test_rejects_negative_min_chars() -> None:
    with pytest.raises(ValueError, match="min_chars"):
        MinAbstractLengthGate(min_chars=-1)


def test_rejects_non_integer_min_chars() -> None:
    with pytest.raises(ValueError, match="min_chars"):
        MinAbstractLengthGate(min_chars=2.5)  # type: ignore[arg-type]


def test_empty_results() -> None:
    assert MinAbstractLengthGate().gate([]) == []


def test_filters_short_text() -> None:
    hits = [_result("a", "short"), _result("b", "x" * 100)]
    kept = MinAbstractLengthGate(min_chars=80).gate(hits)
    assert [r.chunk.chunk_id for r in kept] == ["b"]


def test_keeps_equal_length() -> None:
    hits = [_result("a", "a" * 80)]
    kept = MinAbstractLengthGate(min_chars=80).gate(hits)
    assert len(kept) == 1


def test_provenance_rewritten() -> None:
    kept = MinAbstractLengthGate(min_chars=1).gate([_result("a", "hello")])
    assert kept[0].retriever == "min_abstract_length_gate"
    assert kept[0].path == ["hybrid", "bm25"]


def test_top_k_limits_output() -> None:
    hits = [_result("a", "a" * 100), _result("b", "b" * 100), _result("c", "c" * 100)]
    kept = MinAbstractLengthGate(min_chars=10).gate(hits, top_k=2)
    assert len(kept) == 2


def test_top_k_zero_returns_empty() -> None:
    assert MinAbstractLengthGate(min_chars=0).gate([_result("a", "x")], top_k=0) == []


def test_does_not_mutate_inputs() -> None:
    original = [_result("a", "hello world")]
    path_before = list(original[0].path)
    MinAbstractLengthGate(min_chars=1).gate(original)
    assert original[0].path == path_before
    assert original[0].retriever == "bm25"


def test_docstring_mentions_frontier_models() -> None:
    assert "GPT-5.5" in (MinAbstractLengthGate.__doc__ or "")
