"""Tests for cross-encoder gate postprocessing."""

import pytest

from retrieval.cross_encoder_gate import CrossEncoderGate
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


@pytest.mark.parametrize("min_score", [-0.1, 1.1, float("nan")])
def test_rejects_invalid_min_score(min_score: float) -> None:
    with pytest.raises(ValueError, match="min_score"):
        CrossEncoderGate(min_score=min_score)


def test_drops_unrelated_keeps_overlapping() -> None:
    results = [
        _result("a", 1.0, "transformer attention language models"),
        _result("b", 0.9, "gardening soil compost tomatoes"),
    ]
    kept = CrossEncoderGate(min_score=0.3).gate(results, query="transformer language models")
    ids = [item.chunk.chunk_id for item in kept]
    assert ids == ["a"]
    assert kept[0].retriever == "cross_encoder_gate"
    assert kept[0].path == ["hybrid", "bm25"]


def test_min_score_zero_keeps_all_scored() -> None:
    results = [
        _result("a", 1.0, "alpha"),
        _result("b", 0.5, "beta"),
    ]
    kept = CrossEncoderGate(min_score=0.0).gate(results, query="alpha")
    assert len(kept) == 2


def test_does_not_mutate_inputs() -> None:
    original = _result("a", 0.9, "one two three")
    snapshot = original.score
    CrossEncoderGate(min_score=0.0).gate([original], query="one")
    assert original.score == snapshot
    assert original.retriever == "bm25"


def test_empty_and_top_k() -> None:
    gate = CrossEncoderGate(min_score=0.0)
    assert gate.gate([], query="q") == []
    assert gate.gate([_result("a", 0.5, "hello")], query="hello", top_k=0) == []
    rows = [
        _result("a", 1.0, "cats and dogs"),
        _result("b", 0.9, "cats and birds"),
        _result("c", 0.8, "cats and fish"),
    ]
    assert len(gate.gate(rows, query="cats", top_k=2)) == 2


def test_proxy_formula_known_value() -> None:
    # query={a,b} doc={a,b,c} → jaccard=2/3 coverage=1 → 0.5*(2/3)+0.5*1 = 5/6
    result = _result("a", 0.1, "a b c")
    kept = CrossEncoderGate(min_score=0.0).gate([result], query="a b")
    assert kept[0].score == pytest.approx(5 / 6)


def test_empty_query_and_empty_doc_is_one() -> None:
    kept = CrossEncoderGate(min_score=0.5).gate([_result("a", 0.1, "")], query="")
    assert len(kept) == 1
    assert kept[0].score == pytest.approx(1.0)


def test_empty_query_nonempty_doc_is_zero() -> None:
    kept = CrossEncoderGate(min_score=0.0).gate([_result("a", 0.1, "hello")], query="")
    assert kept[0].score == pytest.approx(0.0)
