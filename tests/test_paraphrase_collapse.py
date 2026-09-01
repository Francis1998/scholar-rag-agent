"""Tests for paraphrase collapse postprocessing."""

import pytest

from retrieval.models import Chunk, SearchResult
from retrieval.paraphrase_collapse import ParaphraseCollapser


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


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan")])
def test_rejects_invalid_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        ParaphraseCollapser(threshold=threshold)


@pytest.mark.parametrize("ngram_size", [1, 0, -1])
def test_rejects_invalid_ngram_size(ngram_size: int) -> None:
    with pytest.raises(ValueError, match="ngram_size"):
        ParaphraseCollapser(ngram_size=ngram_size)


def test_collapses_light_paraphrase() -> None:
    results = [
        _result("a", 1.0, "Transformers use self attention for language modeling tasks"),
        _result("b", 0.95, "Transformers use self-attention for language modelling tasks"),
        _result("c", 0.8, "Graph neural nets predict molecular properties accurately"),
    ]
    kept = ParaphraseCollapser(threshold=0.7).collapse(results)
    ids = [item.chunk.chunk_id for item in kept]
    assert "a" in ids
    assert "c" in ids
    assert "b" not in ids
    assert kept[0].retriever == "paraphrase_collapse"


def test_keeps_unrelated_high_scorers() -> None:
    results = [
        _result("a", 1.0, "alpha beta gamma delta epsilon"),
        _result("b", 0.9, "one two three four five six"),
    ]
    kept = ParaphraseCollapser(threshold=0.85).collapse(results)
    assert [item.chunk.chunk_id for item in kept] == ["a", "b"]


def test_does_not_mutate_inputs() -> None:
    original = _result("a", 0.9, "same text content here")
    other = _result("b", 0.8, "different material entirely now")
    snapshot = original.score
    ParaphraseCollapser().collapse([original, other])
    assert original.score == snapshot
    assert original.retriever == "bm25"


def test_empty_and_top_k() -> None:
    collapser = ParaphraseCollapser()
    assert collapser.collapse([]) == []
    assert collapser.collapse([_result("a", 0.5, "hello world")], top_k=0) == []
    rows = [
        _result("a", 1.0, "one unique sentence about cats"),
        _result("b", 0.9, "another unique sentence about dogs"),
        _result("c", 0.8, "third unique sentence about birds"),
    ]
    assert len(collapser.collapse(rows, top_k=2)) == 2


def test_path_rewritten() -> None:
    result = _result("a", 0.5, "hello world again")
    kept = ParaphraseCollapser().collapse([result])
    assert kept[0].path == ["hybrid", "bm25"]


def test_identical_empty_texts_collapse() -> None:
    results = [
        _result("a", 1.0, ""),
        _result("b", 0.9, ""),
        _result("c", 0.8, "novel unique content present"),
    ]
    kept = ParaphraseCollapser().collapse(results)
    ids = [item.chunk.chunk_id for item in kept]
    assert ids == ["a", "c"]
