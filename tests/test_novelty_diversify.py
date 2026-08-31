"""Tests for novelty diversify postprocessing."""

import pytest

from retrieval.models import Chunk, SearchResult
from retrieval.novelty_diversify import NoveltyDiversifier


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


@pytest.mark.parametrize("alpha", [-0.1, 1.1, float("nan"), float("inf")])
def test_rejects_invalid_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        NoveltyDiversifier(alpha=alpha)


def test_accepts_alpha_bounds() -> None:
    NoveltyDiversifier(alpha=0.0)
    NoveltyDiversifier(alpha=1.0)


def test_demotes_near_duplicate_of_top_result() -> None:
    results = [
        _result("a", 1.0, "transformer attention mechanism language models"),
        _result("b", 0.95, "transformer attention mechanism language modeling"),
        _result("c", 0.85, "graph neural networks molecular property prediction"),
    ]
    ordered = NoveltyDiversifier(alpha=0.8).diversify(results)
    ids = [item.chunk.chunk_id for item in ordered]
    assert ids[0] == "a"
    assert ids.index("c") < ids.index("b")
    assert ordered[0].retriever == "novelty_diversify"
    assert ordered[0].path == ["hybrid", "bm25"]


def test_alpha_zero_preserves_relevance_order() -> None:
    results = [
        _result("low", 0.2, "alpha alpha alpha"),
        _result("high", 0.9, "beta beta beta"),
        _result("mid", 0.5, "gamma gamma gamma"),
    ]
    ordered = NoveltyDiversifier(alpha=0.0).diversify(results)
    assert [item.chunk.chunk_id for item in ordered] == ["high", "mid", "low"]


def test_alpha_one_ranks_by_novelty_only() -> None:
    # First pick is any row (all novelty=1); after selecting "a", the near-dupe
    # "b" should score near 0 while novel "c" stays near 1.
    results = [
        _result("a", 0.1, "transformer attention mechanism language"),
        _result("b", 0.9, "transformer attention mechanism language"),
        _result("c", 0.2, "graph neural molecular property"),
    ]
    ordered = NoveltyDiversifier(alpha=1.0).diversify(results)
    ids = [item.chunk.chunk_id for item in ordered]
    assert ids[0] in {"a", "b", "c"}
    # After first selection of a near-dupe pair member, the other is last.
    assert ids[-1] in {"a", "b"}
    assert "c" in ids[:2]


def test_does_not_mutate_inputs() -> None:
    original = _result("a", 0.9, "one two three four")
    other = _result("b", 0.8, "five six seven eight")
    snapshot = original.score
    NoveltyDiversifier(alpha=0.5).diversify([original, other])
    assert original.score == snapshot
    assert original.retriever == "bm25"
    assert original.path == ["hybrid"]


def test_empty_and_top_k() -> None:
    diversifier = NoveltyDiversifier(alpha=0.5)
    assert diversifier.diversify([]) == []
    assert diversifier.diversify([_result("a", 0.5, "hello")], top_k=0) == []
    rows = [
        _result("a", 1.0, "one two three"),
        _result("b", 0.9, "four five six"),
        _result("c", 0.8, "seven eight nine"),
    ]
    assert len(diversifier.diversify(rows, top_k=2)) == 2


def test_identical_empty_texts_are_full_duplicates() -> None:
    results = [
        _result("a", 1.0, ""),
        _result("b", 0.9, ""),
        _result("c", 0.8, "novel unique content here"),
    ]
    ordered = NoveltyDiversifier(alpha=1.0).diversify(results)
    ids = [item.chunk.chunk_id for item in ordered]
    assert ids[0] in {"a", "b"}
    assert ids[1] == "c"
    assert ids[2] in {"a", "b"}


def test_blend_formula_for_first_pick() -> None:
    result = _result("only", 0.4, "unique tokens here")
    ordered = NoveltyDiversifier(alpha=0.5).diversify([result])
    # First pick: novelty=1.0 → (1-0.5)*0.4 + 0.5*1.0 = 0.7
    assert ordered[0].score == pytest.approx(0.7)


def test_scores_decrease_for_exact_duplicate_second() -> None:
    results = [
        _result("a", 1.0, "same tokens exactly here"),
        _result("b", 1.0, "same tokens exactly here"),
    ]
    ordered = NoveltyDiversifier(alpha=1.0).diversify(results)
    assert ordered[0].score == pytest.approx(1.0)
    assert ordered[1].score == pytest.approx(0.0)


def test_sorting_output_follows_greedy_selection() -> None:
    results = [
        _result("a", 1.0, "aaa bbb ccc ddd"),
        _result("dup", 0.99, "aaa bbb ccc ddd"),
        _result("novel", 0.7, "www xxx yyy zzz"),
    ]
    ordered = NoveltyDiversifier(alpha=0.7).diversify(results)
    assert ordered[0].chunk.chunk_id == "a"
    assert ordered[1].chunk.chunk_id == "novel"
    assert ordered[2].chunk.chunk_id == "dup"


def test_path_and_retriever_rewritten() -> None:
    result = _result("a", 0.5, "hello world")
    ordered = NoveltyDiversifier(alpha=0.0).diversify([result])
    assert ordered[0].retriever == "novelty_diversify"
    assert ordered[0].path == ["hybrid", "bm25"]
