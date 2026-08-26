"""Tests for near-duplicate retrieval collapse."""

import pytest

from retrieval.models import Chunk, SearchResult
from retrieval.near_duplicate_collapse import NearDuplicateCollapser


def _result(
    chunk_id: str,
    text: str,
    score: float,
    title: str = "title",
) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id=f"doc-{chunk_id}",
            title=title,
            text=text,
            source="test",
            metadata={},
        ),
        score=score,
        retriever="bm25",
        path=["hybrid"],
    )


@pytest.mark.parametrize(
    "threshold",
    [-0.1, 1.1, float("nan"), float("inf")],
)
def test_rejects_invalid_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        NearDuplicateCollapser(threshold=threshold)


def test_collapses_near_duplicate_texts_keeping_highest_score() -> None:
    high = _result("high", "graph neural networks predict properties", score=0.9)
    low = _result("low", "graph neural networks predict properties", score=0.2)
    other = _result("other", "soil moisture remote sensing survey", score=0.5)
    collapser = NearDuplicateCollapser(threshold=0.9)

    collapsed = collapser.collapse([low, other, high])

    assert [result.chunk.chunk_id for result in collapsed] == ["high", "other"]
    assert collapsed[0] is high
    assert collapsed[1] is other
    assert low.score == 0.2  # originals not mutated


def test_preserves_score_order_among_survivors() -> None:
    a = _result("a", "alpha beta gamma delta", score=0.3)
    b = _result("b", "completely different wording here", score=0.8)
    c = _result("c", "another unique passage about climate", score=0.5)
    collapser = NearDuplicateCollapser(threshold=0.9)

    collapsed = collapser.collapse([a, b, c])

    assert [result.chunk.chunk_id for result in collapsed] == ["b", "c", "a"]


def test_threshold_controls_collapse() -> None:
    # Two texts share 3 of 4 terms → Jaccard = 0.75
    left = _result("left", "alpha beta gamma", score=0.9)
    right = _result("right", "alpha beta gamma delta", score=0.8)

    strict = NearDuplicateCollapser(threshold=0.9).collapse([left, right])
    loose = NearDuplicateCollapser(threshold=0.7).collapse([left, right])

    assert [result.chunk.chunk_id for result in strict] == ["left", "right"]
    assert [result.chunk.chunk_id for result in loose] == ["left"]


def test_uses_text_not_title() -> None:
    # Identical titles would look like duplicates if title were used; texts differ.
    first = _result("first", "alpha beta gamma", score=0.9, title="same title words")
    second = _result("second", "delta epsilon zeta", score=0.8, title="same title words")
    collapser = NearDuplicateCollapser(threshold=0.9)

    collapsed = collapser.collapse([first, second])

    assert [result.chunk.chunk_id for result in collapsed] == ["first", "second"]


def test_top_k_truncates_after_collapse() -> None:
    results = [
        _result("a", "unique passage about graphs", score=0.9),
        _result("b", "unique passage about graphs", score=0.8),  # near-dup of a
        _result("c", "soil moisture remote sensing", score=0.7),
        _result("d", "transformer attention mechanisms", score=0.6),
    ]
    collapser = NearDuplicateCollapser(threshold=0.9)

    collapsed = collapser.collapse(results, top_k=2)

    assert len(collapsed) == 2
    assert [result.chunk.chunk_id for result in collapsed] == ["a", "c"]


def test_empty_input_returns_empty() -> None:
    assert NearDuplicateCollapser().collapse([]) == []


def test_empty_texts_collapse_together() -> None:
    first = _result("first", "the and of", score=0.9)  # stopwords only → empty terms
    second = _result("second", "a an the", score=0.5)
    other = _result("other", "graph neural networks", score=0.7)
    collapser = NearDuplicateCollapser(threshold=0.9)

    collapsed = collapser.collapse([first, second, other])

    assert [result.chunk.chunk_id for result in collapsed] == ["first", "other"]


def test_identical_objects_survive_as_single_representative() -> None:
    only = _result("only", "graph neural networks", score=0.5)
    collapser = NearDuplicateCollapser()

    collapsed = collapser.collapse([only])

    assert collapsed == [only]
    assert collapsed[0] is only
