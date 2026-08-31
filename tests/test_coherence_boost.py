"""Tests for coherence boost postprocessing."""

import pytest

from retrieval.coherence_boost import CoherenceBooster
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


@pytest.mark.parametrize("alpha", [-0.1, 1.1, float("nan"), float("inf")])
def test_rejects_invalid_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        CoherenceBooster(alpha=alpha)


def test_accepts_alpha_bounds() -> None:
    CoherenceBooster(alpha=0.0)
    CoherenceBooster(alpha=1.0)


def test_prefers_coherent_query_aligned_chunk() -> None:
    coherent = _result(
        "coherent",
        score=0.0,
        text=(
            "Transformers use attention for sequence modeling. "
            "Attention mechanisms improve transformer sequence tasks."
        ),
    )
    scattered = _result(
        "scattered",
        score=1.0,
        text="Photosynthesis uses chlorophyll. Neural networks need GPUs.",
    )
    boosted = CoherenceBooster(alpha=1.0).boost(
        [scattered, coherent],
        query="transformer attention",
    )
    assert [item.chunk.chunk_id for item in boosted] == ["coherent", "scattered"]
    assert boosted[0].retriever == "coherence_boost"
    assert boosted[0].path == ["hybrid", "bm25"]
    assert scattered.score == 1.0


def test_empty_query_zeros_continuity_component() -> None:
    result = _result(
        "only",
        score=0.8,
        text="Alpha beta gamma. Alpha beta delta.",
    )
    boosted = CoherenceBooster(alpha=1.0).boost([result], query="")
    # Neighbor Jaccard of shared {alpha,beta} over unions is > 0; continuity = 0.
    assert 0.0 < boosted[0].score < 1.0
    assert boosted[0].score == pytest.approx(0.5 * (2 / 4))


def test_empty_text_yields_zero_coherence() -> None:
    result = _result("only", score=0.8, text="")
    boosted = CoherenceBooster(alpha=1.0).boost([result], query="alpha")
    assert boosted[0].score == pytest.approx(0.0)


def test_single_sentence_has_zero_neighbor_overlap() -> None:
    result = _result("only", score=0.0, text="Transformers use attention mechanisms.")
    boosted = CoherenceBooster(alpha=1.0).boost([result], query="transformers attention")
    # Neighbor = 0; continuity = 1.0 → coherence = 0.5
    assert boosted[0].score == pytest.approx(0.5)


def test_blend_formula() -> None:
    result = _result(
        "only",
        score=0.4,
        text="Transformers use attention. Attention helps transformers.",
    )
    boosted = CoherenceBooster(alpha=0.5).boost([result], query="transformers attention")
    coherence = CoherenceBooster(alpha=1.0).boost([result], query="transformers attention")[0].score
    assert boosted[0].score == pytest.approx(0.5 * 0.4 + 0.5 * coherence)


def test_does_not_mutate_inputs() -> None:
    original = _result("a", score=0.9, text="One two three. One two four.")
    snapshot_score = original.score
    snapshot_path = list(original.path)
    CoherenceBooster(alpha=1.0).boost([original], query="one two")
    assert original.score == snapshot_score
    assert original.path == snapshot_path
    assert original.retriever == "bm25"


def test_stable_sort_for_tied_scores() -> None:
    first = _result("first", score=0.5, text="Same text here. Same text there.")
    second = _result("second", score=0.5, text="Same text here. Same text there.")
    boosted = CoherenceBooster(alpha=0.0).boost([first, second], query="same")
    assert [item.chunk.chunk_id for item in boosted] == ["first", "second"]


def test_top_k_and_empty() -> None:
    booster = CoherenceBooster(alpha=1.0)
    assert booster.boost([], query="q") == []
    assert booster.boost([_result("a", 0.5, "Hello world.")], query="hello", top_k=0) == []
    rows = [
        _result("a", 0.0, text="Alpha method works. Alpha method scales."),
        _result("b", 0.0, text="Beta gamma delta. Unrelated other topic."),
    ]
    assert len(booster.boost(rows, query="alpha method", top_k=1)) == 1


def test_sorting_is_descending_by_score() -> None:
    low = _result("low", 0.0, text="Unrelated cooking pasta. Baking bread recipes.")
    high = _result(
        "high",
        0.0,
        text="Retrieval ranking uses coherence. Coherence improves retrieval ranking.",
    )
    mid = _result("mid", 0.0, text="Retrieval uses sparse scores. Baking bread recipes.")
    boosted = CoherenceBooster(alpha=1.0).boost(
        [low, mid, high],
        query="retrieval coherence",
    )
    scores = [item.score for item in boosted]
    assert scores == sorted(scores, reverse=True)
    assert boosted[0].chunk.chunk_id == "high"
