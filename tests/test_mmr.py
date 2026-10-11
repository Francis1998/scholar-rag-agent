"""Tests for the MMR diversity re-ranker."""

import math
import sys

import pytest

from retrieval.mmr import MMRDiversifier
from retrieval.models import Chunk, SearchResult


def _result(chunk_id: str, text: str, score: float) -> SearchResult:
    """Build a SearchResult with a chunk of given text and score."""
    return SearchResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id=f"doc-{chunk_id}",
            title=chunk_id,
            text=text,
            source="test",
        ),
        score=score,
        retriever="test",
    )


def test_mmr_rejects_out_of_range_lambda() -> None:
    """A lambda outside [0.0, 1.0] should fail fast at construction."""
    with pytest.raises(ValueError, match="lambda_param must be within"):
        MMRDiversifier(lambda_param=1.5)


def test_mmr_demotes_near_duplicate_of_top_result() -> None:
    """A near-duplicate of the top result should be demoted below a novel one."""
    results = [
        _result("a", "transformer attention mechanism language models", 1.0),
        _result("b", "transformer attention mechanism language modeling", 0.9),
        _result("c", "graph neural networks molecular property", 0.85),
        _result("d", "unrelated cooking recipe pasta bread", 0.1),
    ]
    diversifier = MMRDiversifier(lambda_param=0.5)

    ordered = [result.chunk.chunk_id for result in diversifier.diversify(results)]

    assert ordered[0] == "a"
    # The novel chunk "c" outranks the near-duplicate "b" under MMR.
    assert ordered == ["a", "c", "b", "d"]


def test_mmr_pure_relevance_preserves_score_order() -> None:
    """With lambda=1.0 MMR must reduce to plain relevance ordering."""
    results = [
        _result("a", "alpha alpha alpha", 0.2),
        _result("b", "beta beta beta", 0.9),
        _result("c", "gamma gamma gamma", 0.5),
    ]
    diversifier = MMRDiversifier(lambda_param=1.0)

    ordered = diversifier.diversify(results)

    assert [result.chunk.chunk_id for result in ordered] == ["b", "c", "a"]


def test_mmr_honors_top_k_and_empty_input() -> None:
    """top_k bounds the output and empty input yields an empty list."""
    results = [
        _result("a", "one two three", 0.9),
        _result("b", "four five six", 0.8),
        _result("c", "seven eight nine", 0.7),
    ]
    diversifier = MMRDiversifier(lambda_param=0.7)

    assert len(diversifier.diversify(results, top_k=2)) == 2
    assert diversifier.diversify([], top_k=5) == []
    assert diversifier.diversify(results, top_k=0) == []


def test_mmr_pure_relevance_orders_overflowing_scores() -> None:
    """Finite mixed-sign scores must sort correctly even when their span overflows."""
    results = [
        _result("low", "low relevance evidence", -1e308),
        _result("middle", "middle relevance evidence", 0.0),
        _result("high", "high relevance evidence", 1e308),
    ]

    ordered = MMRDiversifier(lambda_param=1.0).diversify(results)

    assert [result.chunk.chunk_id for result in ordered] == ["high", "middle", "low"]


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        pytest.param([0.2, 0.9, 0.5], [0.0, 1.0, 3.0 / 7.0], id="ordinary"),
        pytest.param(
            [
                sys.float_info.max / 2.0,
                -sys.float_info.max / 2.0,
                sys.float_info.max,
                -sys.float_info.max,
                0.0,
            ],
            [0.75, 0.25, 1.0, 0.0, 0.5],
            id="mixed-extremes",
        ),
        pytest.param(
            [-sys.float_info.max, -sys.float_info.max / 2.0, sys.float_info.max / 2.0],
            [0.0, 1.0 / 3.0, 1.0],
            id="asymmetric-negative",
        ),
        pytest.param(
            [-sys.float_info.max / 2.0, sys.float_info.max / 2.0, sys.float_info.max],
            [0.0, 2.0 / 3.0, 1.0],
            id="asymmetric-positive",
        ),
        pytest.param(
            [0.0, sys.float_info.max / 2.0, sys.float_info.max],
            [0.0, 0.5, 1.0],
            id="finite-extreme-span",
        ),
        pytest.param(
            [math.nextafter(sys.float_info.max, 0.0), sys.float_info.max],
            [0.0, 1.0],
            id="adjacent-extremes",
        ),
        pytest.param(
            [-sys.float_info.min, 0.0, sys.float_info.min],
            [0.0, 0.5, 1.0],
            id="tiny-normal",
        ),
        pytest.param(
            [-math.ulp(0.0), 0.0, math.ulp(0.0)],
            [0.0, 0.5, 1.0],
            id="subnormal",
        ),
        pytest.param([sys.float_info.max] * 2, [1.0, 1.0], id="equal-maximum"),
        pytest.param([-sys.float_info.max] * 2, [1.0, 1.0], id="equal-minimum"),
        pytest.param([-0.0, 0.0], [1.0, 1.0], id="signed-zero"),
        pytest.param([math.ulp(0.0)] * 2, [1.0, 1.0], id="equal-subnormal"),
        pytest.param([0.0], [1.0], id="single-zero"),
    ],
)
def test_mmr_normalization_preserves_finite_values(
    scores: list[float], expected: list[float]
) -> None:
    original_scores = scores.copy()

    normalized = MMRDiversifier._normalized_relevance(scores)

    assert len(normalized) == len(scores)
    assert all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in normalized)
    assert normalized == pytest.approx(expected, rel=1e-15, abs=0.0)
    assert scores == original_scores


@pytest.mark.parametrize(
    "scores",
    [
        pytest.param([-sys.float_info.max, 1.0, 2.0], id="rounded-high-scores"),
        pytest.param([0.0, math.ulp(0.0), sys.float_info.max], id="rounded-low-scores"),
    ],
)
def test_mmr_pure_relevance_uses_raw_order_when_normalized_scores_tie(
    scores: list[float],
) -> None:
    results = [
        _result(chunk_id, "shared evidence", score)
        for chunk_id, score in zip(("low", "middle", "high"), scores, strict=True)
    ]
    normalized = MMRDiversifier._normalized_relevance(scores)
    assert len(set(normalized)) < len(set(scores))

    ordered = MMRDiversifier(lambda_param=1.0).diversify(results)

    assert [result.chunk.chunk_id for result in ordered] == ["high", "middle", "low"]


@pytest.mark.parametrize("lambda_param", [0.0, 0.5, 0.7])
@pytest.mark.parametrize(
    "scores",
    [
        pytest.param([-sys.float_info.max, 0.0, sys.float_info.max], id="overflowing-span"),
        pytest.param([-sys.float_info.max, 1.0, 2.0], id="rounded-high-scores"),
    ],
)
def test_mmr_starts_with_raw_most_relevant_result(lambda_param: float, scores: list[float]) -> None:
    results = [
        _result(chunk_id, "shared evidence", score)
        for chunk_id, score in zip(("low", "middle", "high"), scores, strict=True)
    ]

    ordered = MMRDiversifier(lambda_param=lambda_param).diversify(results, top_k=1)

    assert len(ordered) == 1
    assert ordered[0] is results[2]


@pytest.mark.parametrize("scale", [1.0, sys.float_info.max], ids=["ordinary", "extreme"])
def test_mmr_keeps_diversity_penalty_with_extreme_scores(scale: float) -> None:
    results = [
        _result("low", "graph neural networks", -scale),
        _result("middle", "transformer attention mechanism", 0.0),
        _result("high", "transformer attention mechanism", scale),
    ]

    ordered = MMRDiversifier(lambda_param=0.5).diversify(results)

    assert [result.chunk.chunk_id for result in ordered] == ["high", "low", "middle"]


@pytest.mark.parametrize("lambda_param", [0.5, 1.0])
@pytest.mark.parametrize("top_k", [None, -1, 0, 1, 2, 4, 10])
def test_mmr_preserves_stable_ties_top_k_and_original_results(
    lambda_param: float, top_k: int | None
) -> None:
    results = [
        _result("low-first", "shared evidence", -sys.float_info.max),
        _result("high-first", "shared evidence", sys.float_info.max),
        _result("middle-first", "shared evidence", 0.0),
        _result("high-second", "shared evidence", sys.float_info.max),
        _result("low-second", "shared evidence", -sys.float_info.max),
        _result("middle-second", "shared evidence", 0.0),
    ]
    for result in results:
        result.path = [result.chunk.document_id, result.chunk.chunk_id]
        result.chunk.metadata = {"section": result.chunk.chunk_id}
    original_results = results.copy()
    original_values = [result.model_dump() for result in results]
    expected_indices = [1, 3, 2, 5, 0, 4]
    if top_k is not None:
        expected_indices = expected_indices[: max(0, top_k)]

    ordered = MMRDiversifier(lambda_param=lambda_param).diversify(results, top_k=top_k)

    assert len(ordered) == len(expected_indices)
    assert all(
        result is original_results[index]
        for result, index in zip(ordered, expected_indices, strict=True)
    )
    assert all(
        result is original for result, original in zip(results, original_results, strict=True)
    )
    assert [result.model_dump() for result in results] == original_values


@pytest.mark.parametrize("lambda_param", [0.0, 0.7, 1.0])
@pytest.mark.parametrize("score", [0.0, math.ulp(0.0), -sys.float_info.max, sys.float_info.max])
def test_mmr_equal_scores_and_similarity_preserve_input_order(
    lambda_param: float, score: float
) -> None:
    results = [_result(chunk_id, "shared evidence", score) for chunk_id in ("b", "a", "c")]

    ordered = MMRDiversifier(lambda_param=lambda_param).diversify(results, top_k=2)

    assert len(ordered) == 2
    assert ordered[0] is results[0]
    assert ordered[1] is results[1]


@pytest.mark.parametrize("lambda_param", [0.0, 0.7, 1.0])
@pytest.mark.parametrize("invalid_score", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("invalid_index", [0, 1, 2])
def test_mmr_rejects_nonfinite_scores_before_ranking(
    lambda_param: float,
    invalid_score: float,
    invalid_index: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = [_result(str(index), "shared evidence", float(index)) for index in range(3)]
    results[invalid_index].score = invalid_score

    def unexpected_tokenize(_text: str) -> list[str]:
        pytest.fail("Non-finite scores must be rejected before tokenizing.")

    monkeypatch.setattr("retrieval.mmr.tokenize", unexpected_tokenize)

    with pytest.raises(ValueError, match="scores must be finite"):
        MMRDiversifier(lambda_param=lambda_param).diversify(results, top_k=1)


@pytest.mark.parametrize("top_k", [-1, 0])
@pytest.mark.parametrize("invalid_score", [math.nan, math.inf, -math.inf])
def test_mmr_empty_selections_skip_score_validation(top_k: int, invalid_score: float) -> None:
    assert MMRDiversifier().diversify([], top_k=top_k) == []
    assert (
        MMRDiversifier().diversify(
            [_result("invalid", "shared evidence", invalid_score)], top_k=top_k
        )
        == []
    )
