"""RRF counts input rankings, not repeated occurrences of a chunk."""

from collections.abc import Callable
from copy import deepcopy

import pytest

from retrieval.models import Chunk, SearchResult
from retrieval.reciprocal_rank_fusion_gate import ReciprocalRankFusionGate
from retrieval.rrf import reciprocal_rank_fusion

Fusion = Callable[[list[list[SearchResult]], int], list[SearchResult]]


def _result(chunk_id: str, score: float = 1.0, retriever: str = "dense") -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id="doc-1",
            title="Paper",
            text="Identical scientific evidence.",
            source="fixture",
            metadata={"origin": retriever, "citation_key": "fixture-1"},
        ),
        score=score,
        retriever=retriever,
        path=["embedding", f"input:{retriever}"],
    )


@pytest.fixture(params=["helper", "gate"])
def fuse(request: pytest.FixtureRequest) -> Fusion:
    if request.param == "helper":
        return lambda rankings, k: reciprocal_rank_fusion(rankings, rank_constant=k)
    return lambda rankings, k: ReciprocalRankFusionGate(k=k).gate(rankings)


@pytest.mark.parametrize("rank_constant", [10, 60])
def test_single_repeated_hit_contributes_once(fuse: Fusion, rank_constant: int) -> None:
    hit = _result("shared")

    fused = fuse([[hit, hit]], rank_constant)

    assert len(fused) == 1
    assert fused[0].score == pytest.approx(1 / (rank_constant + 1))
    assert fused[0].path == ["dense"]


def test_nonadjacent_duplicates_use_first_original_rank_not_highest_score(fuse: Fusion) -> None:
    first = _result("shared", score=0.1)
    duplicate = _result("shared", score=100, retriever="duplicate")
    rankings = [[_result("ahead"), first, duplicate, _result("other"), duplicate]]

    fused = fuse(rankings, 60)
    shared = next(result for result in fused if result.chunk.chunk_id == "shared")

    assert shared.score == pytest.approx(1 / 62)
    assert shared.path == ["dense"]


def test_duplicates_do_not_compact_later_ranks(fuse: Fusion) -> None:
    hit = _result("a")
    fused = fuse([[hit, hit, _result("b"), hit, _result("c")]], 60)
    scores = {result.chunk.chunk_id: result.score for result in fused}

    assert scores["b"] == pytest.approx(1 / 63)
    assert scores["c"] == pytest.approx(1 / 65)


@pytest.mark.parametrize("repetitions", [2, 100])
def test_repetition_cannot_outvote_cross_list_support(fuse: Fusion, repetitions: int) -> None:
    repeated = _result("repeated", score=100)
    shared = _result("shared", score=0.1)
    rankings = [[*[repeated] * repetitions, shared], [_result("shared", retriever="bm25")]]

    fused = fuse(rankings, 60)

    assert [result.chunk.chunk_id for result in fused] == ["shared", "repeated"]
    assert fused[0].score == pytest.approx(1 / (60 + repetitions + 1) + 1 / 61)
    assert fused[0].path == ["dense", "bm25"]
    assert fused[1].score == pytest.approx(1 / 61)
    assert fused[1].path == ["dense"]


def test_each_list_votes_even_with_identical_source_labels(fuse: Fusion) -> None:
    shared = _result("shared")

    fused = fuse([[shared], [_result("other"), shared]], 60)

    assert [result.chunk.chunk_id for result in fused] == ["shared", "other"]
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert fused[0].path == ["dense", "dense"]


def test_first_metadata_provenance_and_inputs_are_preserved(fuse: Fusion) -> None:
    first = _result("shared", score=0.1)
    duplicate = _result("shared", score=100, retriever="duplicate")
    duplicate.chunk.document_id = "doc-2"
    duplicate.chunk.title = "Later metadata"
    duplicate.chunk.text = "Different text for the same exact ID."
    duplicate.chunk.source = "later-source"
    second_list_hit = _result("shared", score=2, retriever="bm25")
    rankings = [[first, duplicate], [second_list_hit, duplicate]]
    before = deepcopy(rankings)

    fused = fuse(rankings, 60)

    assert rankings == before
    assert len(fused) == 1
    assert fused[0] is not first
    assert fused[0].chunk == first.chunk
    assert fused[0].path == ["dense", "bm25"]
    assert fused[0].path is not first.path
    assert fused[0].score == pytest.approx(2 / 61)


def test_identical_text_with_distinct_exact_ids_stays_distinct(fuse: Fusion) -> None:
    rankings = [[_result("a"), _result("A"), _result("a ")]]

    fused = fuse(rankings, 60)

    assert [result.chunk for result in fused] == [result.chunk for result in rankings[0]]
    assert [result.score for result in fused] == pytest.approx([1 / 61, 1 / 62, 1 / 63])


def test_unique_rankings_keep_first_seen_ties_and_source_order(fuse: Fusion) -> None:
    rankings = [
        [_result("b", score=-100, retriever="dense"), _result("a", score=100)],
        [_result("a", retriever="bm25"), _result("b", retriever="bm25")],
        [],
        [_result("c", retriever="graph")],
    ]
    before = deepcopy(rankings)

    fused = fuse(rankings, 60)

    assert rankings == before
    assert [result.chunk.chunk_id for result in fused] == ["b", "a", "c"]
    assert [result.score for result in fused] == pytest.approx(
        [1 / 61 + 1 / 62, 1 / 61 + 1 / 62, 1 / 61]
    )
    assert [result.path for result in fused] == [["dense", "bm25"], ["dense", "bm25"], ["graph"]]


@pytest.mark.parametrize("rankings", [[], [[]], [[], []]])
def test_empty_rankings(fuse: Fusion, rankings: list[list[SearchResult]]) -> None:
    assert fuse(rankings, 60) == []


@pytest.mark.parametrize(
    ("limit", "expected_ids"),
    [(0, []), (1, ["a"]), (2, ["a", "b"]), (10, ["a", "b", "c"]), (-1, ["a", "b"]), (-10, [])],
)
def test_helper_retains_limit_slicing(limit: int, expected_ids: list[str]) -> None:
    hit = _result("a")

    fused = reciprocal_rank_fusion([[hit, hit, _result("b"), _result("c")]], limit=limit)

    assert [result.chunk.chunk_id for result in fused] == expected_ids
    assert all(result.retriever == "rrf" for result in fused)


@pytest.mark.parametrize("use_gate", [False, True], ids=["helper", "gate"])
def test_limit_applies_after_duplicate_safe_fusion(use_gate: bool) -> None:
    repeated = _result("repeated")
    rankings = [
        [repeated, repeated, _result("shared")],
        [_result("other", retriever="bm25"), _result("shared", retriever="bm25")],
    ]

    fused = (
        ReciprocalRankFusionGate().gate(rankings, top_k=1)
        if use_gate
        else reciprocal_rank_fusion(rankings, limit=1)
    )

    assert [result.chunk.chunk_id for result in fused] == ["shared"]
    assert fused[0].score == pytest.approx(1 / 63 + 1 / 62)
    assert fused[0].path == ["dense", "bm25"]


def test_default_limits_stay_distinct() -> None:
    rankings = [[_result(f"chunk-{index}") for index in range(12)]]

    assert len(reciprocal_rank_fusion(rankings)) == 10
    assert len(ReciprocalRankFusionGate().gate(rankings)) == 12


@pytest.mark.parametrize(
    ("top_k", "expected_count"), [(None, 12), (0, 0), (1, 1), (2, 2), (20, 12), (-1, 0), (-100, 0)]
)
def test_gate_preserves_unlimited_and_nonpositive_caps(
    top_k: int | None, expected_count: int
) -> None:
    hits = [_result(f"chunk-{index}") for index in range(12)]
    rankings = [[hits[0], *hits]]

    fused = ReciprocalRankFusionGate().gate(rankings, top_k=top_k)

    assert [result.chunk.chunk_id for result in fused] == [
        result.chunk.chunk_id for result in hits[:expected_count]
    ]
    assert all(result.retriever == "reciprocal_rank_fusion_gate" for result in fused)
