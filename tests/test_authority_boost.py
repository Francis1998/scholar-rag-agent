"""Tests for authority boost postprocessing."""

import pytest

from retrieval.authority_boost import AuthorityBooster
from retrieval.models import Chunk, SearchResult


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


@pytest.mark.parametrize("alpha", [-0.1, 1.1, float("nan"), float("inf")])
def test_rejects_invalid_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        AuthorityBooster(alpha=alpha)


def test_accepts_alpha_bounds() -> None:
    AuthorityBooster(alpha=0.0)
    AuthorityBooster(alpha=1.0)


def test_prefers_high_source_authority() -> None:
    weak = _result("weak", score=1.0, metadata={"source_authority": "0.1"})
    strong = _result("strong", score=0.0, metadata={"source_authority": "0.9"})
    boosted = AuthorityBooster(alpha=1.0).boost([weak, strong])
    assert [item.chunk.chunk_id for item in boosted] == ["strong", "weak"]
    assert boosted[0].retriever == "authority_boost"
    assert boosted[0].path == ["hybrid", "bm25"]
    assert weak.score == 1.0


def test_missing_metadata_is_neutral_half() -> None:
    result = _result("only", score=0.0, metadata={})
    boosted = AuthorityBooster(alpha=1.0).boost([result])
    assert boosted[0].score == pytest.approx(0.5)


def test_venue_rank_one_is_full_authority() -> None:
    result = _result("r1", score=0.0, metadata={"venue_rank": "1"})
    boosted = AuthorityBooster(alpha=1.0).boost([result])
    assert boosted[0].score == pytest.approx(1.0)


def test_venue_rank_decay() -> None:
    result = _result("r3", score=0.0, metadata={"venue_rank": "3"})
    boosted = AuthorityBooster(alpha=1.0).boost([result])
    assert boosted[0].score == pytest.approx(0.8)


def test_peer_reviewed_and_impact_factor_mean() -> None:
    result = _result(
        "combo",
        score=0.0,
        metadata={"is_peer_reviewed": "true", "impact_factor": "high"},
    )
    boosted = AuthorityBooster(alpha=1.0).boost([result])
    assert boosted[0].score == pytest.approx(1.0)


def test_impact_factor_numeric_buckets() -> None:
    high = _result("h", 0.0, {"impact_factor": "12.5"})
    mid = _result("m", 0.0, {"impact_factor": "4"})
    low = _result("l", 0.0, {"impact_factor": "1.2"})
    booster = AuthorityBooster(alpha=1.0)
    assert booster.boost([high])[0].score == pytest.approx(1.0)
    assert booster.boost([mid])[0].score == pytest.approx(0.65)
    assert booster.boost([low])[0].score == pytest.approx(0.35)


def test_source_authority_takes_precedence() -> None:
    result = _result(
        "direct",
        score=0.0,
        metadata={
            "source_authority": "0.25",
            "venue_rank": "1",
            "is_peer_reviewed": "true",
            "impact_factor": "high",
        },
    )
    boosted = AuthorityBooster(alpha=1.0).boost([result])
    assert boosted[0].score == pytest.approx(0.25)


def test_blend_formula() -> None:
    result = _result("only", score=0.8, metadata={"source_authority": "1.0"})
    boosted = AuthorityBooster(alpha=0.3).boost([result])
    assert boosted[0].score == pytest.approx((1 - 0.3) * 0.8 + 0.3 * 1.0)


def test_does_not_mutate_inputs() -> None:
    original = _result("a", score=0.9, metadata={"source_authority": "0.2"})
    snapshot = original.score
    AuthorityBooster(alpha=1.0).boost([original])
    assert original.score == snapshot
    assert original.retriever == "bm25"


def test_stable_sort_for_tied_scores() -> None:
    first = _result("first", score=0.5, metadata={})
    second = _result("second", score=0.5, metadata={})
    boosted = AuthorityBooster(alpha=0.0).boost([first, second])
    assert [item.chunk.chunk_id for item in boosted] == ["first", "second"]


def test_top_k_and_empty() -> None:
    booster = AuthorityBooster(alpha=1.0)
    assert booster.boost([]) == []
    assert booster.boost([_result("a", 0.5, {})], top_k=0) == []
    rows = [
        _result("a", 0.0, {"source_authority": "0.9"}),
        _result("b", 0.0, {"source_authority": "0.1"}),
    ]
    assert len(booster.boost(rows, top_k=1)) == 1
    assert booster.boost(rows, top_k=1)[0].chunk.chunk_id == "a"


def test_false_peer_reviewed_is_soft_demotion() -> None:
    result = _result("pre", score=0.0, metadata={"is_peer_reviewed": "false"})
    boosted = AuthorityBooster(alpha=1.0).boost([result])
    assert boosted[0].score == pytest.approx(0.2)


def test_invalid_source_authority_falls_through_to_neutral() -> None:
    result = _result("bad", score=0.0, metadata={"source_authority": "not-a-float"})
    boosted = AuthorityBooster(alpha=1.0).boost([result])
    assert boosted[0].score == pytest.approx(0.5)
