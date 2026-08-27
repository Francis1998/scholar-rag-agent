"""Tests for venue-tier retrieval boosting."""

import pytest

from retrieval.models import Chunk, SearchResult
from retrieval.venue_tier_boost import VenueTierBooster


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


@pytest.mark.parametrize(
    "alpha",
    [-0.1, 1.1, float("nan"), float("inf")],
)
def test_rejects_invalid_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        VenueTierBooster(alpha=alpha)


def test_rejects_invalid_venue_tier_labels() -> None:
    with pytest.raises(ValueError, match="venue_tiers"):
        VenueTierBooster(venue_tiers={"Fancy Journal": "tier9"})


def test_boost_prefers_tier1_venues() -> None:
    unknown = _result("unknown", score=1.0, metadata={"venue": "obscure letters"})
    nature = _result("nature", score=0.0, metadata={"venue": "Nature"})
    booster = VenueTierBooster(alpha=1.0)

    boosted = booster.boost([unknown, nature])

    assert [result.chunk.chunk_id for result in boosted] == ["nature", "unknown"]
    assert boosted[0].score == pytest.approx(1.0)
    assert boosted[1].score == pytest.approx(0.2)
    assert boosted[0].retriever == "venue_tier_boost"
    assert boosted[0].path == ["hybrid", "bm25"]
    assert unknown.score == 1.0  # originals not mutated


def test_builtin_prestige_list_covers_core_venues() -> None:
    venues = ["Science", "Cell", "NEJM", "Lancet", "JAMA"]
    results = [_result(name, score=0.0, metadata={"journal": name}) for name in venues]
    booster = VenueTierBooster(alpha=1.0)

    boosted = booster.boost(results)

    assert all(result.score == pytest.approx(1.0) for result in boosted)


def test_explicit_venue_tier_metadata() -> None:
    tier2 = _result("t2", score=0.0, metadata={"venue_tier": "tier2"})
    tier3 = _result("t3", score=0.0, metadata={"venue_tier": "tier3"})
    booster = VenueTierBooster(alpha=1.0)

    boosted = booster.boost([tier3, tier2])

    by_id = {result.chunk.chunk_id: result.score for result in boosted}
    assert by_id["t2"] == pytest.approx(0.7)
    assert by_id["t3"] == pytest.approx(0.4)


def test_custom_venue_tiers_overlay_builtin() -> None:
    result = _result("custom", score=0.0, metadata={"venue": "My Prestige Journal"})
    booster = VenueTierBooster(
        alpha=1.0,
        venue_tiers={"My Prestige Journal": "tier2"},
    )

    boosted = booster.boost([result])

    assert boosted[0].score == pytest.approx(0.7)


def test_formula_matches_one_minus_alpha_old_plus_alpha_tier() -> None:
    result = _result("only", score=0.8, metadata={"venue": "Nature"})
    booster = VenueTierBooster(alpha=0.3)

    boosted = booster.boost([result])

    assert boosted[0].score == pytest.approx((1 - 0.3) * 0.8 + 0.3 * 1.0)


def test_stable_ordering_for_tied_scores() -> None:
    first = _result("first", score=0.5, metadata={"venue_tier": "tier2"})
    second = _result("second", score=0.5, metadata={"venue_tier": "tier2"})
    booster = VenueTierBooster(alpha=0.0)

    boosted = booster.boost([first, second])

    assert [result.chunk.chunk_id for result in boosted] == ["first", "second"]


def test_top_k_truncates_after_sort() -> None:
    results = [
        _result("a", score=0.9, metadata={"venue": "obscure"}),
        _result("b", score=0.1, metadata={"venue": "Nature"}),
        _result("c", score=0.2, metadata={"venue_tier": "tier2"}),
    ]
    booster = VenueTierBooster(alpha=1.0)

    boosted = booster.boost(results, top_k=2)

    assert len(boosted) == 2
    assert boosted[0].chunk.chunk_id == "b"


def test_empty_input_returns_empty() -> None:
    assert VenueTierBooster().boost([]) == []
