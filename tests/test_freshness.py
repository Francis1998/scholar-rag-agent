"""Tests for publication-recency retrieval boosting."""

from datetime import date, datetime, timezone

import pytest

from retrieval.freshness import FreshnessBooster
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
            text=f"Evidence from {chunk_id}",
            source="test",
            metadata=metadata or {},
        ),
        score=score,
        retriever="bm25",
        path=["hybrid"],
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"half_life_days": 0}, "half_life_days"),
        ({"half_life_days": float("inf")}, "half_life_days"),
        ({"relevance_weight": -0.1}, "relevance_weight"),
        ({"relevance_weight": 1.1}, "relevance_weight"),
    ],
)
def test_freshness_rejects_invalid_configuration(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        FreshnessBooster(**kwargs)


def test_freshness_can_promote_recent_evidence_over_more_relevant_old_evidence() -> None:
    results = [
        _result("old", 1.0, {"year": "2010"}),
        _result("recent", 0.8, {"published_at": "2025-12-31T12:00:00Z"}),
    ]
    booster = FreshnessBooster(
        half_life_days=365,
        relevance_weight=0.25,
        as_of=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    boosted = booster.boost(results)

    assert [result.chunk.chunk_id for result in boosted] == ["recent", "old"]
    assert boosted[0].retriever == "freshness"
    assert boosted[0].path == ["hybrid", "bm25"]
    assert results[0].score == 1.0


def test_freshness_uses_date_fields_in_priority_order_and_falls_back() -> None:
    booster = FreshnessBooster(
        half_life_days=366,
        relevance_weight=0.0,
        as_of=date(2025, 1, 1),
    )
    results = [
        _result(
            "published-at-first",
            1.0,
            {"published_at": "2024-01-01", "year": "2025", "date": "2025-01-01"},
        ),
        _result("invalid-then-year", 1.0, {"published_at": "unknown", "year": "2025"}),
        _result("date-only", 1.0, {"date": "2025-01-01"}),
    ]

    boosted = booster.boost(results)
    by_id = {result.chunk.chunk_id: result.score for result in boosted}

    assert by_id["published-at-first"] == pytest.approx(0.5)
    assert by_id["invalid-then-year"] == pytest.approx(1.0)
    assert by_id["date-only"] == pytest.approx(1.0)
    assert [result.chunk.chunk_id for result in boosted[:2]] == [
        "invalid-then-year",
        "date-only",
    ]


def test_freshness_handles_missing_dates_future_dates_and_output_bounds() -> None:
    booster = FreshnessBooster(
        half_life_days=30,
        relevance_weight=0.0,
        as_of=date(2025, 1, 1),
    )
    results = [
        _result("missing", 0.9),
        _result("malformed", 0.8, {"date": "not-a-date"}),
        _result("future", 0.7, {"date": "2026-01-01"}),
    ]

    boosted = booster.boost(results, top_k=2)

    assert [result.chunk.chunk_id for result in boosted] == ["future", "missing"]
    assert [result.score for result in boosted] == [1.0, 0.0]
    assert booster.boost(results, top_k=0) == []
    assert booster.boost([]) == []
