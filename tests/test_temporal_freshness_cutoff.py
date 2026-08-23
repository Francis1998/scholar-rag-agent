"""Tests for the deterministic temporal freshness cutoff filter."""

from datetime import UTC, date, datetime

import pytest

from retrieval.models import Chunk, SearchResult
from retrieval.temporal_freshness_cutoff import TemporalFreshnessCutoff


def _result(
    chunk_id: str,
    score: float = 1.0,
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
        ({"max_age_days": 0}, "max_age_days"),
        ({"max_age_days": -1}, "max_age_days"),
        ({"max_age_days": float("inf")}, "max_age_days"),
        ({"max_age_days": float("nan")}, "max_age_days"),
    ],
)
def test_rejects_invalid_max_age_days(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        TemporalFreshnessCutoff(**kwargs)


def test_drops_chunks_older_than_max_age_days() -> None:
    cutoff = TemporalFreshnessCutoff(max_age_days=365, as_of=datetime(2026, 1, 1, tzinfo=UTC))
    results = [
        _result("recent", metadata={"published_at": "2025-06-01T00:00:00Z"}),
        _result("old", metadata={"published_at": "2020-01-01T00:00:00Z"}),
    ]

    filtered = cutoff.filter(results)

    assert [result.chunk.chunk_id for result in filtered] == ["recent"]


def test_keeps_undated_chunks_by_default() -> None:
    cutoff = TemporalFreshnessCutoff(max_age_days=30, as_of=date(2026, 1, 1))
    results = [_result("undated"), _result("old", metadata={"year": "2000"})]

    filtered = cutoff.filter(results)

    assert [result.chunk.chunk_id for result in filtered] == ["undated"]


def test_can_drop_undated_chunks_when_keep_undated_is_false() -> None:
    cutoff = TemporalFreshnessCutoff(max_age_days=30, keep_undated=False, as_of=date(2026, 1, 1))
    results = [_result("undated"), _result("recent", metadata={"year": "2026"})]

    filtered = cutoff.filter(results)

    assert [result.chunk.chunk_id for result in filtered] == ["recent"]


def test_uses_date_fields_in_priority_order() -> None:
    cutoff = TemporalFreshnessCutoff(max_age_days=10, as_of=date(2026, 1, 1))
    results = [
        _result(
            "published-at-wins",
            metadata={"published_at": "2000-01-01", "year": "2026", "date": "2026-01-01"},
        ),
        _result("malformed-then-year", metadata={"published_at": "not-a-date", "year": "2026"}),
        _result("date-only", metadata={"date": "2025-12-31"}),
    ]

    filtered = cutoff.filter(results)

    kept_ids = [result.chunk.chunk_id for result in filtered]
    assert kept_ids == ["malformed-then-year", "date-only"]


def test_malformed_date_is_treated_as_undated() -> None:
    cutoff = TemporalFreshnessCutoff(max_age_days=1, as_of=date(2026, 1, 1))
    results = [_result("malformed", metadata={"date": "not-a-date"})]

    filtered = cutoff.filter(results)

    assert [result.chunk.chunk_id for result in filtered] == ["malformed"]


def test_boundary_age_is_inclusive_and_future_dates_are_kept() -> None:
    cutoff = TemporalFreshnessCutoff(max_age_days=365, as_of=date(2026, 1, 1))
    results = [
        _result("exact-boundary", metadata={"date": "2025-01-01"}),
        _result("future", metadata={"date": "2027-01-01"}),
    ]

    filtered = cutoff.filter(results)

    assert [result.chunk.chunk_id for result in filtered] == ["exact-boundary", "future"]


def test_preserves_scores_order_and_object_identity() -> None:
    cutoff = TemporalFreshnessCutoff(max_age_days=100, as_of=date(2026, 1, 1))
    keeper = _result("keeper", score=0.42, metadata={"year": "2026"})
    results = [keeper, _result("dropped", metadata={"year": "2000"})]

    filtered = cutoff.filter(results)

    assert filtered == [keeper]
    assert filtered[0] is keeper
    assert filtered[0].score == 0.42


def test_empty_input_returns_empty_list() -> None:
    cutoff = TemporalFreshnessCutoff(max_age_days=30)

    assert cutoff.filter([]) == []
