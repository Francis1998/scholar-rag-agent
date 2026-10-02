"""Unit tests for InterruptedTimeSeriesCueExtractor."""

from __future__ import annotations

from retrieval.interrupted_time_series_cues import InterruptedTimeSeriesCueExtractor


def test_empty() -> None:
    """Empty input returns empty tuple."""

    assert InterruptedTimeSeriesCueExtractor().extract([]) == ()


def test_flagged() -> None:
    """Matching methods text is flagged."""

    rows = InterruptedTimeSeriesCueExtractor().extract(
        [
            {
                "paper_id": "p1",
                "methods": (
                    "An interrupted time series showed a "
                    "level change and slope change after intervention."
                ),
            }
        ]
    )
    assert rows[0].flagged is True
    assert rows[0].cues


def test_unflagged() -> None:
    """Unrelated text is not flagged."""

    rows = InterruptedTimeSeriesCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "A narrative review of nursing education."}]
    )
    assert rows[0].flagged is False


def test_id_fallback() -> None:
    """Missing ids fall back to paper-index."""

    rows = InterruptedTimeSeriesCueExtractor().extract([{"title": "x"}])
    assert rows[0].paper_id == "paper-0"
