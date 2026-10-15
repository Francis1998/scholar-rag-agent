"""Tests for IntervalCensoringCueExtractor."""

from __future__ import annotations

from retrieval.interval_censoring_cues import IntervalCensoringCueExtractor


def test_empty() -> None:
    """Empty input returns empty tuple."""

    assert IntervalCensoringCueExtractor().extract([]) == ()


def test_flags_match() -> None:
    """Matching abstract is flagged."""

    rows = [
        {
            "paper_id": "p1",
            "abstract": "We discuss interval censoring in the survival analysis.",
        }
    ]
    out = IntervalCensoringCueExtractor().extract(rows)
    assert len(out) == 1
    assert out[0].flagged is True
    assert out[0].cue_kind == "interval_censoring"


def test_no_match() -> None:
    """Unrelated abstract is not flagged."""

    rows = [{"paper_id": "p2", "abstract": "No relevant bias language here."}]
    out = IntervalCensoringCueExtractor().extract(rows)
    assert out[0].flagged is False
