"""Tests for FollowUpDurationCueExtractor."""

from __future__ import annotations

from retrieval.follow_up_duration_cues import FollowUpDurationCueExtractor


def test_extracts_cue() -> None:
    rows = FollowUpDurationCueExtractor().extract(
        [{"paper_id": "p1", "abstract": "Median follow-up was 24 months."}]
    )
    assert rows[0].flagged is True
    assert rows[0].duration_kind is not None
    assert "months_follow_up" in rows[0].cues or rows[0].cues


def test_empty_papers() -> None:
    assert FollowUpDurationCueExtractor().extract([]) == ()


def test_unflagged_without_match() -> None:
    rows = FollowUpDurationCueExtractor().extract(
        [{"id": "x", "abstract": "No relevant methods text."}]
    )
    assert rows[0].flagged is False
