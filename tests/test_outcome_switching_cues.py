"""Tests for OutcomeSwitchingCueExtractor."""

from __future__ import annotations

from retrieval.outcome_switching_cues import OutcomeSwitchingCueExtractor


def test_empty() -> None:
    """Empty input returns empty."""

    assert OutcomeSwitchingCueExtractor().extract([]) == ()


def test_detects_cue() -> None:
    """Detects primary cue phrase."""

    rows = OutcomeSwitchingCueExtractor().extract(
        [{"paper_id": "p1", "abstract": "We discuss outcome switching in the cohort."}]
    )
    assert rows[0].flagged is True
    assert "outcome_switching" in rows[0].cues


def test_no_match() -> None:
    """Unrelated text is not flagged."""

    rows = OutcomeSwitchingCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "No relevant methods described."}]
    )
    assert rows[0].flagged is False
