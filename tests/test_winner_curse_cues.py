"""Tests for WinnerCurseCueExtractor."""

from __future__ import annotations

from retrieval.winner_curse_cues import WinnerCurseCueExtractor


def test_empty() -> None:
    """Empty input returns empty."""

    assert WinnerCurseCueExtractor().extract([]) == ()


def test_detects_cue() -> None:
    """Detects primary cue phrase."""

    rows = WinnerCurseCueExtractor().extract(
        [{"paper_id": "p1", "abstract": "We discuss winner curse in the cohort."}]
    )
    assert rows[0].flagged is True
    assert "winner_curse" in rows[0].cues


def test_no_match() -> None:
    """Unrelated text is not flagged."""

    rows = WinnerCurseCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "No relevant methods described."}]
    )
    assert rows[0].flagged is False
