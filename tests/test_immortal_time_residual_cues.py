"""Tests for ImmortalTimeResidualCueExtractor."""

from __future__ import annotations

from retrieval.immortal_time_residual_cues import ImmortalTimeResidualCueExtractor


def test_empty() -> None:
    """Empty input returns empty."""

    assert ImmortalTimeResidualCueExtractor().extract([]) == ()


def test_detects_cue() -> None:
    """Detects primary cue phrase."""

    rows = ImmortalTimeResidualCueExtractor().extract(
        [{"paper_id": "p1", "abstract": "We discuss immortal time residual in the cohort."}]
    )
    assert rows[0].flagged is True
    assert "immortal_time_residual" in rows[0].cues


def test_no_match() -> None:
    """Unrelated text is not flagged."""

    rows = ImmortalTimeResidualCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "No relevant methods described."}]
    )
    assert rows[0].flagged is False
