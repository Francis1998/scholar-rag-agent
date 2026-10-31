"""Tests for CompetingRisksResidualCueExtractor."""

from __future__ import annotations

from retrieval.competing_risks_residual_cues import CompetingRisksResidualCueExtractor


def test_empty() -> None:
    """Empty input returns empty."""

    assert CompetingRisksResidualCueExtractor().extract([]) == ()


def test_detects_cue() -> None:
    """Detects primary cue phrase."""

    rows = CompetingRisksResidualCueExtractor().extract(
        [{"paper_id": "p1", "abstract": "We discuss competing risks residual in the cohort."}]
    )
    assert rows[0].flagged is True
    assert "competing_risks_residual" in rows[0].cues


def test_no_match() -> None:
    """Unrelated text is not flagged."""

    rows = CompetingRisksResidualCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "No relevant methods described."}]
    )
    assert rows[0].flagged is False
