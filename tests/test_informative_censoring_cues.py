"""Tests for InformativeCensoringCueExtractor."""

from __future__ import annotations

from retrieval.informative_censoring_cues import InformativeCensoringCueExtractor


def test_empty() -> None:
    """Empty input returns empty."""

    assert InformativeCensoringCueExtractor().extract([]) == ()


def test_detects_cue() -> None:
    """Detects primary cue phrase."""

    rows = InformativeCensoringCueExtractor().extract(
        [{"paper_id": "p1", "abstract": "We discuss informative censoring in the cohort."}]
    )
    assert rows[0].flagged is True
    assert "informative_censoring" in rows[0].cues


def test_no_match() -> None:
    """Unrelated text is not flagged."""

    rows = InformativeCensoringCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "No relevant methods described."}]
    )
    assert rows[0].flagged is False
