"""Tests for GComputationCueExtractor."""

from __future__ import annotations

from retrieval.g_computation_cues import GComputationCueExtractor


def test_empty() -> None:
    """Empty input returns empty."""

    assert GComputationCueExtractor().extract([]) == ()


def test_detects_cue() -> None:
    """Detects primary cue phrase."""

    rows = GComputationCueExtractor().extract(
        [{"paper_id": "p1", "abstract": "We discuss g computation in the cohort."}]
    )
    assert rows[0].flagged is True
    assert "g_computation" in rows[0].cues


def test_no_match() -> None:
    """Unrelated text is not flagged."""

    rows = GComputationCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "No relevant methods described."}]
    )
    assert rows[0].flagged is False
