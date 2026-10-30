"""Tests for MendelianRandomizationWeakCueExtractor."""

from __future__ import annotations

from retrieval.mendelian_randomization_weak_cues import MendelianRandomizationWeakCueExtractor


def test_empty() -> None:
    """Empty input returns empty."""

    assert MendelianRandomizationWeakCueExtractor().extract([]) == ()


def test_detects_cue() -> None:
    """Detects primary cue phrase."""

    rows = MendelianRandomizationWeakCueExtractor().extract(
        [{"paper_id": "p1", "abstract": "We discuss weak instrument in the cohort."}]
    )
    assert rows[0].flagged is True
    assert "weak_instrument" in rows[0].cues


def test_no_match() -> None:
    """Unrelated text is not flagged."""

    rows = MendelianRandomizationWeakCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "No relevant methods described."}]
    )
    assert rows[0].flagged is False
