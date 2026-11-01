"""Tests for ColliderBiasOpeningCueExtractor."""

from __future__ import annotations

from retrieval.collider_bias_opening_cues import ColliderBiasOpeningCueExtractor


def test_empty() -> None:
    """Empty input returns empty."""

    assert ColliderBiasOpeningCueExtractor().extract([]) == ()


def test_detects_cue() -> None:
    """Detects primary cue phrase."""

    rows = ColliderBiasOpeningCueExtractor().extract(
        [{"paper_id": "p1", "abstract": "We discuss collider bias opening in the cohort."}]
    )
    assert rows[0].flagged is True
    assert "collider_bias_opening" in rows[0].cues


def test_no_match() -> None:
    """Unrelated text is not flagged."""

    rows = ColliderBiasOpeningCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "No relevant methods described."}]
    )
    assert rows[0].flagged is False
