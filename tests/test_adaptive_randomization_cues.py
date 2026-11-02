"""Tests for AdaptiveRandomizationCueExtractor."""

from __future__ import annotations

from retrieval.adaptive_randomization_cues import AdaptiveRandomizationCueExtractor


def test_empty() -> None:
    """Empty input returns empty."""

    assert AdaptiveRandomizationCueExtractor().extract([]) == ()


def test_detects_cue() -> None:
    """Detects primary cue phrase."""

    rows = AdaptiveRandomizationCueExtractor().extract(
        [
            {
                "paper_id": "p1",
                "abstract": "We discuss adaptive randomization.",
            }
        ]
    )
    assert rows[0].flagged is True
    assert "adaptive_randomization" in rows[0].cues


def test_no_match() -> None:
    """Unrelated text is not flagged."""

    rows = AdaptiveRandomizationCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "No relevant methods described."}]
    )
    assert rows[0].flagged is False
