"""Tests for HawthorneEffectCueExtractor."""

from __future__ import annotations

from retrieval.hawthorne_effect_cues import HawthorneEffectCueExtractor


def test_empty() -> None:
    """Empty input returns empty tuple."""

    assert HawthorneEffectCueExtractor().extract([]) == ()


def test_flagged() -> None:
    """Known cue phrase flags paper."""

    rows = HawthorneEffectCueExtractor().extract(
        [{"paper_id": "p1", "abstract": "Study notes hawthorne effect in the cohort."}]
    )
    assert len(rows) == 1
    assert rows[0].flagged is True
    assert "hawthorne_effect" in rows[0].cues


def test_unflagged() -> None:
    """Neutral abstract is not flagged."""

    rows = HawthorneEffectCueExtractor().extract(
        [{"id": "p2", "abstract": "A routine cohort without bias language."}]
    )
    assert rows[0].flagged is False


def test_fallback_id() -> None:
    """Missing ids fall back to paper-index."""

    rows = HawthorneEffectCueExtractor().extract([{"abstract": "hawthorne effect"}])
    assert rows[0].paper_id == "paper-0"
