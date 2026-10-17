"""Tests for ChannelingBiasCueExtractor."""

from __future__ import annotations

from retrieval.channeling_bias_cues import ChannelingBiasCueExtractor


def test_empty() -> None:
    """Empty input returns empty tuple."""

    assert ChannelingBiasCueExtractor().extract([]) == ()


def test_flagged() -> None:
    """Known cue phrase flags paper."""

    rows = ChannelingBiasCueExtractor().extract(
        [{"paper_id": "p1", "abstract": "Study notes channeling bias in the cohort."}]
    )
    assert len(rows) == 1
    assert rows[0].flagged is True
    assert "channeling_bias" in rows[0].cues


def test_unflagged() -> None:
    """Neutral abstract is not flagged."""

    rows = ChannelingBiasCueExtractor().extract(
        [{"id": "p2", "abstract": "A routine cohort without bias language."}]
    )
    assert rows[0].flagged is False


def test_fallback_id() -> None:
    """Missing ids fall back to paper-index."""

    rows = ChannelingBiasCueExtractor().extract([{"abstract": "channeling bias"}])
    assert rows[0].paper_id == "paper-0"
