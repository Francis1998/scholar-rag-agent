"""Tests for NoveltyBiasCueExtractor."""

from __future__ import annotations

from retrieval.novelty_bias_cues import NoveltyBiasCueExtractor


def test_empty() -> None:
    """Empty input returns empty tuple."""

    assert NoveltyBiasCueExtractor().extract([]) == ()


def test_flagged() -> None:
    """Known cue phrase flags paper."""

    rows = NoveltyBiasCueExtractor().extract(
        [{"paper_id": "p1", "abstract": "Study notes novelty bias in the cohort."}]
    )
    assert len(rows) == 1
    assert rows[0].flagged is True
    assert "novelty_bias" in rows[0].cues


def test_unflagged() -> None:
    """Neutral abstract is not flagged."""

    rows = NoveltyBiasCueExtractor().extract(
        [{"id": "p2", "abstract": "A routine cohort without bias language."}]
    )
    assert rows[0].flagged is False


def test_fallback_id() -> None:
    """Missing ids fall back to paper-index."""

    rows = NoveltyBiasCueExtractor().extract([{"abstract": "novelty bias"}])
    assert rows[0].paper_id == "paper-0"
