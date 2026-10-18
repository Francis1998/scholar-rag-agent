"""Tests for IncorporationBiasCueExtractor."""

from __future__ import annotations

from retrieval.incorporation_bias_cues import IncorporationBiasCueExtractor


def test_empty() -> None:
    """Empty input returns empty tuple."""

    assert IncorporationBiasCueExtractor().extract([]) == ()


def test_flagged() -> None:
    """Known cue phrase flags paper."""

    rows = IncorporationBiasCueExtractor().extract(
        [{"paper_id": "p1", "abstract": "Methods note incorporation bias from reference standard."}]
    )
    assert len(rows) == 1
    assert rows[0].flagged is True
    assert "incorporation_bias" in rows[0].cues


def test_unflagged() -> None:
    """Neutral abstract is not flagged."""

    rows = IncorporationBiasCueExtractor().extract(
        [{"id": "p2", "abstract": "A routine cohort without bias language."}]
    )
    assert rows[0].flagged is False


def test_fallback_id() -> None:
    """Missing ids fall back to paper-index."""

    rows = IncorporationBiasCueExtractor().extract(
        [{"abstract": "Methods note incorporation bias from reference standard."}]
    )
    assert rows[0].paper_id == "paper-0"
