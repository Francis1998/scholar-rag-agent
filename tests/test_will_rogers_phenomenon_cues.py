"""Tests for WillRogersPhenomenonCueExtractor."""

from __future__ import annotations

from retrieval.will_rogers_phenomenon_cues import WillRogersPhenomenonCueExtractor


def test_empty() -> None:
    """Empty input returns empty tuple."""

    assert WillRogersPhenomenonCueExtractor().extract([]) == ()


def test_flagged() -> None:
    """Known cue phrase flags paper."""

    rows = WillRogersPhenomenonCueExtractor().extract(
        [{"paper_id": "p1", "abstract": "Authors describe will rogers phenomenon after restaging."}]
    )
    assert len(rows) == 1
    assert rows[0].flagged is True
    assert "will_rogers_phenomenon" in rows[0].cues


def test_unflagged() -> None:
    """Neutral abstract is not flagged."""

    rows = WillRogersPhenomenonCueExtractor().extract(
        [{"id": "p2", "abstract": "A routine cohort without bias language."}]
    )
    assert rows[0].flagged is False


def test_fallback_id() -> None:
    """Missing ids fall back to paper-index."""

    rows = WillRogersPhenomenonCueExtractor().extract(
        [{"abstract": "Authors describe will rogers phenomenon after restaging."}]
    )
    assert rows[0].paper_id == "paper-0"
