"""Unit tests for LengthTimeBiasCueExtractor."""

from __future__ import annotations

from retrieval.length_time_bias_cues import LengthTimeBiasCueExtractor


def test_empty() -> None:
    """Empty input yields empty tuple."""

    assert LengthTimeBiasCueExtractor().extract([]) == ()


def test_flagged() -> None:
    """Matching abstract is flagged."""

    rows = LengthTimeBiasCueExtractor().extract(
        [{"paper_id": "p1", "abstract": "Evidence of length time bias in the cohort."}],
    )
    assert rows[0].flagged is True
    assert "length_time_bias" in rows[0].cues


def test_unflagged() -> None:
    """Unrelated abstract is not flagged."""

    rows = LengthTimeBiasCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "Routine descriptive statistics only."}],
    )
    assert rows[0].flagged is False
