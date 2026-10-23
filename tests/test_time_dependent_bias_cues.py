"""Unit tests for TimeDependentBiasCueExtractor."""

from __future__ import annotations

from retrieval.time_dependent_bias_cues import TimeDependentBiasCueExtractor


def test_empty() -> None:
    """Empty input yields empty tuple."""

    assert TimeDependentBiasCueExtractor().extract([]) == ()


def test_flagged() -> None:
    """Matching abstract is flagged."""

    rows = TimeDependentBiasCueExtractor().extract(
        [{"paper_id": "p1", "abstract": "Evidence of time dependent bias in the cohort."}],
    )
    assert rows[0].flagged is True
    assert "time_dependent_bias" in rows[0].cues


def test_unflagged() -> None:
    """Unrelated abstract is not flagged."""

    rows = TimeDependentBiasCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "Routine descriptive statistics only."}],
    )
    assert rows[0].flagged is False
