"""Unit tests for LateLookerBiasCueExtractor."""

from __future__ import annotations

from retrieval.late_looker_bias_cues import LateLookerBiasCueExtractor


def test_empty() -> None:
    """Empty input yields empty tuple."""

    assert LateLookerBiasCueExtractor().extract([]) == ()


def test_flagged() -> None:
    """Matching abstract is flagged."""

    rows = LateLookerBiasCueExtractor().extract(
        [{"paper_id": "p1", "abstract": "Evidence of late-looker bias in the cohort."}],
    )
    assert rows[0].flagged is True
    assert "late_looker_bias" in rows[0].cues


def test_unflagged() -> None:
    """Unrelated abstract is not flagged."""

    rows = LateLookerBiasCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "Routine descriptive statistics only."}],
    )
    assert rows[0].flagged is False
