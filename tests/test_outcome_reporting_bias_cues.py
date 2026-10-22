"""Unit tests for OutcomeReportingBiasCueExtractor."""

from __future__ import annotations

from retrieval.outcome_reporting_bias_cues import OutcomeReportingBiasCueExtractor


def test_empty() -> None:
    """Empty input yields empty tuple."""

    assert OutcomeReportingBiasCueExtractor().extract([]) == ()


def test_flagged() -> None:
    """Matching abstract is flagged."""

    rows = OutcomeReportingBiasCueExtractor().extract(
        [{"paper_id": "p1", "abstract": "Evidence of outcome reporting bias in the cohort."}],
    )
    assert rows[0].flagged is True
    assert "outcome_reporting_bias" in rows[0].cues


def test_unflagged() -> None:
    """Unrelated abstract is not flagged."""

    rows = OutcomeReportingBiasCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "Routine descriptive statistics only."}],
    )
    assert rows[0].flagged is False
