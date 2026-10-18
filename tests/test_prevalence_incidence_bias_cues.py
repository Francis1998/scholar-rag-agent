"""Tests for PrevalenceIncidenceBiasCueExtractor."""

from __future__ import annotations

from retrieval.prevalence_incidence_bias_cues import PrevalenceIncidenceBiasCueExtractor


def test_empty() -> None:
    """Empty input returns empty tuple."""

    assert PrevalenceIncidenceBiasCueExtractor().extract([]) == ()


def test_flagged() -> None:
    """Known cue phrase flags paper."""

    rows = PrevalenceIncidenceBiasCueExtractor().extract(
        [{"paper_id": "p1", "abstract": "Study discusses prevalence-incidence bias in the cohort."}]
    )
    assert len(rows) == 1
    assert rows[0].flagged is True
    assert "prevalence_incidence_bias" in rows[0].cues


def test_unflagged() -> None:
    """Neutral abstract is not flagged."""

    rows = PrevalenceIncidenceBiasCueExtractor().extract(
        [{"id": "p2", "abstract": "A routine cohort without bias language."}]
    )
    assert rows[0].flagged is False


def test_fallback_id() -> None:
    """Missing ids fall back to paper-index."""

    rows = PrevalenceIncidenceBiasCueExtractor().extract(
        [{"abstract": "Study discusses prevalence-incidence bias in the cohort."}]
    )
    assert rows[0].paper_id == "paper-0"
