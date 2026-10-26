"""Unit tests for CompetingRiskInformativeCensoringCueExtractor."""

from __future__ import annotations

from retrieval.competing_risk_informative_censoring_cues import (
    CompetingRiskInformativeCensoringCueExtractor,
)


def test_empty() -> None:
    """Empty input yields empty tuple."""

    assert CompetingRiskInformativeCensoringCueExtractor().extract([]) == ()


def test_flagged() -> None:
    """Matching abstract is flagged."""

    abstract = "Competing-risk informative censoring may bias the cumulative incidence."
    rows = CompetingRiskInformativeCensoringCueExtractor().extract(
        [{"paper_id": "p1", "abstract": abstract}],
    )
    assert rows[0].flagged is True
    assert "competing_risk_informative_censoring" in rows[0].cues


def test_unflagged() -> None:
    """Unrelated abstract is not flagged."""

    rows = CompetingRiskInformativeCensoringCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "Routine descriptive statistics only."}],
    )
    assert rows[0].flagged is False
