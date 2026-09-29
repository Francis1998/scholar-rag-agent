"""Unit tests for CompetingRiskCueExtractor."""

from __future__ import annotations

from retrieval.competing_risk_cues import CompetingRiskCueExtractor


def test_empty_papers() -> None:
    """Empty input returns empty tuple."""

    assert CompetingRiskCueExtractor().extract([]) == ()


def test_flagged_match() -> None:
    """Matching methods text is flagged."""

    rows = CompetingRiskCueExtractor().extract(
        [
            {
                "paper_id": "p1",
                "methods": "We used a Fine-Gray competing risk model for the cumulative incidence.",
            }
        ]
    )
    assert rows[0].flagged is True
    assert "competing_risk" in rows[0].cues


def test_no_match() -> None:
    """Unrelated text is not flagged."""

    rows = CompetingRiskCueExtractor().extract(
        [{"paper_id": "p1", "abstract": "No relevant design cues."}]
    )
    assert rows[0].flagged is False


def test_resolves_doi() -> None:
    """Falls back to doi when paper_id missing."""

    rows = CompetingRiskCueExtractor().extract(
        [
            {
                "doi": "10.1/x",
                "methods": "We used a Fine-Gray competing risk model for the cumulative incidence.",
            }
        ]
    )
    assert rows[0].paper_id == "10.1/x"
