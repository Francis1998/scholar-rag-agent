"""Unit tests for MediationAnalysisCueExtractor."""

from __future__ import annotations

from retrieval.mediation_analysis_cues import MediationAnalysisCueExtractor


def test_empty_papers() -> None:
    """Empty input returns empty tuple."""

    assert MediationAnalysisCueExtractor().extract([]) == ()


def test_flagged_match() -> None:
    """Matching methods text is flagged."""

    rows = MediationAnalysisCueExtractor().extract(
        [
            {
                "paper_id": "p1",
                "methods": "We conducted a causal mediation analysis of the indirect effect.",
            }
        ]
    )
    assert rows[0].flagged is True
    assert "mediation_analysis" in rows[0].cues


def test_no_match() -> None:
    """Unrelated text is not flagged."""

    rows = MediationAnalysisCueExtractor().extract(
        [{"paper_id": "p1", "abstract": "No relevant design cues."}]
    )
    assert rows[0].flagged is False


def test_resolves_doi() -> None:
    """Falls back to doi when paper_id missing."""

    rows = MediationAnalysisCueExtractor().extract(
        [
            {
                "doi": "10.1/x",
                "methods": "We conducted a causal mediation analysis of the indirect effect.",
            }
        ]
    )
    assert rows[0].paper_id == "10.1/x"
