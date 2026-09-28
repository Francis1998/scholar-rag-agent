"""Unit tests for BayesianInterimPriorCueExtractor."""

from __future__ import annotations

from retrieval.bayesian_interim_prior_cues import BayesianInterimPriorCueExtractor


def test_empty_papers() -> None:
    """Empty input returns empty tuple."""

    assert BayesianInterimPriorCueExtractor().extract([]) == ()


def test_flagged_match() -> None:
    """Matching methods text is flagged."""

    rows = BayesianInterimPriorCueExtractor().extract(
        [
            {
                "paper_id": "p1",
                "methods": "A weakly informative prior guided the Bayesian interim analysis.",
            }
        ]
    )
    assert rows[0].flagged is True
    assert "bayesian_prior" in rows[0].cues


def test_no_match() -> None:
    """Unrelated text is not flagged."""

    rows = BayesianInterimPriorCueExtractor().extract(
        [{"paper_id": "p1", "abstract": "No relevant design cues."}]
    )
    assert rows[0].flagged is False


def test_resolves_doi() -> None:
    """Falls back to doi when paper_id missing."""

    rows = BayesianInterimPriorCueExtractor().extract(
        [
            {
                "doi": "10.1/x",
                "methods": "A weakly informative prior guided the Bayesian interim analysis.",
            }
        ]
    )
    assert rows[0].paper_id == "10.1/x"
