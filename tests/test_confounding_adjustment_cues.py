"""Unit tests for ConfoundingAdjustmentCueExtractor."""

from __future__ import annotations

from retrieval.confounding_adjustment_cues import ConfoundingAdjustmentCueExtractor


def test_empty_papers() -> None:
    """Empty input returns empty tuple."""

    assert ConfoundingAdjustmentCueExtractor().extract([]) == ()


def test_flagged_match() -> None:
    """Matching methods text is flagged."""

    rows = ConfoundingAdjustmentCueExtractor().extract(
        [
            {
                "paper_id": "p1",
                "methods": "We used propensity score matching to adjust for confounding.",
            }
        ]
    )
    assert rows[0].flagged is True
    assert "propensity_score" in rows[0].cues


def test_no_match() -> None:
    """Unrelated text is not flagged."""

    rows = ConfoundingAdjustmentCueExtractor().extract(
        [{"paper_id": "p1", "abstract": "No relevant design cues."}]
    )
    assert rows[0].flagged is False


def test_resolves_doi() -> None:
    """Falls back to doi when paper_id missing."""

    rows = ConfoundingAdjustmentCueExtractor().extract(
        [
            {
                "doi": "10.1/x",
                "methods": "We used propensity score matching to adjust for confounding.",
            }
        ]
    )
    assert rows[0].paper_id == "10.1/x"
