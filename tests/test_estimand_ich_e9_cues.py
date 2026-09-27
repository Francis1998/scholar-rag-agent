"""Unit tests for EstimandIchE9CueExtractor."""

from __future__ import annotations

from retrieval.estimand_ich_e9_cues import EstimandIchE9CueExtractor


def test_empty_papers() -> None:
    """Empty input returns empty tuple."""

    assert EstimandIchE9CueExtractor().extract([]) == ()


def test_flagged_match() -> None:
    """Matching methods text is flagged."""

    rows = EstimandIchE9CueExtractor().extract(
        [{"paper_id": "p1", "methods": "The treatment-policy estimand was pre-specified."}]
    )
    assert rows[0].flagged is True
    assert "treatment_policy_estimand" in rows[0].cues


def test_no_match() -> None:
    """Unrelated text is not flagged."""

    rows = EstimandIchE9CueExtractor().extract(
        [{"paper_id": "p1", "abstract": "No relevant design cues."}]
    )
    assert rows[0].flagged is False


def test_resolves_doi() -> None:
    """Falls back to doi when paper_id missing."""

    rows = EstimandIchE9CueExtractor().extract(
        [{"doi": "10.1/x", "methods": "The treatment-policy estimand was pre-specified."}]
    )
    assert rows[0].paper_id == "10.1/x"
