"""Tests for SocialDesirabilityBiasCueExtractor."""

from __future__ import annotations

from retrieval.social_desirability_bias_cues import SocialDesirabilityBiasCueExtractor


def test_empty() -> None:
    """Empty input returns empty tuple."""

    assert SocialDesirabilityBiasCueExtractor().extract([]) == ()


def test_flagged() -> None:
    """Known phrase flags paper."""

    rows = SocialDesirabilityBiasCueExtractor().extract(
        [{"paper_id": "p1", "abstract": "Study notes social-desirability bias in methods."}]
    )
    assert len(rows) == 1
    assert rows[0].flagged is True
    assert "social_desirability_bias" in rows[0].cues


def test_unflagged() -> None:
    """Unrelated abstract stays clean."""

    rows = SocialDesirabilityBiasCueExtractor().extract(
        [{"id": "p2", "title": "Routine cohort description without bias terms."}]
    )
    assert rows[0].flagged is False
