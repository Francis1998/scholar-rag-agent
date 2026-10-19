"""Tests for ReferralBiasCueExtractor."""

from __future__ import annotations

from retrieval.referral_bias_cues import ReferralBiasCueExtractor


def test_flagged_match() -> None:
    """Known cue text is flagged."""

    rows = ReferralBiasCueExtractor().extract(
        [
            {
                "paper_id": "p1",
                "abstract": "Authors note referral bias in clinic enrollment.",
            }
        ]
    )
    assert rows[0].flagged is True
    assert rows[0].cues


def test_no_match() -> None:
    """Unrelated abstract is not flagged."""

    rows = ReferralBiasCueExtractor().extract(
        [
            {
                "paper_id": "p2",
                "abstract": "No relevant epidemiologic bias language here.",
            }
        ]
    )
    assert rows[0].flagged is False


def test_empty_input() -> None:
    """Empty papers returns empty tuple."""

    assert ReferralBiasCueExtractor().extract([]) == ()


def test_resolves_fallback_id() -> None:
    """Missing ids fall back to paper-index."""

    rows = ReferralBiasCueExtractor().extract(
        [{"abstract": "Authors note referral bias in clinic enrollment."}]
    )
    assert rows[0].paper_id == "paper-0"
