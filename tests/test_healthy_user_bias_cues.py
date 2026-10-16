"""Tests for HealthyUserBiasCueExtractor."""

from __future__ import annotations

from retrieval.healthy_user_bias_cues import HealthyUserBiasCueExtractor


def test_empty() -> None:
    """Empty input returns empty tuple."""

    assert HealthyUserBiasCueExtractor().extract([]) == ()


def test_flags_match() -> None:
    """Matching abstract is flagged."""

    rows = [
        {
            "paper_id": "p1",
            "abstract": "Results may reflect healthy user bias among adherent participants.",
        }
    ]
    out = HealthyUserBiasCueExtractor().extract(rows)
    assert len(out) == 1
    assert out[0].flagged is True
    assert out[0].cue_kind == "healthy_user_bias"


def test_no_match() -> None:
    """Unrelated abstract is not flagged."""

    rows = [{"paper_id": "p2", "abstract": "No relevant bias language here."}]
    out = HealthyUserBiasCueExtractor().extract(rows)
    assert out[0].flagged is False
