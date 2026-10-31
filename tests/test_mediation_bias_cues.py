"""Tests for MediationBiasCueExtractor."""

from __future__ import annotations

from retrieval.mediation_bias_cues import MediationBiasCueExtractor


def test_empty() -> None:
    """Empty input returns empty."""

    assert MediationBiasCueExtractor().extract([]) == ()


def test_detects_cue() -> None:
    """Detects primary cue phrase."""

    rows = MediationBiasCueExtractor().extract(
        [{"paper_id": "p1", "abstract": "We discuss mediation bias in the cohort."}]
    )
    assert rows[0].flagged is True
    assert "mediation_bias" in rows[0].cues


def test_no_match() -> None:
    """Unrelated text is not flagged."""

    rows = MediationBiasCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "No relevant methods described."}]
    )
    assert rows[0].flagged is False
