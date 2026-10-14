"""Tests for SurvivorBiasCueExtractor."""

from __future__ import annotations

from retrieval.survivor_bias_cues import SurvivorBiasCueExtractor


def test_empty() -> None:
    """Empty input returns empty tuple."""

    assert SurvivorBiasCueExtractor().extract([]) == ()


def test_flags_match() -> None:
    """Matching abstract is flagged."""

    rows = [
        {
            "paper_id": "p1",
            "abstract": "We discuss survivor bias in the cohort.",
        }
    ]
    out = SurvivorBiasCueExtractor().extract(rows)
    assert len(out) == 1
    assert out[0].flagged is True
    assert out[0].cue_kind == "survivor_bias"


def test_no_match() -> None:
    """Unrelated abstract is not flagged."""

    rows = [{"paper_id": "p2", "abstract": "No relevant bias language here."}]
    out = SurvivorBiasCueExtractor().extract(rows)
    assert out[0].flagged is False
