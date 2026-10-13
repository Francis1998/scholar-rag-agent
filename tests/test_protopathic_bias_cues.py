"""Tests for ProtopathicBiasCueExtractor."""

from __future__ import annotations

from retrieval.protopathic_bias_cues import ProtopathicBiasCueExtractor


def test_empty() -> None:
    """Empty input returns empty tuple."""

    assert ProtopathicBiasCueExtractor().extract([]) == ()


def test_flags_match() -> None:
    """Matching abstract is flagged."""

    rows = [
        {
            "paper_id": "p1",
            "abstract": "We discuss protopathic bias in the cohort.",
        }
    ]
    out = ProtopathicBiasCueExtractor().extract(rows)
    assert len(out) == 1
    assert out[0].flagged is True
    assert out[0].cue_kind == "protopathic_bias"


def test_no_match() -> None:
    """Unrelated abstract is not flagged."""

    rows = [{"paper_id": "p2", "abstract": "No relevant bias language here."}]
    out = ProtopathicBiasCueExtractor().extract(rows)
    assert out[0].flagged is False
