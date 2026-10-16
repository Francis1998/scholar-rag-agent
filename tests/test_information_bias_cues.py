"""Tests for InformationBiasCueExtractor."""

from __future__ import annotations

from retrieval.information_bias_cues import InformationBiasCueExtractor


def test_empty() -> None:
    """Empty input returns empty tuple."""

    assert InformationBiasCueExtractor().extract([]) == ()


def test_flags_match() -> None:
    """Matching abstract is flagged."""

    rows = [
        {
            "paper_id": "p1",
            "abstract": "We discuss information bias from exposure misclassification.",
        }
    ]
    out = InformationBiasCueExtractor().extract(rows)
    assert len(out) == 1
    assert out[0].flagged is True
    assert out[0].cue_kind == "information_bias"


def test_no_match() -> None:
    """Unrelated abstract is not flagged."""

    rows = [{"paper_id": "p2", "abstract": "No relevant bias language here."}]
    out = InformationBiasCueExtractor().extract(rows)
    assert out[0].flagged is False
