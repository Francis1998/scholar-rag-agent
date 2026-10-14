"""Tests for ResidualConfoundingCueExtractor."""

from __future__ import annotations

from retrieval.residual_confounding_cues import ResidualConfoundingCueExtractor


def test_empty() -> None:
    """Empty input returns empty tuple."""

    assert ResidualConfoundingCueExtractor().extract([]) == ()


def test_flags_match() -> None:
    """Matching abstract is flagged."""

    rows = [
        {
            "paper_id": "p1",
            "abstract": "We discuss residual confounding after adjustment.",
        }
    ]
    out = ResidualConfoundingCueExtractor().extract(rows)
    assert len(out) == 1
    assert out[0].flagged is True
    assert out[0].cue_kind == "residual_confounding"


def test_no_match() -> None:
    """Unrelated abstract is not flagged."""

    rows = [{"paper_id": "p2", "abstract": "No relevant bias language here."}]
    out = ResidualConfoundingCueExtractor().extract(rows)
    assert out[0].flagged is False
