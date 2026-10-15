"""Tests for EcologicalFallacyCueExtractor."""

from __future__ import annotations

from retrieval.ecological_fallacy_cues import EcologicalFallacyCueExtractor


def test_empty() -> None:
    """Empty input returns empty tuple."""

    assert EcologicalFallacyCueExtractor().extract([]) == ()


def test_flags_match() -> None:
    """Matching abstract is flagged."""

    rows = [
        {
            "paper_id": "p1",
            "abstract": "We discuss ecological fallacy when interpreting group rates.",
        }
    ]
    out = EcologicalFallacyCueExtractor().extract(rows)
    assert len(out) == 1
    assert out[0].flagged is True
    assert out[0].cue_kind == "ecological_fallacy"


def test_no_match() -> None:
    """Unrelated abstract is not flagged."""

    rows = [{"paper_id": "p2", "abstract": "No relevant bias language here."}]
    out = EcologicalFallacyCueExtractor().extract(rows)
    assert out[0].flagged is False
