"""Tests for ReverseCausationCueExtractor."""

from __future__ import annotations

from retrieval.reverse_causation_cues import ReverseCausationCueExtractor


def test_empty() -> None:
    """Empty input returns empty tuple."""

    assert ReverseCausationCueExtractor().extract([]) == ()


def test_flags_match() -> None:
    """Matching abstract is flagged."""

    rows = [
        {
            "paper_id": "p1",
            "abstract": "We discuss reverse causation in the cohort.",
        }
    ]
    out = ReverseCausationCueExtractor().extract(rows)
    assert len(out) == 1
    assert out[0].flagged is True
    assert out[0].cue_kind == "reverse_causation"


def test_no_match() -> None:
    """Unrelated abstract is not flagged."""

    rows = [{"paper_id": "p2", "abstract": "No relevant bias language here."}]
    out = ReverseCausationCueExtractor().extract(rows)
    assert out[0].flagged is False
