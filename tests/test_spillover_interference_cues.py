"""Unit tests for SpilloverInterferenceCueExtractor."""

from __future__ import annotations

from retrieval.spillover_interference_cues import SpilloverInterferenceCueExtractor


def test_extracts_hit() -> None:
    """Flag papers with cue wording."""

    cues = SpilloverInterferenceCueExtractor().extract(
        [
            {
                "paper_id": "p1",
                "abstract": "Treatment spillover and interference violated SUTVA.",
            }
        ]
    )
    assert len(cues) == 1
    assert cues[0].flagged is True
    assert cues[0].cue_kind is not None


def test_empty_papers() -> None:
    """Empty input returns empty tuple."""

    assert SpilloverInterferenceCueExtractor().extract([]) == ()


def test_no_match() -> None:
    """Unrelated abstract is not flagged."""

    cues = SpilloverInterferenceCueExtractor().extract(
        [
            {
                "paper_id": "p2",
                "abstract": "A narrative review of nursing education.",
            }
        ]
    )
    assert cues[0].flagged is False
