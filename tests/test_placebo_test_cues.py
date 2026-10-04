"""Unit tests for PlaceboTestCueExtractor."""

from __future__ import annotations

from retrieval.placebo_test_cues import PlaceboTestCueExtractor


def test_extracts_hit() -> None:
    """Flag papers with cue wording."""

    cues = PlaceboTestCueExtractor().extract(
        [
            {
                "paper_id": "p1",
                "abstract": "We ran a placebo test and falsification check with null effect.",
            }
        ]
    )
    assert len(cues) == 1
    assert cues[0].flagged is True
    assert cues[0].cue_kind is not None


def test_empty_papers() -> None:
    """Empty input returns empty tuple."""

    assert PlaceboTestCueExtractor().extract([]) == ()


def test_no_match() -> None:
    """Unrelated abstract is not flagged."""

    cues = PlaceboTestCueExtractor().extract(
        [
            {
                "paper_id": "p2",
                "abstract": "A narrative review of nursing education.",
            }
        ]
    )
    assert cues[0].flagged is False
