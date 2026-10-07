"""Unit tests for LeftTruncationCueExtractor."""

from __future__ import annotations

from retrieval.left_truncation_cues import LeftTruncationCueExtractor


def test_extracts_hit() -> None:
    """Flag papers with cue wording."""

    cues = LeftTruncationCueExtractor().extract(
        [
            {
                "paper_id": "p1",
                "abstract": (
                    "Left truncation and delayed entry required late-entry "
                    "adjustment for left-truncated survival."
                ),
            }
        ]
    )
    assert len(cues) == 1
    assert cues[0].flagged is True
    assert cues[0].cue_kind is not None


def test_empty_papers() -> None:
    """Empty input returns empty tuple."""

    assert LeftTruncationCueExtractor().extract([]) == ()


def test_no_match() -> None:
    """Unrelated abstract is not flagged."""

    cues = LeftTruncationCueExtractor().extract(
        [
            {
                "paper_id": "p2",
                "abstract": "A narrative review of nursing education.",
            }
        ]
    )
    assert cues[0].flagged is False
