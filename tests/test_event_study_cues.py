"""Unit tests for EventStudyCueExtractor."""

from __future__ import annotations

from retrieval.event_study_cues import EventStudyCueExtractor


def test_extracts_hit() -> None:
    """Flag papers with cue wording."""

    cues = EventStudyCueExtractor().extract(
        [
            {
                "paper_id": "p1",
                "abstract": "An event study design supports parallel trends.",
            }
        ]
    )
    assert len(cues) == 1
    assert cues[0].flagged is True
    assert cues[0].cue_kind is not None


def test_empty_papers() -> None:
    """Empty input returns empty tuple."""

    assert EventStudyCueExtractor().extract([]) == ()


def test_no_match() -> None:
    """Unrelated abstract is not flagged."""

    cues = EventStudyCueExtractor().extract(
        [
            {
                "paper_id": "p2",
                "abstract": "A narrative review of nursing education.",
            }
        ]
    )
    assert cues[0].flagged is False
