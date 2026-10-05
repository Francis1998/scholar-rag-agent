"""Unit tests for DoseResponseCueExtractor."""

from __future__ import annotations

from retrieval.dose_response_cues import DoseResponseCueExtractor


def test_extracts_hit() -> None:
    """Flag papers with cue wording."""

    cues = DoseResponseCueExtractor().extract(
        [
            {
                "paper_id": "p1",
                "abstract": "Clear dose-response with ED50 estimates.",
            }
        ]
    )
    assert len(cues) == 1
    assert cues[0].flagged is True
    assert cues[0].cue_kind is not None


def test_empty_papers() -> None:
    """Empty input returns empty tuple."""

    assert DoseResponseCueExtractor().extract([]) == ()


def test_no_match() -> None:
    """Unrelated abstract is not flagged."""

    cues = DoseResponseCueExtractor().extract(
        [
            {
                "paper_id": "p2",
                "abstract": "A narrative review of nursing education.",
            }
        ]
    )
    assert cues[0].flagged is False
