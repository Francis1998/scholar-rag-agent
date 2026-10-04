"""Unit tests for NegativeControlExposureCueExtractor."""

from __future__ import annotations

from retrieval.negative_control_exposure_cues import NegativeControlExposureCueExtractor


def test_extracts_hit() -> None:
    """Flag papers with cue wording."""

    cues = NegativeControlExposureCueExtractor().extract(
        [
            {
                "paper_id": "p1",
                "abstract": "We included a negative control exposure check.",
            }
        ]
    )
    assert len(cues) == 1
    assert cues[0].flagged is True
    assert cues[0].cue_kind is not None


def test_empty_papers() -> None:
    """Empty input returns empty tuple."""

    assert NegativeControlExposureCueExtractor().extract([]) == ()


def test_no_match() -> None:
    """Unrelated abstract is not flagged."""

    cues = NegativeControlExposureCueExtractor().extract(
        [
            {
                "paper_id": "p2",
                "abstract": "A narrative review of nursing education.",
            }
        ]
    )
    assert cues[0].flagged is False
