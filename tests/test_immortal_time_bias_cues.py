"""Unit tests for ImmortalTimeBiasCueExtractor."""

from __future__ import annotations

from retrieval.immortal_time_bias_cues import ImmortalTimeBiasCueExtractor


def test_extracts_hit() -> None:
    """Flag papers with cue wording."""

    cues = ImmortalTimeBiasCueExtractor().extract(
        [
            {
                "paper_id": "p1",
                "abstract": (
                    "Immortal time bias and guarantee-time bias from "
                    "time-dependent exposure misclassification."
                ),
            }
        ]
    )
    assert len(cues) == 1
    assert cues[0].flagged is True
    assert cues[0].cue_kind is not None


def test_empty_papers() -> None:
    """Empty input returns empty tuple."""

    assert ImmortalTimeBiasCueExtractor().extract([]) == ()


def test_no_match() -> None:
    """Unrelated abstract is not flagged."""

    cues = ImmortalTimeBiasCueExtractor().extract(
        [
            {
                "paper_id": "p2",
                "abstract": "A narrative review of nursing education.",
            }
        ]
    )
    assert cues[0].flagged is False
