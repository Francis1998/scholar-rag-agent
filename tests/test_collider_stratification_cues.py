"""Unit tests for ColliderStratificationCueExtractor."""

from __future__ import annotations

from retrieval.collider_stratification_cues import ColliderStratificationCueExtractor


def test_extracts_hit() -> None:
    """Flag papers with cue wording."""

    cues = ColliderStratificationCueExtractor().extract(
        [
            {
                "paper_id": "p1",
                "abstract": (
                    "Collider stratification and Berkson's bias from "
                    "conditioning on a collider, producing collider bias."
                ),
            }
        ]
    )
    assert len(cues) == 1
    assert cues[0].flagged is True
    assert cues[0].cue_kind is not None


def test_empty_papers() -> None:
    """Empty input returns empty tuple."""

    assert ColliderStratificationCueExtractor().extract([]) == ()


def test_no_match() -> None:
    """Unrelated abstract is not flagged."""

    cues = ColliderStratificationCueExtractor().extract(
        [
            {
                "paper_id": "p2",
                "abstract": "A narrative review of nursing education.",
            }
        ]
    )
    assert cues[0].flagged is False
