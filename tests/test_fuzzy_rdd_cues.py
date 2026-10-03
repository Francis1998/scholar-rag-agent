"""Unit tests for FuzzyRddCueExtractor."""

from __future__ import annotations

from retrieval.fuzzy_rdd_cues import FuzzyRddCueExtractor


def test_extracts_hit() -> None:
    """Flag papers with cue wording."""

    cues = FuzzyRddCueExtractor().extract(
        [
            {
                "paper_id": "p1",
                "abstract": "A fuzzy RDD shows treatment probability jump.",
            }
        ]
    )
    assert len(cues) == 1
    assert cues[0].flagged is True
    assert cues[0].cue_kind is not None


def test_empty_papers() -> None:
    """Empty input returns empty tuple."""

    assert FuzzyRddCueExtractor().extract([]) == ()


def test_no_match() -> None:
    """Unrelated abstract is not flagged."""

    cues = FuzzyRddCueExtractor().extract(
        [
            {
                "paper_id": "p2",
                "abstract": "A narrative review of nursing education.",
            }
        ]
    )
    assert cues[0].flagged is False
