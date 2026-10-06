"""Unit tests for SelectionBiasCueExtractor."""

from __future__ import annotations

from retrieval.selection_bias_cues import SelectionBiasCueExtractor


def test_extracts_hit() -> None:
    """Flag papers with cue wording."""

    cues = SelectionBiasCueExtractor().extract(
        [
            {
                "paper_id": "p1",
                "abstract": (
                    "Selection bias and volunteer bias, including healthy-user bias "
                    "and a selection effect from collider selection."
                ),
            }
        ]
    )
    assert len(cues) == 1
    assert cues[0].flagged is True
    assert cues[0].cue_kind is not None


def test_empty_papers() -> None:
    """Empty input returns empty tuple."""

    assert SelectionBiasCueExtractor().extract([]) == ()


def test_no_match() -> None:
    """Unrelated abstract is not flagged."""

    cues = SelectionBiasCueExtractor().extract(
        [
            {
                "paper_id": "p2",
                "abstract": "A narrative review of nursing education.",
            }
        ]
    )
    assert cues[0].flagged is False
