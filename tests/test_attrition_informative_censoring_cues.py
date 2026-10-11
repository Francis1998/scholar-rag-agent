"""Unit tests for AttritionInformativeCensoringCueExtractor."""

from __future__ import annotations

from retrieval.attrition_informative_censoring_cues import AttritionInformativeCensoringCueExtractor


def test_flags_matched_abstract() -> None:
    """Flag papers with cue wording."""

    cues = AttritionInformativeCensoringCueExtractor().extract(
        [
            {
                "paper_id": "p1",
                "abstract": ("Bias may arise from informative censoring in the cohort."),
            }
        ]
    )
    assert cues[0].flagged is True
    assert cues[0].cue_kind is not None
    assert cues[0].cues


def test_empty_input() -> None:
    """Empty paper list returns empty tuple."""

    assert AttritionInformativeCensoringCueExtractor().extract([]) == ()


def test_no_match() -> None:
    """Unrelated abstract is not flagged."""

    cues = AttritionInformativeCensoringCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "A randomized trial of vitamin C."}]
    )
    assert cues[0].flagged is False
