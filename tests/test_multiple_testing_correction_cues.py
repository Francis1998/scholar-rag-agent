"""Unit tests for MultipleTestingCorrectionCueExtractor."""

from __future__ import annotations

from retrieval.multiple_testing_correction_cues import MultipleTestingCorrectionCueExtractor


def test_extracts_hit() -> None:
    """Flag papers with cue wording."""

    cues = MultipleTestingCorrectionCueExtractor().extract(
        [
            {
                "paper_id": "p1",
                "abstract": (
                    "Bonferroni correction and FDR correction controlled "
                    "family-wise error with a Holm-Bonferroni step."
                ),
            }
        ]
    )
    assert len(cues) == 1
    assert cues[0].flagged is True
    assert cues[0].cue_kind is not None


def test_empty_papers() -> None:
    """Empty input returns empty tuple."""

    assert MultipleTestingCorrectionCueExtractor().extract([]) == ()


def test_no_match() -> None:
    """Unrelated abstract is not flagged."""

    cues = MultipleTestingCorrectionCueExtractor().extract(
        [
            {
                "paper_id": "p2",
                "abstract": "A narrative review of nursing education.",
            }
        ]
    )
    assert cues[0].flagged is False
