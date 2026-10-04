"""Unit tests for HeterogeneousTreatmentEffectCueExtractor."""

from __future__ import annotations

from retrieval.heterogeneous_treatment_cues import HeterogeneousTreatmentEffectCueExtractor


def test_extracts_hit() -> None:
    """Flag papers with cue wording."""

    cues = HeterogeneousTreatmentEffectCueExtractor().extract(
        [
            {
                "paper_id": "p1",
                "abstract": "We estimate heterogeneous treatment effects and CATE.",
            }
        ]
    )
    assert len(cues) == 1
    assert cues[0].flagged is True
    assert cues[0].cue_kind is not None


def test_empty_papers() -> None:
    """Empty input returns empty tuple."""

    assert HeterogeneousTreatmentEffectCueExtractor().extract([]) == ()


def test_no_match() -> None:
    """Unrelated abstract is not flagged."""

    cues = HeterogeneousTreatmentEffectCueExtractor().extract(
        [
            {
                "paper_id": "p2",
                "abstract": "A narrative review of nursing education.",
            }
        ]
    )
    assert cues[0].flagged is False
