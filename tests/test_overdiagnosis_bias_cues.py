"""Unit tests for OverdiagnosisBiasCueExtractor."""

from __future__ import annotations

from retrieval.overdiagnosis_bias_cues import OverdiagnosisBiasCueExtractor


def test_flagged() -> None:
    """Flag known cue wording."""

    cues = OverdiagnosisBiasCueExtractor().extract(
        [
            {
                "paper_id": "p1",
                "abstract": "Screening studies may reflect overdiagnosis bias in indolent lesions.",
            }
        ]
    )
    assert cues[0].flagged is True
    assert cues[0].cue_kind == "overdiagnosis_bias"


def test_empty() -> None:
    """Empty input returns empty tuple."""

    assert OverdiagnosisBiasCueExtractor().extract([]) == ()


def test_unflagged() -> None:
    """Neutral abstract is not flagged."""

    cues = OverdiagnosisBiasCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "A randomized trial of usual care."}]
    )
    assert cues[0].flagged is False
