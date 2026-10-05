"""Unit tests for EValueSensitivityCueExtractor."""

from __future__ import annotations

from retrieval.evalue_sensitivity_cues import EValueSensitivityCueExtractor


def test_extracts_hit() -> None:
    """Flag papers with cue wording."""

    cues = EValueSensitivityCueExtractor().extract(
        [
            {
                "paper_id": "p1",
                "abstract": "E-value sensitivity analysis for unmeasured confounding.",
            }
        ]
    )
    assert len(cues) == 1
    assert cues[0].flagged is True
    assert cues[0].cue_kind is not None


def test_empty_papers() -> None:
    """Empty input returns empty tuple."""

    assert EValueSensitivityCueExtractor().extract([]) == ()


def test_no_match() -> None:
    """Unrelated abstract is not flagged."""

    cues = EValueSensitivityCueExtractor().extract(
        [
            {
                "paper_id": "p2",
                "abstract": "A narrative review of nursing education.",
            }
        ]
    )
    assert cues[0].flagged is False
