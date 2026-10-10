"""Unit tests for SpectrumBiasCueExtractor."""

from __future__ import annotations

from retrieval.spectrum_bias_cues import SpectrumBiasCueExtractor


def test_flags_matched_abstract() -> None:
    """Flag papers with cue wording."""

    cues = SpectrumBiasCueExtractor().extract(
        [
            {
                "paper_id": "p1",
                "abstract": (
                    "Diagnostic accuracy may be inflated by spectrum bias in tertiary care."
                ),
            }
        ]
    )
    assert cues[0].flagged is True
    assert cues[0].cue_kind is not None
    assert cues[0].cues


def test_empty_input() -> None:
    """Empty paper list returns empty tuple."""

    assert SpectrumBiasCueExtractor().extract([]) == ()


def test_no_match() -> None:
    """Unrelated abstract is not flagged."""

    cues = SpectrumBiasCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "A randomized trial of vitamin C."}]
    )
    assert cues[0].flagged is False
