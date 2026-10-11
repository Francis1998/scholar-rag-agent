"""Unit tests for VerificationBiasCueExtractor."""

from __future__ import annotations

from retrieval.verification_bias_cues import VerificationBiasCueExtractor


def test_flags_matched_abstract() -> None:
    """Flag papers with cue wording."""

    cues = VerificationBiasCueExtractor().extract(
        [
            {
                "paper_id": "p1",
                "abstract": ("Results may reflect verification bias in tertiary clinics."),
            }
        ]
    )
    assert cues[0].flagged is True
    assert cues[0].cue_kind is not None
    assert cues[0].cues


def test_empty_input() -> None:
    """Empty paper list returns empty tuple."""

    assert VerificationBiasCueExtractor().extract([]) == ()


def test_no_match() -> None:
    """Unrelated abstract is not flagged."""

    cues = VerificationBiasCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "A randomized trial of vitamin C."}]
    )
    assert cues[0].flagged is False
