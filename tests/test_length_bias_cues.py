"""Unit tests for LengthBiasCueExtractor."""

from __future__ import annotations

from retrieval.length_bias_cues import LengthBiasCueExtractor


def test_flags_matched_abstract() -> None:
    """Flag papers with cue wording."""

    cues = LengthBiasCueExtractor().extract(
        [
            {
                "paper_id": "p1",
                "abstract": (
                    "Screening studies can suffer from length-bias favoring indolent cases."
                ),
            }
        ]
    )
    assert cues[0].flagged is True
    assert cues[0].cue_kind is not None
    assert cues[0].cues


def test_empty_input() -> None:
    """Empty paper list returns empty tuple."""

    assert LengthBiasCueExtractor().extract([]) == ()


def test_no_match() -> None:
    """Unrelated abstract is not flagged."""

    cues = LengthBiasCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "A randomized trial of vitamin C."}]
    )
    assert cues[0].flagged is False
