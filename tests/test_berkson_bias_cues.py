"""Unit tests for BerksonBiasCueExtractor."""

from __future__ import annotations

from retrieval.berkson_bias_cues import BerksonBiasCueExtractor


def test_flags_matched_abstract() -> None:
    """Flag papers with cue wording."""

    cues = BerksonBiasCueExtractor().extract(
        [
            {
                "paper_id": "p1",
                "abstract": ("Results may reflect Berkson's bias from hospital sampling."),
            }
        ]
    )
    assert cues[0].flagged is True
    assert cues[0].cue_kind is not None
    assert cues[0].cues


def test_empty_input() -> None:
    """Empty paper list returns empty tuple."""

    assert BerksonBiasCueExtractor().extract([]) == ()


def test_no_match() -> None:
    """Unrelated abstract is not flagged."""

    cues = BerksonBiasCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "A randomized trial of vitamin C."}]
    )
    assert cues[0].flagged is False
