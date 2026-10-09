"""Unit tests for ConfoundingByIndicationCueExtractor."""

from __future__ import annotations

from retrieval.confounding_by_indication_cues import ConfoundingByIndicationCueExtractor


def test_flags_matched_abstract() -> None:
    """Flag papers with cue wording."""

    cues = ConfoundingByIndicationCueExtractor().extract(
        [
            {
                "paper_id": "p1",
                "abstract": (
                    "We discuss confounding by indication when sicker patients receive treatment."
                ),
            }
        ]
    )
    assert cues[0].flagged is True
    assert cues[0].cue_kind is not None
    assert cues[0].cues


def test_empty_input() -> None:
    """Empty paper list returns empty tuple."""

    assert ConfoundingByIndicationCueExtractor().extract([]) == ()


def test_no_match() -> None:
    """Unrelated abstract is not flagged."""

    cues = ConfoundingByIndicationCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "A randomized trial of vitamin C."}]
    )
    assert cues[0].flagged is False
