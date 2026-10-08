"""Unit tests for RecallBiasCueExtractor."""

from __future__ import annotations

from retrieval.recall_bias_cues import RecallBiasCueExtractor


def test_flags_matched_abstract() -> None:
    """Flag papers with cue wording."""

    cues = RecallBiasCueExtractor().extract(
        [
            {
                "paper_id": "p1",
                "abstract": (
                    "Case-control studies may be affected by recall bias in exposureassessment."
                ),
            }
        ]
    )
    assert cues[0].flagged is True
    assert cues[0].cue_kind is not None
    assert cues[0].cues


def test_empty_input() -> None:
    """Empty paper list returns empty tuple."""

    assert RecallBiasCueExtractor().extract([]) == ()


def test_no_match() -> None:
    """Unrelated abstract is not flagged."""

    cues = RecallBiasCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "A randomized trial of vitamin C."}]
    )
    assert cues[0].flagged is False
