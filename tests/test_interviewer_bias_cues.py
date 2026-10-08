"""Unit tests for InterviewerBiasCueExtractor."""

from __future__ import annotations

from retrieval.interviewer_bias_cues import InterviewerBiasCueExtractor


def test_flags_matched_abstract() -> None:
    """Flag papers with cue wording."""

    cues = InterviewerBiasCueExtractor().extract(
        [
            {
                "paper_id": "p1",
                "abstract": (
                    "Structured interviews reduced interviewer bias during outcomeascertainment."
                ),
            }
        ]
    )
    assert cues[0].flagged is True
    assert cues[0].cue_kind is not None
    assert cues[0].cues


def test_empty_input() -> None:
    """Empty paper list returns empty tuple."""

    assert InterviewerBiasCueExtractor().extract([]) == ()


def test_no_match() -> None:
    """Unrelated abstract is not flagged."""

    cues = InterviewerBiasCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "A randomized trial of vitamin C."}]
    )
    assert cues[0].flagged is False
