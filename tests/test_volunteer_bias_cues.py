"""Unit tests for VolunteerBiasCueExtractor."""

from __future__ import annotations

from retrieval.volunteer_bias_cues import VolunteerBiasCueExtractor


def test_flags_matched_abstract() -> None:
    """Flag papers with cue wording."""

    cues = VolunteerBiasCueExtractor().extract(
        [
            {
                "paper_id": "p1",
                "abstract": (
                    "Enrollment may suffer from volunteer bias among health-conscious participants."
                ),
            }
        ]
    )
    assert cues[0].flagged is True
    assert cues[0].cue_kind is not None
    assert cues[0].cues


def test_empty_input() -> None:
    """Empty paper list returns empty tuple."""

    assert VolunteerBiasCueExtractor().extract([]) == ()


def test_no_match() -> None:
    """Unrelated abstract is not flagged."""

    cues = VolunteerBiasCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "A randomized trial of vitamin C."}]
    )
    assert cues[0].flagged is False
