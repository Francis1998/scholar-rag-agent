"""Unit tests for LeadTimeBiasCueExtractor."""

from __future__ import annotations

from retrieval.lead_time_bias_cues import LeadTimeBiasCueExtractor


def test_flags_matched_abstract() -> None:
    """Flag papers with cue wording."""

    cues = LeadTimeBiasCueExtractor().extract(
        [
            {
                "paper_id": "p1",
                "abstract": ("Survival gains may reflect lead-time bias from earlier detection."),
            }
        ]
    )
    assert cues[0].flagged is True
    assert cues[0].cue_kind is not None
    assert cues[0].cues


def test_empty_input() -> None:
    """Empty paper list returns empty tuple."""

    assert LeadTimeBiasCueExtractor().extract([]) == ()


def test_no_match() -> None:
    """Unrelated abstract is not flagged."""

    cues = LeadTimeBiasCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "A randomized trial of vitamin C."}]
    )
    assert cues[0].flagged is False
