"""Tests for CompetingEventBiasCueExtractor."""

from __future__ import annotations

from retrieval.competing_event_bias_cues import CompetingEventBiasCueExtractor


def test_flagged_match() -> None:
    """Known cue text is flagged."""

    rows = CompetingEventBiasCueExtractor().extract(
        [
            {
                "paper_id": "p1",
                "abstract": "Authors warn about competing event bias.",
            }
        ]
    )
    assert rows[0].flagged is True
    assert rows[0].cues


def test_no_match() -> None:
    """Unrelated abstract is not flagged."""

    rows = CompetingEventBiasCueExtractor().extract(
        [
            {
                "paper_id": "p2",
                "abstract": "No relevant epidemiologic bias language here.",
            }
        ]
    )
    assert rows[0].flagged is False


def test_empty_input() -> None:
    """Empty papers returns empty tuple."""

    assert CompetingEventBiasCueExtractor().extract([]) == ()


def test_resolves_fallback_id() -> None:
    """Missing ids fall back to paper-index."""

    rows = CompetingEventBiasCueExtractor().extract(
        [{"abstract": "Authors warn about competing event bias."}]
    )
    assert rows[0].paper_id == "paper-0"
