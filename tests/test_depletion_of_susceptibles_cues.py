"""Tests for DepletionOfSusceptiblesCueExtractor."""

from __future__ import annotations

from retrieval.depletion_of_susceptibles_cues import DepletionOfSusceptiblesCueExtractor


def test_empty() -> None:
    """Empty input returns empty tuple."""

    assert DepletionOfSusceptiblesCueExtractor().extract([]) == ()


def test_flags_match() -> None:
    """Matching abstract is flagged."""

    rows = [
        {
            "paper_id": "p1",
            "abstract": "We discuss depletion of susceptibles in the cohort.",
        }
    ]
    out = DepletionOfSusceptiblesCueExtractor().extract(rows)
    assert len(out) == 1
    assert out[0].flagged is True
    assert out[0].cue_kind == "depletion_of_susceptibles"


def test_no_match() -> None:
    """Unrelated abstract is not flagged."""

    rows = [{"paper_id": "p2", "abstract": "No relevant bias language here."}]
    out = DepletionOfSusceptiblesCueExtractor().extract(rows)
    assert out[0].flagged is False
