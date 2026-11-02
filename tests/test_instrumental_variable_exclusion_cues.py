"""Tests for InstrumentalVariableExclusionCueExtractor."""

from __future__ import annotations

from retrieval.instrumental_variable_exclusion_cues import InstrumentalVariableExclusionCueExtractor


def test_empty() -> None:
    """Empty input returns empty."""

    assert InstrumentalVariableExclusionCueExtractor().extract([]) == ()


def test_detects_cue() -> None:
    """Detects primary cue phrase."""

    rows = InstrumentalVariableExclusionCueExtractor().extract(
        [
            {
                "paper_id": "p1",
                "abstract": "We discuss instrumental variable exclusion.",
            }
        ]
    )
    assert rows[0].flagged is True
    assert "instrumental_variable_exclusion" in rows[0].cues


def test_no_match() -> None:
    """Unrelated text is not flagged."""

    rows = InstrumentalVariableExclusionCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "No relevant methods described."}]
    )
    assert rows[0].flagged is False
