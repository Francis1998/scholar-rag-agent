"""Tests for SelectionIntoTreatmentCueExtractor."""

from __future__ import annotations

from retrieval.selection_into_treatment_cues import SelectionIntoTreatmentCueExtractor


def test_empty() -> None:
    """Empty input returns empty."""

    assert SelectionIntoTreatmentCueExtractor().extract([]) == ()


def test_detects_cue() -> None:
    """Detects primary cue phrase."""

    rows = SelectionIntoTreatmentCueExtractor().extract(
        [{"paper_id": "p1", "abstract": "We discuss selection into treatment in the cohort."}]
    )
    assert rows[0].flagged is True
    assert "selection_into_treatment" in rows[0].cues


def test_no_match() -> None:
    """Unrelated text is not flagged."""

    rows = SelectionIntoTreatmentCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "No relevant methods described."}]
    )
    assert rows[0].flagged is False
