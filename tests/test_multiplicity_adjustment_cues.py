"""Tests for MultiplicityAdjustmentCueExtractor."""

from __future__ import annotations

from retrieval.multiplicity_adjustment_cues import MultiplicityAdjustmentCueExtractor


def test_extracts_cue() -> None:
    rows = MultiplicityAdjustmentCueExtractor().extract(
        [{"paper_id": "p1", "abstract": "P-values used Bonferroni correction."}]
    )
    assert rows[0].flagged is True
    assert rows[0].adjustment_kind is not None
    assert "bonferroni" in rows[0].cues or rows[0].cues


def test_empty_papers() -> None:
    assert MultiplicityAdjustmentCueExtractor().extract([]) == ()


def test_unflagged_without_match() -> None:
    rows = MultiplicityAdjustmentCueExtractor().extract(
        [{"id": "x", "abstract": "No relevant methods text."}]
    )
    assert rows[0].flagged is False
