"""Tests for FalsificationTestResidualCueExtractor."""

from __future__ import annotations

from retrieval.falsification_test_residual_cues import FalsificationTestResidualCueExtractor


def test_extracts_cue() -> None:
    """Match expected cue phrase."""

    rows = FalsificationTestResidualCueExtractor().extract(
        [{"paper_id": "p1", "abstract": "We discuss falsification test residual."}]
    )
    assert rows[0].flagged is True
    assert "falsification_test_residual" in rows[0].cues


def test_empty_papers() -> None:
    """Empty input returns empty tuple."""

    assert FalsificationTestResidualCueExtractor().extract([]) == ()


def test_no_match() -> None:
    """Unrelated abstract is not flagged."""

    rows = FalsificationTestResidualCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "No relevant methods described."}]
    )
    assert rows[0].flagged is False
