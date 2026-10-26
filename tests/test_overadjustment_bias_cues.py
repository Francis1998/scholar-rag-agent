"""Unit tests for OveradjustmentBiasCueExtractor."""

from __future__ import annotations

from retrieval.overadjustment_bias_cues import OveradjustmentBiasCueExtractor


def test_empty() -> None:
    """Empty input yields empty tuple."""

    assert OveradjustmentBiasCueExtractor().extract([]) == ()


def test_flagged() -> None:
    """Matching abstract is flagged."""

    abstract = "Concern for overadjustment bias after adjusting for a mediator."
    rows = OveradjustmentBiasCueExtractor().extract(
        [{"paper_id": "p1", "abstract": abstract}],
    )
    assert rows[0].flagged is True
    assert "overadjustment_bias" in rows[0].cues


def test_unflagged() -> None:
    """Unrelated abstract is not flagged."""

    rows = OveradjustmentBiasCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "Routine descriptive statistics only."}],
    )
    assert rows[0].flagged is False
