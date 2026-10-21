"""Tests for DifferentialMisclassificationCueExtractor."""

from __future__ import annotations

from retrieval.differential_misclassification_cues import DifferentialMisclassificationCueExtractor


def test_empty() -> None:
    """Empty input returns empty tuple."""

    assert DifferentialMisclassificationCueExtractor().extract([]) == ()


def test_flagged() -> None:
    """Known phrase flags paper."""

    rows = DifferentialMisclassificationCueExtractor().extract(
        [{"paper_id": "p1", "abstract": "Study notes differential misclassification in methods."}]
    )
    assert len(rows) == 1
    assert rows[0].flagged is True
    assert "differential_misclassification" in rows[0].cues


def test_unflagged() -> None:
    """Unrelated abstract stays clean."""

    rows = DifferentialMisclassificationCueExtractor().extract(
        [{"id": "p2", "title": "Routine cohort description without bias terms."}]
    )
    assert rows[0].flagged is False
