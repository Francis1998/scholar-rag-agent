"""Tests for NonDifferentialMisclassificationCueExtractor."""

from __future__ import annotations

from retrieval.non_differential_misclassification_cues import (
    NonDifferentialMisclassificationCueExtractor,
)


def test_empty() -> None:
    """Empty input returns empty tuple."""

    assert NonDifferentialMisclassificationCueExtractor().extract([]) == ()


def test_flagged() -> None:
    """Known phrase flags paper."""

    rows = NonDifferentialMisclassificationCueExtractor().extract(
        [
            {
                "paper_id": "p1",
                "abstract": "Study notes non-differential misclassification in methods.",
            }
        ]
    )
    assert len(rows) == 1
    assert rows[0].flagged is True
    assert "non_differential_misclassification" in rows[0].cues


def test_unflagged() -> None:
    """Unrelated abstract stays clean."""

    rows = NonDifferentialMisclassificationCueExtractor().extract(
        [{"id": "p2", "title": "Routine cohort description without bias terms."}]
    )
    assert rows[0].flagged is False
