"""Unit tests for AtomisticFallacyCueExtractor."""

from __future__ import annotations

from retrieval.atomistic_fallacy_cues import AtomisticFallacyCueExtractor


def test_empty() -> None:
    """Empty input yields empty tuple."""

    assert AtomisticFallacyCueExtractor().extract([]) == ()


def test_flagged() -> None:
    """Matching abstract is flagged."""

    rows = AtomisticFallacyCueExtractor().extract(
        [{"paper_id": "p1", "abstract": "Evidence of atomistic fallacy in the cohort."}],
    )
    assert rows[0].flagged is True
    assert "atomistic_fallacy" in rows[0].cues


def test_unflagged() -> None:
    """Unrelated abstract is not flagged."""

    rows = AtomisticFallacyCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "Routine descriptive statistics only."}],
    )
    assert rows[0].flagged is False
