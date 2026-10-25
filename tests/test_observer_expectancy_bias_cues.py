"""Unit tests for ObserverExpectancyBiasCueExtractor."""

from __future__ import annotations

from retrieval.observer_expectancy_bias_cues import ObserverExpectancyBiasCueExtractor


def test_empty() -> None:
    """Empty input yields empty tuple."""

    assert ObserverExpectancyBiasCueExtractor().extract([]) == ()


def test_flagged() -> None:
    """Matching abstract is flagged."""

    rows = ObserverExpectancyBiasCueExtractor().extract(
        [{"paper_id": "p1", "abstract": "Risk of observer expectancy bias without blinding."}],
    )
    assert rows[0].flagged is True
    assert "observer_expectancy" in rows[0].cues


def test_unflagged() -> None:
    """Unrelated abstract is not flagged."""

    rows = ObserverExpectancyBiasCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "Routine descriptive statistics only."}],
    )
    assert rows[0].flagged is False
