"""Unit tests for HealthyAdhererBiasCueExtractor."""

from __future__ import annotations

from retrieval.healthy_adherer_bias_cues import HealthyAdhererBiasCueExtractor


def test_empty() -> None:
    """Empty input yields empty tuple."""

    assert HealthyAdhererBiasCueExtractor().extract([]) == ()


def test_flagged() -> None:
    """Matching abstract is flagged."""

    rows = HealthyAdhererBiasCueExtractor().extract(
        [{"paper_id": "p1", "abstract": "Suspected healthy adherer bias in the adherence cohort."}],
    )
    assert rows[0].flagged is True
    assert "healthy_adherer" in rows[0].cues


def test_unflagged() -> None:
    """Unrelated abstract is not flagged."""

    rows = HealthyAdhererBiasCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "Routine descriptive statistics only."}],
    )
    assert rows[0].flagged is False
