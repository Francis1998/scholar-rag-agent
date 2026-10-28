"""Unit tests for HealthyWorkerSurvivorCueExtractor."""

from __future__ import annotations

from retrieval.healthy_worker_survivor_cues import (
    HealthyWorkerSurvivorCueExtractor,
)


def test_empty() -> None:
    """Empty input yields empty tuple."""

    assert HealthyWorkerSurvivorCueExtractor().extract([]) == ()


def test_flagged() -> None:
    """Matching abstract is flagged."""

    abstract = "Healthy Worker Survivor bias may attenuate occupational exposure associations."
    rows = HealthyWorkerSurvivorCueExtractor().extract(
        [{"paper_id": "p1", "abstract": abstract}],
    )
    assert rows[0].flagged is True
    assert "healthy_worker_survivor" in rows[0].cues


def test_unflagged() -> None:
    """Unrelated abstract is not flagged."""

    rows = HealthyWorkerSurvivorCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "Routine descriptive statistics only."}],
    )
    assert rows[0].flagged is False
