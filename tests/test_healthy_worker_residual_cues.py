"""Tests for HealthyWorkerResidualCueExtractor."""

from __future__ import annotations

from retrieval.healthy_worker_residual_cues import (
    HealthyWorkerResidualCueExtractor,
)


def test_empty() -> None:
    """Empty input returns empty."""

    assert HealthyWorkerResidualCueExtractor().extract([]) == ()


def test_match() -> None:
    """Matching abstract is flagged."""

    rows = HealthyWorkerResidualCueExtractor().extract(
        [{"paper_id": "p1", "abstract": "We discuss healthy worker residual."}]
    )
    assert rows[0].flagged is True
    assert rows[0].cues


def test_no_match() -> None:
    """Unrelated abstract is not flagged."""

    rows = HealthyWorkerResidualCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "No relevant methodology discussed."}]
    )
    assert rows[0].flagged is False


def test_missing_id() -> None:
    """Missing paper_id falls back."""

    rows = HealthyWorkerResidualCueExtractor().extract([{"abstract": "healthy worker residual"}])
    assert rows[0].paper_id.startswith("paper-")
