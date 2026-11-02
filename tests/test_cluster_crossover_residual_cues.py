"""Tests for ClusterCrossoverResidualCueExtractor."""

from __future__ import annotations

from retrieval.cluster_crossover_residual_cues import ClusterCrossoverResidualCueExtractor


def test_empty() -> None:
    """Empty input returns empty."""

    assert ClusterCrossoverResidualCueExtractor().extract([]) == ()


def test_detects_cue() -> None:
    """Detects primary cue phrase."""

    rows = ClusterCrossoverResidualCueExtractor().extract(
        [
            {
                "paper_id": "p1",
                "abstract": "We discuss cluster crossover residual.",
            }
        ]
    )
    assert rows[0].flagged is True
    assert "cluster_crossover_residual" in rows[0].cues


def test_no_match() -> None:
    """Unrelated text is not flagged."""

    rows = ClusterCrossoverResidualCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "No relevant methods described."}]
    )
    assert rows[0].flagged is False
