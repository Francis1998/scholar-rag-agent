"""Unit tests for ColliderBiasInteractionCueExtractor."""

from __future__ import annotations

from retrieval.collider_bias_interaction_cues import (
    ColliderBiasInteractionCueExtractor,
)


def test_empty() -> None:
    """Empty input yields empty tuple."""

    assert ColliderBiasInteractionCueExtractor().extract([]) == ()


def test_flagged() -> None:
    """Matching abstract is flagged."""

    abstract = "Collider Bias Interaction may bias effect estimates in observational cohorts."
    rows = ColliderBiasInteractionCueExtractor().extract(
        [{"paper_id": "p1", "abstract": abstract}],
    )
    assert rows[0].flagged is True
    assert "collider_bias_interaction" in rows[0].cues


def test_unflagged() -> None:
    """Unrelated abstract is not flagged."""

    rows = ColliderBiasInteractionCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "Routine descriptive statistics only."}],
    )
    assert rows[0].flagged is False
