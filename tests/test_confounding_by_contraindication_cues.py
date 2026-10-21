"""Tests for ConfoundingByContraindicationCueExtractor."""

from __future__ import annotations

from retrieval.confounding_by_contraindication_cues import ConfoundingByContraindicationCueExtractor


def test_empty() -> None:
    """Empty input returns empty tuple."""

    assert ConfoundingByContraindicationCueExtractor().extract([]) == ()


def test_flagged() -> None:
    """Known phrase flags paper."""

    rows = ConfoundingByContraindicationCueExtractor().extract(
        [
            {
                "paper_id": "p1",
                "abstract": "Analysis discusses confounding by contraindication carefully.",
            }
        ]
    )
    assert len(rows) == 1
    assert rows[0].flagged is True
    assert "confounding_by_contraindication" in rows[0].cues


def test_unflagged() -> None:
    """Unrelated abstract stays clean."""

    rows = ConfoundingByContraindicationCueExtractor().extract(
        [{"id": "p2", "title": "Routine cohort description without bias terms."}]
    )
    assert rows[0].flagged is False
