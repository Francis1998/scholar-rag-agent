"""Unit tests for NeymanBiasCueExtractor."""

from __future__ import annotations

from retrieval.neyman_bias_cues import NeymanBiasCueExtractor


def test_empty() -> None:
    """Empty input yields empty tuple."""

    assert NeymanBiasCueExtractor().extract([]) == ()


def test_flagged() -> None:
    """Matching abstract is flagged."""

    rows = NeymanBiasCueExtractor().extract(
        [{"paper_id": "p1", "abstract": "Evidence of Neyman bias from selective survival."}],
    )
    assert rows[0].flagged is True
    assert "neyman_bias" in rows[0].cues


def test_unflagged() -> None:
    """Unrelated abstract is not flagged."""

    rows = NeymanBiasCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "Routine descriptive statistics only."}],
    )
    assert rows[0].flagged is False
