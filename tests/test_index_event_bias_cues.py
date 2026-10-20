"""Tests for IndexEventBiasCueExtractor."""

from __future__ import annotations

from retrieval.index_event_bias_cues import IndexEventBiasCueExtractor


def test_empty() -> None:
    """Empty input returns empty tuple."""

    assert IndexEventBiasCueExtractor().extract([]) == ()


def test_flagged() -> None:
    """Known phrase flags paper."""

    rows = IndexEventBiasCueExtractor().extract(
        [{"paper_id": "p1", "abstract": "Study notes index-event bias in methods."}]
    )
    assert len(rows) == 1
    assert rows[0].flagged is True
    assert "index_event_bias" in rows[0].cues


def test_unflagged() -> None:
    """Unrelated abstract stays clean."""

    rows = IndexEventBiasCueExtractor().extract(
        [{"id": "p2", "title": "Routine cohort description without bias terms."}]
    )
    assert rows[0].flagged is False
