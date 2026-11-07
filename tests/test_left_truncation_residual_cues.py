"""Tests for LeftTruncationResidualCueExtractor."""

from __future__ import annotations

from retrieval.left_truncation_residual_cues import LeftTruncationResidualCueExtractor


def test_empty() -> None:
    """Empty input returns empty tuple."""

    assert LeftTruncationResidualCueExtractor().extract([]) == ()


def test_flagged() -> None:
    """Matching abstract is flagged."""

    rows = LeftTruncationResidualCueExtractor().extract(
        [{"paper_id": "p1", "abstract": "We discuss left truncation residual."}]
    )
    assert rows[0].flagged is True
    assert rows[0].cues


def test_unflagged() -> None:
    """Unrelated abstract is not flagged."""

    rows = LeftTruncationResidualCueExtractor().extract([{"abstract": "unrelated methods"}])
    assert rows[0].flagged is False
