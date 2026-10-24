"""Unit tests for RightTruncationBiasCueExtractor."""

from __future__ import annotations

from retrieval.right_truncation_bias_cues import RightTruncationBiasCueExtractor


def test_empty() -> None:
    """Empty input yields empty tuple."""

    assert RightTruncationBiasCueExtractor().extract([]) == ()


def test_flagged() -> None:
    """Matching abstract is flagged."""

    rows = RightTruncationBiasCueExtractor().extract(
        [{"paper_id": "p1", "abstract": "Evidence of right truncation in the cohort."}],
    )
    assert rows[0].flagged is True
    assert "right_truncation" in rows[0].cues


def test_unflagged() -> None:
    """Unrelated abstract is not flagged."""

    rows = RightTruncationBiasCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "Routine descriptive statistics only."}],
    )
    assert rows[0].flagged is False
