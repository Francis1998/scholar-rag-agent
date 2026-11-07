"""Tests for SyntheticDiffInDiffResidualCueExtractor."""

from __future__ import annotations

from retrieval.synthetic_diff_in_diff_residual_cues import SyntheticDiffInDiffResidualCueExtractor


def test_empty() -> None:
    """Empty input returns empty tuple."""

    assert SyntheticDiffInDiffResidualCueExtractor().extract([]) == ()


def test_flagged() -> None:
    """Matching abstract is flagged."""

    rows = SyntheticDiffInDiffResidualCueExtractor().extract(
        [{"paper_id": "p1", "abstract": "We discuss synthetic diff in diff residual."}]
    )
    assert rows[0].flagged is True
    assert rows[0].cues


def test_unflagged() -> None:
    """Unrelated abstract is not flagged."""

    rows = SyntheticDiffInDiffResidualCueExtractor().extract([{"abstract": "unrelated methods"}])
    assert rows[0].flagged is False
