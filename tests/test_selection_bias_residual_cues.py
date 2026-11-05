"""Tests for SelectionBiasResidualCueExtractor."""

from __future__ import annotations

from retrieval.selection_bias_residual_cues import (
    SelectionBiasResidualCueExtractor,
)


def test_empty() -> None:
    """Empty input returns empty."""

    assert SelectionBiasResidualCueExtractor().extract([]) == ()


def test_match() -> None:
    """Matching abstract is flagged."""

    rows = SelectionBiasResidualCueExtractor().extract(
        [{"paper_id": "p1", "abstract": "We discuss selection bias residual."}]
    )
    assert rows[0].flagged is True
    assert rows[0].cues


def test_no_match() -> None:
    """Unrelated abstract is not flagged."""

    rows = SelectionBiasResidualCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "No relevant methodology discussed."}]
    )
    assert rows[0].flagged is False


def test_missing_id() -> None:
    """Missing paper_id falls back."""

    rows = SelectionBiasResidualCueExtractor().extract([{"abstract": "selection bias residual"}])
    assert rows[0].paper_id.startswith("paper-")
