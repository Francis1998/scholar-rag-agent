"""Tests for EValueResidualCueExtractor."""

from __future__ import annotations

from retrieval.evalue_residual_cues import (
    EValueResidualCueExtractor,
)


def test_empty() -> None:
    """Empty input returns empty."""

    assert EValueResidualCueExtractor().extract([]) == ()


def test_match() -> None:
    """Matching abstract is flagged."""

    rows = EValueResidualCueExtractor().extract(
        [{"paper_id": "p1", "abstract": "We discuss evalue residual."}]
    )
    assert rows[0].flagged is True
    assert rows[0].cues


def test_no_match() -> None:
    """Unrelated abstract is not flagged."""

    rows = EValueResidualCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "No relevant methodology discussed."}]
    )
    assert rows[0].flagged is False


def test_missing_id() -> None:
    """Missing paper_id falls back."""

    rows = EValueResidualCueExtractor().extract([{"abstract": "evalue residual"}])
    assert rows[0].paper_id.startswith("paper-")
