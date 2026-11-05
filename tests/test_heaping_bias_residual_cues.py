"""Tests for HeapingBiasResidualCueExtractor."""

from __future__ import annotations

from retrieval.heaping_bias_residual_cues import (
    HeapingBiasResidualCueExtractor,
)


def test_empty() -> None:
    """Empty input returns empty."""

    assert HeapingBiasResidualCueExtractor().extract([]) == ()


def test_match() -> None:
    """Matching abstract is flagged."""

    rows = HeapingBiasResidualCueExtractor().extract(
        [{"paper_id": "p1", "abstract": "We discuss heaping bias residual."}]
    )
    assert rows[0].flagged is True
    assert rows[0].cues


def test_no_match() -> None:
    """Unrelated abstract is not flagged."""

    rows = HeapingBiasResidualCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "No relevant methodology discussed."}]
    )
    assert rows[0].flagged is False


def test_missing_id() -> None:
    """Missing paper_id falls back."""

    rows = HeapingBiasResidualCueExtractor().extract([{"abstract": "heaping bias residual"}])
    assert rows[0].paper_id.startswith("paper-")
