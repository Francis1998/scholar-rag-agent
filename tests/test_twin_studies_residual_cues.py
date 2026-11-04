"""Tests for TwinStudiesResidualCueExtractor."""

from __future__ import annotations

from retrieval.twin_studies_residual_cues import TwinStudiesResidualCueExtractor


def test_empty() -> None:
    """Empty input returns empty."""

    assert TwinStudiesResidualCueExtractor().extract([]) == ()


def test_match() -> None:
    """Matching abstract is flagged."""

    rows = TwinStudiesResidualCueExtractor().extract(
        [{"paper_id": "p1", "abstract": "We discuss twin studies residual."}]
    )
    assert rows[0].flagged is True
    assert rows[0].cues


def test_no_match() -> None:
    """Unrelated abstract is not flagged."""

    rows = TwinStudiesResidualCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "No relevant methodology discussed."}]
    )
    assert rows[0].flagged is False


def test_missing_id() -> None:
    """Missing paper_id falls back."""

    rows = TwinStudiesResidualCueExtractor().extract([{"abstract": "twin studies residual"}])
    assert rows[0].paper_id.startswith("paper-")
