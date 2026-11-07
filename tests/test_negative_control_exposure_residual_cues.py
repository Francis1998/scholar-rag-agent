"""Tests for NegativeControlExposureResidualCueExtractor."""

from __future__ import annotations

from retrieval.negative_control_exposure_residual_cues import (
    NegativeControlExposureResidualCueExtractor,
)


def test_empty() -> None:
    """Empty input returns empty tuple."""

    assert NegativeControlExposureResidualCueExtractor().extract([]) == ()


def test_flagged() -> None:
    """Matching abstract is flagged."""

    rows = NegativeControlExposureResidualCueExtractor().extract(
        [{"paper_id": "p1", "abstract": "We discuss negative control exposure residual."}]
    )
    assert rows[0].flagged is True
    assert rows[0].cues


def test_unflagged() -> None:
    """Unrelated abstract is not flagged."""

    rows = NegativeControlExposureResidualCueExtractor().extract(
        [{"abstract": "unrelated methods"}]
    )
    assert rows[0].flagged is False
