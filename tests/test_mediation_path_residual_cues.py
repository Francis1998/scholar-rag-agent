"""Tests for MediationPathResidualCueExtractor."""

from __future__ import annotations

from retrieval.mediation_path_residual_cues import MediationPathResidualCueExtractor


def test_extracts_cue() -> None:
    """Match expected cue phrase."""

    rows = MediationPathResidualCueExtractor().extract(
        [{"paper_id": "p1", "abstract": "We discuss mediation path residual."}]
    )
    assert rows[0].flagged is True
    assert "mediation_path_residual" in rows[0].cues


def test_empty_papers() -> None:
    """Empty input returns empty tuple."""

    assert MediationPathResidualCueExtractor().extract([]) == ()


def test_no_match() -> None:
    """Unrelated abstract is not flagged."""

    rows = MediationPathResidualCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "No relevant methods described."}]
    )
    assert rows[0].flagged is False
