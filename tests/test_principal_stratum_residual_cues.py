"""Tests for PrincipalStratumResidualCueExtractor."""

from __future__ import annotations

from retrieval.principal_stratum_residual_cues import PrincipalStratumResidualCueExtractor


def test_extracts_cue() -> None:
    """Match expected cue phrase."""

    rows = PrincipalStratumResidualCueExtractor().extract(
        [{"paper_id": "p1", "abstract": "We discuss principal stratum residual."}]
    )
    assert rows[0].flagged is True
    assert "principal_stratum_residual" in rows[0].cues


def test_empty_papers() -> None:
    """Empty input returns empty tuple."""

    assert PrincipalStratumResidualCueExtractor().extract([]) == ()


def test_no_match() -> None:
    """Unrelated abstract is not flagged."""

    rows = PrincipalStratumResidualCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "No relevant methods described."}]
    )
    assert rows[0].flagged is False
