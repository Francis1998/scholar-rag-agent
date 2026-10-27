"""Unit tests for AscertainmentDifferentialResidualCueExtractor."""

from __future__ import annotations

from retrieval.ascertainment_differential_residual_cues import (
    AscertainmentDifferentialResidualCueExtractor,
)


def test_empty() -> None:
    """Empty input yields empty tuple."""

    assert AscertainmentDifferentialResidualCueExtractor().extract([]) == ()


def test_flagged() -> None:
    """Matching abstract is flagged."""

    abstract = (
        "Ascertainment Differential Residual may bias effect estimates in observational cohorts."
    )
    rows = AscertainmentDifferentialResidualCueExtractor().extract(
        [{"paper_id": "p1", "abstract": abstract}],
    )
    assert rows[0].flagged is True
    assert "ascertainment_differential_residual" in rows[0].cues


def test_unflagged() -> None:
    """Unrelated abstract is not flagged."""

    rows = AscertainmentDifferentialResidualCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "Routine descriptive statistics only."}],
    )
    assert rows[0].flagged is False
