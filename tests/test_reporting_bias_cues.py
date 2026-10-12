"""Unit tests for ReportingBiasCueExtractor."""

from __future__ import annotations

from retrieval.reporting_bias_cues import ReportingBiasCueExtractor


def test_flagged() -> None:
    """Flag known cue wording."""

    cues = ReportingBiasCueExtractor().extract(
        [
            {
                "paper_id": "p1",
                "abstract": "Meta-analyses may suffer from reporting bias in published endpoints.",
            }
        ]
    )
    assert cues[0].flagged is True
    assert cues[0].cue_kind == "reporting_bias"


def test_empty() -> None:
    """Empty input returns empty tuple."""

    assert ReportingBiasCueExtractor().extract([]) == ()


def test_unflagged() -> None:
    """Neutral abstract is not flagged."""

    cues = ReportingBiasCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "A randomized trial of usual care."}]
    )
    assert cues[0].flagged is False
