"""Unit tests for InterimAnalysisCueExtractor."""

from __future__ import annotations

from retrieval.interim_analysis_cues import InterimAnalysisCueExtractor


def test_flags_interim_analysis() -> None:
    """Interim analysis phrase is flagged."""

    rows = InterimAnalysisCueExtractor().extract(
        [{"paper_id": "p1", "abstract": "A pre-specified interim analysis was planned."}]
    )
    assert rows[0].flagged is True
    assert rows[0].cue_kind == "interim_analysis"


def test_flags_dsmb() -> None:
    """DSMB cue is flagged."""

    rows = InterimAnalysisCueExtractor().extract(
        [{"id": "p2", "methods": "Oversight by the DSMB every 6 months."}]
    )
    assert rows[0].flagged is True
    assert "dsmb_review" in rows[0].cues


def test_empty_papers() -> None:
    """Empty input returns empty tuple."""

    assert InterimAnalysisCueExtractor().extract([]) == ()


def test_no_match() -> None:
    """Unrelated abstract is not flagged."""

    rows = InterimAnalysisCueExtractor().extract(
        [{"paper_id": "p3", "abstract": "A randomized trial of diet."}]
    )
    assert rows[0].flagged is False
