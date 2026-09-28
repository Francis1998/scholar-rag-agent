"""Unit tests for PerProtocolAnalysisCueExtractor."""

from __future__ import annotations

from retrieval.per_protocol_analysis_cues import PerProtocolAnalysisCueExtractor


def test_empty_papers() -> None:
    """Empty input returns empty tuple."""

    assert PerProtocolAnalysisCueExtractor().extract([]) == ()


def test_flagged_match() -> None:
    """Matching methods text is flagged."""

    rows = PerProtocolAnalysisCueExtractor().extract(
        [{"paper_id": "p1", "methods": "Primary analysis used a per-protocol population."}]
    )
    assert rows[0].flagged is True
    assert "per_protocol_analysis" in rows[0].cues


def test_no_match() -> None:
    """Unrelated text is not flagged."""

    rows = PerProtocolAnalysisCueExtractor().extract(
        [{"paper_id": "p1", "abstract": "No relevant design cues."}]
    )
    assert rows[0].flagged is False


def test_resolves_doi() -> None:
    """Falls back to doi when paper_id missing."""

    rows = PerProtocolAnalysisCueExtractor().extract(
        [{"doi": "10.1/x", "methods": "Primary analysis used a per-protocol population."}]
    )
    assert rows[0].paper_id == "10.1/x"
