"""Tests for AdaptiveEnrichmentCueExtractor."""

from __future__ import annotations

from retrieval.adaptive_enrichment_cues import AdaptiveEnrichmentCueExtractor


def test_empty() -> None:
    """Empty input returns empty."""

    assert AdaptiveEnrichmentCueExtractor().extract([]) == ()


def test_detects_cue() -> None:
    """Detects primary cue phrase."""

    rows = AdaptiveEnrichmentCueExtractor().extract(
        [{"paper_id": "p1", "abstract": "We discuss adaptive enrichment in the cohort."}]
    )
    assert rows[0].flagged is True
    assert "adaptive_enrichment" in rows[0].cues


def test_no_match() -> None:
    """Unrelated text is not flagged."""

    rows = AdaptiveEnrichmentCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "No relevant methods described."}]
    )
    assert rows[0].flagged is False
