"""Unit tests for AdaptiveDesignCueExtractor."""

from __future__ import annotations

from retrieval.adaptive_design_cues import AdaptiveDesignCueExtractor


def test_empty_papers() -> None:
    """Empty input returns empty tuple."""

    assert AdaptiveDesignCueExtractor().extract([]) == ()


def test_flagged_match() -> None:
    """Matching methods text is flagged."""

    rows = AdaptiveDesignCueExtractor().extract(
        [{"paper_id": "p1", "methods": "An interim sample-size re-estimation was planned."}]
    )
    assert rows[0].flagged is True
    assert "sample_size_reestimation" in rows[0].cues


def test_no_match() -> None:
    """Unrelated text is not flagged."""

    rows = AdaptiveDesignCueExtractor().extract(
        [{"paper_id": "p1", "abstract": "No relevant design cues."}]
    )
    assert rows[0].flagged is False


def test_resolves_doi() -> None:
    """Falls back to doi when paper_id missing."""

    rows = AdaptiveDesignCueExtractor().extract(
        [{"doi": "10.1/x", "methods": "An interim sample-size re-estimation was planned."}]
    )
    assert rows[0].paper_id == "10.1/x"
