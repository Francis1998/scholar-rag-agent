"""Unit tests for NonresponseBiasCueExtractor."""

from __future__ import annotations

from retrieval.nonresponse_bias_cues import NonresponseBiasCueExtractor


def test_empty() -> None:
    """Empty input yields empty tuple."""

    assert NonresponseBiasCueExtractor().extract([]) == ()


def test_flagged() -> None:
    """Matching abstract is flagged."""

    rows = NonresponseBiasCueExtractor().extract(
        [{"paper_id": "p1", "abstract": "Evidence of nonresponse bias in the cohort."}],
    )
    assert rows[0].flagged is True
    assert "nonresponse_bias" in rows[0].cues


def test_unflagged() -> None:
    """Unrelated abstract is not flagged."""

    rows = NonresponseBiasCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "Routine descriptive statistics only."}],
    )
    assert rows[0].flagged is False
