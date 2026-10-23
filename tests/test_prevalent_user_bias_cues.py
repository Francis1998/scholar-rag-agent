"""Unit tests for PrevalentUserBiasCueExtractor."""

from __future__ import annotations

from retrieval.prevalent_user_bias_cues import PrevalentUserBiasCueExtractor


def test_empty() -> None:
    """Empty input yields empty tuple."""

    assert PrevalentUserBiasCueExtractor().extract([]) == ()


def test_flagged() -> None:
    """Matching abstract is flagged."""

    rows = PrevalentUserBiasCueExtractor().extract(
        [{"paper_id": "p1", "abstract": "Evidence of prevalent user bias in the cohort."}],
    )
    assert rows[0].flagged is True
    assert "prevalent_user_bias" in rows[0].cues


def test_unflagged() -> None:
    """Unrelated abstract is not flagged."""

    rows = PrevalentUserBiasCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "Routine descriptive statistics only."}],
    )
    assert rows[0].flagged is False
