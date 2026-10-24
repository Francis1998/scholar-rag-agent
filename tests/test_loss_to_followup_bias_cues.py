"""Unit tests for LossToFollowupBiasCueExtractor."""

from __future__ import annotations

from retrieval.loss_to_followup_bias_cues import LossToFollowupBiasCueExtractor


def test_empty() -> None:
    """Empty input yields empty tuple."""

    assert LossToFollowupBiasCueExtractor().extract([]) == ()


def test_flagged() -> None:
    """Matching abstract is flagged."""

    rows = LossToFollowupBiasCueExtractor().extract(
        [{"paper_id": "p1", "abstract": "Evidence of loss to follow-up in the cohort."}],
    )
    assert rows[0].flagged is True
    assert "loss_to_followup" in rows[0].cues


def test_unflagged() -> None:
    """Unrelated abstract is not flagged."""

    rows = LossToFollowupBiasCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "Routine descriptive statistics only."}],
    )
    assert rows[0].flagged is False
