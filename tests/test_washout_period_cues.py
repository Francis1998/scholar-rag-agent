"""Unit tests for WashoutPeriodCueExtractor."""

from __future__ import annotations

from retrieval.washout_period_cues import WashoutPeriodCueExtractor


def test_washout_flagged() -> None:
    """Washout period language is flagged."""

    rows = WashoutPeriodCueExtractor().extract(
        [{"paper_id": "p1", "abstract": "After a washout period of 14 days."}]
    )
    assert rows[0].flagged is True
    assert rows[0].cue_kind == "washout_period"


def test_run_in_flagged() -> None:
    """Run-in period language is flagged."""

    rows = WashoutPeriodCueExtractor().extract(
        [{"paper_id": "p2", "methods": "A placebo run-in period preceded randomization."}]
    )
    assert rows[0].flagged is True
    assert "run_in_period" in rows[0].cues


def test_no_cue() -> None:
    """Abstract without washout cues is not flagged."""

    rows = WashoutPeriodCueExtractor().extract(
        [{"paper_id": "p3", "abstract": "A parallel-group randomized trial."}]
    )
    assert rows[0].flagged is False


def test_empty_input() -> None:
    """Empty paper list returns empty tuple."""

    assert WashoutPeriodCueExtractor().extract([]) == ()
