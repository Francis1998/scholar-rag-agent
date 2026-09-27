"""Unit tests for BaselineImbalanceCueExtractor."""

from __future__ import annotations

from retrieval.baseline_imbalance_cues import BaselineImbalanceCueExtractor


def test_baseline_imbalance_flagged() -> None:
    """Baseline imbalance language is flagged."""

    rows = BaselineImbalanceCueExtractor().extract(
        [{"paper_id": "p1", "abstract": "There was baseline imbalance in age."}]
    )
    assert rows[0].flagged is True
    assert rows[0].cue_kind == "baseline_imbalance"


def test_covariate_imbalance_flagged() -> None:
    """Covariate imbalance language is flagged."""

    rows = BaselineImbalanceCueExtractor().extract(
        [{"paper_id": "p2", "results": "Covariate imbalance required adjustment."}]
    )
    assert rows[0].flagged is True
    assert "covariate_imbalance" in rows[0].cues


def test_no_cue() -> None:
    """Balanced trial abstract is not flagged."""

    rows = BaselineImbalanceCueExtractor().extract(
        [{"paper_id": "p3", "abstract": "Groups were well balanced at baseline."}]
    )
    assert rows[0].flagged is False


def test_empty_input() -> None:
    """Empty paper list returns empty tuple."""

    assert BaselineImbalanceCueExtractor().extract([]) == ()
