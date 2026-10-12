"""Unit tests for IndicationBiasCueExtractor."""

from __future__ import annotations

from retrieval.indication_bias_cues import IndicationBiasCueExtractor


def test_flagged() -> None:
    """Flag known cue wording."""

    cues = IndicationBiasCueExtractor().extract(
        [
            {
                "paper_id": "p1",
                "abstract": "Observational cohorts may show indication bias in treated arms.",
            }
        ]
    )
    assert cues[0].flagged is True
    assert cues[0].cue_kind == "indication_bias"


def test_empty() -> None:
    """Empty input returns empty tuple."""

    assert IndicationBiasCueExtractor().extract([]) == ()


def test_unflagged() -> None:
    """Neutral abstract is not flagged."""

    cues = IndicationBiasCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "A randomized trial of usual care."}]
    )
    assert cues[0].flagged is False
