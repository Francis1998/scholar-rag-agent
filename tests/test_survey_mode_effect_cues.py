"""Unit tests for SurveyModeEffectCueExtractor."""

from __future__ import annotations

from retrieval.survey_mode_effect_cues import (
    SurveyModeEffectCueExtractor,
)


def test_empty() -> None:
    """Empty input yields empty tuple."""

    assert SurveyModeEffectCueExtractor().extract([]) == ()


def test_flagged() -> None:
    """Matching abstract is flagged."""

    abstract = "Survey Mode Effect may bias effect estimates in observational cohorts."
    rows = SurveyModeEffectCueExtractor().extract(
        [{"paper_id": "p1", "abstract": abstract}],
    )
    assert rows[0].flagged is True
    assert "survey_mode_effect" in rows[0].cues


def test_unflagged() -> None:
    """Unrelated abstract is not flagged."""

    rows = SurveyModeEffectCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "Routine descriptive statistics only."}],
    )
    assert rows[0].flagged is False
