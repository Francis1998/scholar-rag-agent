"""Unit tests for TargetTrialEmulationCueExtractor."""

from __future__ import annotations

from retrieval.target_trial_emulation_cues import (
    TargetTrialEmulationCueExtractor,
)


def test_empty() -> None:
    """Empty input yields empty tuple."""

    assert TargetTrialEmulationCueExtractor().extract([]) == ()


def test_flagged() -> None:
    """Matching abstract is flagged."""

    abstract = "Target Trial Emulation may clarify causal contrasts in observational cohorts."
    rows = TargetTrialEmulationCueExtractor().extract(
        [{"paper_id": "p1", "abstract": abstract}],
    )
    assert rows[0].flagged is True
    assert "target_trial_emulation" in rows[0].cues


def test_unflagged() -> None:
    """Unrelated abstract is not flagged."""

    rows = TargetTrialEmulationCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "Routine descriptive statistics only."}],
    )
    assert rows[0].flagged is False
