"""Tests for InstrumentalVariableWeakResidualCueExtractor."""

from __future__ import annotations

from retrieval.instrumental_variable_weak_residual_cues import (
    InstrumentalVariableWeakResidualCueExtractor,
)


def test_empty() -> None:
    """Empty input returns empty."""

    assert InstrumentalVariableWeakResidualCueExtractor().extract([]) == ()


def test_match() -> None:
    """Matching abstract is flagged."""

    rows = InstrumentalVariableWeakResidualCueExtractor().extract(
        [{"paper_id": "p1", "abstract": "We discuss iv weak residual."}]
    )
    assert rows[0].flagged is True
    assert rows[0].cues


def test_no_match() -> None:
    """Unrelated abstract is not flagged."""

    rows = InstrumentalVariableWeakResidualCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "No relevant methodology discussed."}]
    )
    assert rows[0].flagged is False


def test_missing_id() -> None:
    """Missing paper_id falls back."""

    rows = InstrumentalVariableWeakResidualCueExtractor().extract(
        [{"abstract": "iv weak residual"}]
    )
    assert rows[0].paper_id.startswith("paper-")
