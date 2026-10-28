"""Unit tests for InstrumentStrengthCueExtractor."""

from __future__ import annotations

from retrieval.instrument_strength_cues import (
    InstrumentStrengthCueExtractor,
)


def test_empty() -> None:
    """Empty input yields empty tuple."""

    assert InstrumentStrengthCueExtractor().extract([]) == ()


def test_flagged() -> None:
    """Matching abstract is flagged."""

    abstract = "Instrument Strength and first-stage F-stat were reported for the IV analysis."
    rows = InstrumentStrengthCueExtractor().extract(
        [{"paper_id": "p1", "abstract": abstract}],
    )
    assert rows[0].flagged is True
    assert "instrument_strength" in rows[0].cues


def test_unflagged() -> None:
    """Unrelated abstract is not flagged."""

    rows = InstrumentStrengthCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "Routine descriptive statistics only."}],
    )
    assert rows[0].flagged is False
