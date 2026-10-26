"""Unit tests for MeasurementScaleShiftCueExtractor."""

from __future__ import annotations

from retrieval.measurement_scale_shift_cues import MeasurementScaleShiftCueExtractor


def test_empty() -> None:
    """Empty input yields empty tuple."""

    assert MeasurementScaleShiftCueExtractor().extract([]) == ()


def test_flagged() -> None:
    """Matching abstract is flagged."""

    abstract = "Evidence of measurement-scale shift after instrument rescaling."
    rows = MeasurementScaleShiftCueExtractor().extract(
        [{"paper_id": "p1", "abstract": abstract}],
    )
    assert rows[0].flagged is True
    assert "measurement_scale_shift" in rows[0].cues


def test_unflagged() -> None:
    """Unrelated abstract is not flagged."""

    rows = MeasurementScaleShiftCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "Routine descriptive statistics only."}],
    )
    assert rows[0].flagged is False
