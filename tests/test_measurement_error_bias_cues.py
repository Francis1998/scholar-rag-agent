"""Unit tests for MeasurementErrorBiasCueExtractor."""

from __future__ import annotations

from retrieval.measurement_error_bias_cues import MeasurementErrorBiasCueExtractor


def test_extracts_hit() -> None:
    """Flag papers with cue wording."""

    cues = MeasurementErrorBiasCueExtractor().extract(
        [
            {
                "paper_id": "p1",
                "abstract": (
                    "Measurement error and misclassification bias produced "
                    "information bias under classical measurement error."
                ),
            }
        ]
    )
    assert len(cues) == 1
    assert cues[0].flagged is True
    assert cues[0].cue_kind is not None


def test_empty_papers() -> None:
    """Empty input returns empty tuple."""

    assert MeasurementErrorBiasCueExtractor().extract([]) == ()


def test_no_match() -> None:
    """Unrelated abstract is not flagged."""

    cues = MeasurementErrorBiasCueExtractor().extract(
        [
            {
                "paper_id": "p2",
                "abstract": "A narrative review of nursing education.",
            }
        ]
    )
    assert cues[0].flagged is False
