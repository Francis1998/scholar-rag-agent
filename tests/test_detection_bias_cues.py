"""Tests for DetectionBiasCueExtractor."""

from __future__ import annotations

from retrieval.detection_bias_cues import DetectionBiasCueExtractor


def test_empty() -> None:
    """Empty input returns empty tuple."""

    assert DetectionBiasCueExtractor().extract([]) == ()


def test_flags_match() -> None:
    """Matching abstract is flagged."""

    rows = [
        {
            "paper_id": "p1",
            "abstract": "We discuss detection bias from differential screening intensity.",
        }
    ]
    out = DetectionBiasCueExtractor().extract(rows)
    assert len(out) == 1
    assert out[0].flagged is True
    assert out[0].cue_kind == "detection_bias"


def test_no_match() -> None:
    """Unrelated abstract is not flagged."""

    rows = [{"paper_id": "p2", "abstract": "No relevant bias language here."}]
    out = DetectionBiasCueExtractor().extract(rows)
    assert out[0].flagged is False
