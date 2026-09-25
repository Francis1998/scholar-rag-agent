"""Unit tests for NonInferiorityMarginCueExtractor."""

from __future__ import annotations

from retrieval.noninferiority_margin_cues import NonInferiorityMarginCueExtractor


def test_flags_noninferiority() -> None:
    """Non-inferiority phrase is flagged."""

    rows = NonInferiorityMarginCueExtractor().extract(
        [{"paper_id": "p1", "abstract": "This was a non-inferiority trial versus SOC."}]
    )
    assert rows[0].flagged is True
    assert rows[0].cue_kind == "noninferiority"


def test_flags_margin() -> None:
    """Non-inferiority margin phrase is flagged."""

    rows = NonInferiorityMarginCueExtractor().extract(
        [{"id": "p2", "methods": "The non-inferiority margin was 5%."}]
    )
    assert rows[0].flagged is True
    assert "margin_delta" in rows[0].cues


def test_empty_papers() -> None:
    """Empty input returns empty tuple."""

    assert NonInferiorityMarginCueExtractor().extract([]) == ()


def test_no_match() -> None:
    """Unrelated abstract is not flagged."""

    rows = NonInferiorityMarginCueExtractor().extract(
        [{"paper_id": "p3", "abstract": "A superiority RCT of drug X."}]
    )
    assert rows[0].flagged is False
