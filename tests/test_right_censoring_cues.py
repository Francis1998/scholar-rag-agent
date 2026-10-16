"""Tests for RightCensoringCueExtractor."""

from __future__ import annotations

from retrieval.right_censoring_cues import RightCensoringCueExtractor


def test_empty() -> None:
    """Empty input returns empty tuple."""

    assert RightCensoringCueExtractor().extract([]) == ()


def test_flags_match() -> None:
    """Matching abstract is flagged."""

    rows = [
        {
            "paper_id": "p1",
            "abstract": "We discuss right censoring from end-of-study administrative cuts.",
        }
    ]
    out = RightCensoringCueExtractor().extract(rows)
    assert len(out) == 1
    assert out[0].flagged is True
    assert out[0].cue_kind == "right_censoring"


def test_no_match() -> None:
    """Unrelated abstract is not flagged."""

    rows = [{"paper_id": "p2", "abstract": "No relevant bias language here."}]
    out = RightCensoringCueExtractor().extract(rows)
    assert out[0].flagged is False
