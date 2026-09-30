"""Unit tests for TimeVaryingConfoundingCueExtractor."""

from __future__ import annotations

from retrieval.time_varying_confounding_cues import TimeVaryingConfoundingCueExtractor


def test_empty() -> None:
    """Empty input returns empty tuple."""

    assert TimeVaryingConfoundingCueExtractor().extract([]) == ()


def test_flagged() -> None:
    """Matching methods text is flagged."""

    rows = TimeVaryingConfoundingCueExtractor().extract(
        [
            {
                "paper_id": "p1",
                "methods": "We fit a marginal structural model for time-varying confounding.",
            }
        ]
    )
    assert rows[0].flagged is True
    assert rows[0].cues


def test_unflagged() -> None:
    """Unrelated text is not flagged."""

    rows = TimeVaryingConfoundingCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "A narrative review of nursing education."}]
    )
    assert rows[0].flagged is False


def test_id_fallback() -> None:
    """Missing ids fall back to paper-index."""

    rows = TimeVaryingConfoundingCueExtractor().extract([{"title": "x"}])
    assert rows[0].paper_id == "paper-0"
