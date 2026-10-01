"""Unit tests for SyntheticControlCueExtractor."""

from __future__ import annotations

from retrieval.synthetic_control_cues import SyntheticControlCueExtractor


def test_empty() -> None:
    """Empty input returns empty tuple."""

    assert SyntheticControlCueExtractor().extract([]) == ()


def test_flagged() -> None:
    """Matching methods text is flagged."""

    rows = SyntheticControlCueExtractor().extract(
        [
            {
                "paper_id": "p1",
                "methods": (
                    "We built a synthetic control from the donor pool with placebo-in-space checks."
                ),
            }
        ]
    )
    assert rows[0].flagged is True
    assert rows[0].cues


def test_unflagged() -> None:
    """Unrelated text is not flagged."""

    rows = SyntheticControlCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "A narrative review of nursing education."}]
    )
    assert rows[0].flagged is False


def test_id_fallback() -> None:
    """Missing ids fall back to paper-index."""

    rows = SyntheticControlCueExtractor().extract([{"title": "x"}])
    assert rows[0].paper_id == "paper-0"
