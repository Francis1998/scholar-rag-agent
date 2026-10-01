"""Unit tests for DifferenceInDifferencesCueExtractor."""

from __future__ import annotations

from retrieval.difference_in_differences_cues import DifferenceInDifferencesCueExtractor


def test_empty() -> None:
    """Empty input returns empty tuple."""

    assert DifferenceInDifferencesCueExtractor().extract([]) == ()


def test_flagged() -> None:
    """Matching methods text is flagged."""

    rows = DifferenceInDifferencesCueExtractor().extract(
        [
            {
                "paper_id": "p1",
                "methods": (
                    "We estimated difference-in-differences under parallel trends with TWFE."
                ),
            }
        ]
    )
    assert rows[0].flagged is True
    assert rows[0].cues


def test_unflagged() -> None:
    """Unrelated text is not flagged."""

    rows = DifferenceInDifferencesCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "A narrative review of nursing education."}]
    )
    assert rows[0].flagged is False


def test_id_fallback() -> None:
    """Missing ids fall back to paper-index."""

    rows = DifferenceInDifferencesCueExtractor().extract([{"title": "x"}])
    assert rows[0].paper_id == "paper-0"
