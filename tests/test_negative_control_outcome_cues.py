"""Unit tests for NegativeControlOutcomeCueExtractor."""

from __future__ import annotations

from retrieval.negative_control_outcome_cues import NegativeControlOutcomeCueExtractor


def test_empty() -> None:
    """Empty input returns empty tuple."""

    assert NegativeControlOutcomeCueExtractor().extract([]) == ()


def test_flagged() -> None:
    """Matching methods text is flagged."""

    rows = NegativeControlOutcomeCueExtractor().extract(
        [
            {
                "paper_id": "p1",
                "methods": ("We used a negative control outcome and falsification test for bias."),
            }
        ]
    )
    assert rows[0].flagged is True
    assert rows[0].cues


def test_unflagged() -> None:
    """Unrelated text is not flagged."""

    rows = NegativeControlOutcomeCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "A narrative review of nursing education."}]
    )
    assert rows[0].flagged is False


def test_id_fallback() -> None:
    """Missing ids fall back to paper-index."""

    rows = NegativeControlOutcomeCueExtractor().extract([{"title": "x"}])
    assert rows[0].paper_id == "paper-0"
