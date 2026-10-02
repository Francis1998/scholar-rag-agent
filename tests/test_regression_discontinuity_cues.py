"""Unit tests for RegressionDiscontinuityCueExtractor."""

from __future__ import annotations

from retrieval.regression_discontinuity_cues import RegressionDiscontinuityCueExtractor


def test_empty() -> None:
    """Empty input returns empty tuple."""

    assert RegressionDiscontinuityCueExtractor().extract([]) == ()


def test_flagged() -> None:
    """Matching methods text is flagged."""

    rows = RegressionDiscontinuityCueExtractor().extract(
        [
            {
                "paper_id": "p1",
                "methods": (
                    "We used a regression discontinuity design "
                    "with the running variable and Imbens-Kalyanaraman bandwidth."
                ),
            }
        ]
    )
    assert rows[0].flagged is True
    assert rows[0].cues


def test_unflagged() -> None:
    """Unrelated text is not flagged."""

    rows = RegressionDiscontinuityCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "A narrative review of nursing education."}]
    )
    assert rows[0].flagged is False


def test_id_fallback() -> None:
    """Missing ids fall back to paper-index."""

    rows = RegressionDiscontinuityCueExtractor().extract([{"title": "x"}])
    assert rows[0].paper_id == "paper-0"
