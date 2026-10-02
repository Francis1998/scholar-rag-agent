"""Unit tests for PropensityScoreMatchingCueExtractor."""

from __future__ import annotations

from retrieval.propensity_score_matching_cues import PropensityScoreMatchingCueExtractor


def test_empty() -> None:
    """Empty input returns empty tuple."""

    assert PropensityScoreMatchingCueExtractor().extract([]) == ()


def test_flagged() -> None:
    """Matching methods text is flagged."""

    rows = PropensityScoreMatchingCueExtractor().extract(
        [
            {
                "paper_id": "p1",
                "methods": (
                    "We applied propensity score matching with "
                    "a caliper and checked standardized mean difference balance."
                ),
            }
        ]
    )
    assert rows[0].flagged is True
    assert rows[0].cues


def test_unflagged() -> None:
    """Unrelated text is not flagged."""

    rows = PropensityScoreMatchingCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "A narrative review of nursing education."}]
    )
    assert rows[0].flagged is False


def test_id_fallback() -> None:
    """Missing ids fall back to paper-index."""

    rows = PropensityScoreMatchingCueExtractor().extract([{"title": "x"}])
    assert rows[0].paper_id == "paper-0"
