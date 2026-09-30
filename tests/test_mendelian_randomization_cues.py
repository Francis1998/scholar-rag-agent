"""Unit tests for MendelianRandomizationCueExtractor."""

from __future__ import annotations

from retrieval.mendelian_randomization_cues import MendelianRandomizationCueExtractor


def test_empty() -> None:
    """Empty input returns empty tuple."""

    assert MendelianRandomizationCueExtractor().extract([]) == ()


def test_flagged() -> None:
    """Matching methods text is flagged."""

    rows = MendelianRandomizationCueExtractor().extract(
        [
            {
                "paper_id": "p1",
                "methods": "We performed Mendelian randomization with MR-Egger for pleiotropy.",
            }
        ]
    )
    assert rows[0].flagged is True
    assert rows[0].cues


def test_unflagged() -> None:
    """Unrelated text is not flagged."""

    rows = MendelianRandomizationCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "A narrative review of nursing education."}]
    )
    assert rows[0].flagged is False


def test_id_fallback() -> None:
    """Missing ids fall back to paper-index."""

    rows = MendelianRandomizationCueExtractor().extract([{"title": "x"}])
    assert rows[0].paper_id == "paper-0"
