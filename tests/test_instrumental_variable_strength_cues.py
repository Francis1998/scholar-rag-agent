"""Unit tests for InstrumentalVariableStrengthCueExtractor."""

from __future__ import annotations

from retrieval.instrumental_variable_strength_cues import InstrumentalVariableStrengthCueExtractor


def test_empty() -> None:
    """Empty input returns empty tuple."""

    assert InstrumentalVariableStrengthCueExtractor().extract([]) == ()


def test_flagged() -> None:
    """Matching methods text is flagged."""

    rows = InstrumentalVariableStrengthCueExtractor().extract(
        [
            {
                "paper_id": "p1",
                "methods": "The first-stage F-statistic suggested a weak instrument.",
            }
        ]
    )
    assert rows[0].flagged is True
    assert rows[0].cues


def test_unflagged() -> None:
    """Unrelated text is not flagged."""

    rows = InstrumentalVariableStrengthCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "A narrative review of nursing education."}]
    )
    assert rows[0].flagged is False


def test_id_fallback() -> None:
    """Missing ids fall back to paper-index."""

    rows = InstrumentalVariableStrengthCueExtractor().extract([{"title": "x"}])
    assert rows[0].paper_id == "paper-0"
