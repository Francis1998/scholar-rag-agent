"""Unit tests for MissingDataMechanismCueExtractor."""

from __future__ import annotations

from retrieval.missing_data_mechanism_cues import MissingDataMechanismCueExtractor


def test_empty_papers() -> None:
    """Empty input returns empty tuple."""

    assert MissingDataMechanismCueExtractor().extract([]) == ()


def test_flagged_match() -> None:
    """Matching methods text is flagged."""

    rows = MissingDataMechanismCueExtractor().extract(
        [{"paper_id": "p1", "methods": "Outcomes were assumed missing at random (MAR)."}]
    )
    assert rows[0].flagged is True
    assert "mar_assumption" in rows[0].cues


def test_no_match() -> None:
    """Unrelated text is not flagged."""

    rows = MissingDataMechanismCueExtractor().extract(
        [{"paper_id": "p1", "abstract": "No relevant design cues."}]
    )
    assert rows[0].flagged is False


def test_resolves_doi() -> None:
    """Falls back to doi when paper_id missing."""

    rows = MissingDataMechanismCueExtractor().extract(
        [{"doi": "10.1/x", "methods": "Outcomes were assumed missing at random (MAR)."}]
    )
    assert rows[0].paper_id == "10.1/x"
