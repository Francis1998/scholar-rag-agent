"""Unit tests for TransportabilityCueExtractor."""

from __future__ import annotations

from retrieval.transportability_cues import TransportabilityCueExtractor


def test_empty_papers() -> None:
    """Empty input returns empty tuple."""

    assert TransportabilityCueExtractor().extract([]) == ()


def test_flagged_match() -> None:
    """Matching methods text is flagged."""

    rows = TransportabilityCueExtractor().extract(
        [
            {
                "paper_id": "p1",
                "methods": (
                    "We assessed transportability to the target population for external validity."
                ),
            }
        ]
    )
    assert rows[0].flagged is True
    assert "transportability" in rows[0].cues


def test_no_match() -> None:
    """Unrelated text is not flagged."""

    rows = TransportabilityCueExtractor().extract(
        [{"paper_id": "p1", "abstract": "No relevant design cues."}]
    )
    assert rows[0].flagged is False


def test_resolves_doi() -> None:
    """Falls back to doi when paper_id missing."""

    rows = TransportabilityCueExtractor().extract(
        [
            {
                "doi": "10.1/x",
                "methods": (
                    "We assessed transportability to the target population for external validity."
                ),
            }
        ]
    )
    assert rows[0].paper_id == "10.1/x"
