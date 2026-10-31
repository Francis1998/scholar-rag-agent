"""Tests for ProtocolAmendmentCueExtractor."""

from __future__ import annotations

from retrieval.protocol_amendment_cues import ProtocolAmendmentCueExtractor


def test_empty() -> None:
    """Empty input returns empty."""

    assert ProtocolAmendmentCueExtractor().extract([]) == ()


def test_detects_cue() -> None:
    """Detects primary cue phrase."""

    rows = ProtocolAmendmentCueExtractor().extract(
        [{"paper_id": "p1", "abstract": "We discuss protocol amendment in the cohort."}]
    )
    assert rows[0].flagged is True
    assert "protocol_amendment" in rows[0].cues


def test_no_match() -> None:
    """Unrelated text is not flagged."""

    rows = ProtocolAmendmentCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "No relevant methods described."}]
    )
    assert rows[0].flagged is False
