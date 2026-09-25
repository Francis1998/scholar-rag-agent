"""Unit tests for ProtocolDeviationCueExtractor."""

from __future__ import annotations

from retrieval.protocol_deviation_cues import ProtocolDeviationCueExtractor

_ABS = "Several protocol deviations were recorded."


def test_flagged_on_match() -> None:
    """Known cue text is flagged."""

    rows = ProtocolDeviationCueExtractor().extract([{"paper_id": "p1", "abstract": _ABS}])
    assert rows[0].flagged is True
    assert rows[0].cues


def test_empty_input() -> None:
    """Empty paper list returns empty tuple."""

    assert ProtocolDeviationCueExtractor().extract([]) == ()


def test_unflagged_without_cues() -> None:
    """Benign abstract is not flagged."""

    rows = ProtocolDeviationCueExtractor().extract(
        [{"paper_id": "p2", "abstract": "We report general laboratory methods."}]
    )
    assert rows[0].flagged is False


def test_resolves_doi_id() -> None:
    """Falls back to doi when paper_id missing."""

    rows = ProtocolDeviationCueExtractor().extract([{"doi": "10.1/x", "abstract": _ABS}])
    assert rows[0].paper_id == "10.1/x"
