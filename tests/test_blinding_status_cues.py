"""Tests for BlindingStatusCueExtractor."""

from retrieval.blinding_status_cues import BlindingStatusCueExtractor


def test_empty_ok() -> None:
    assert BlindingStatusCueExtractor().extract([]) == ()


def test_double_blind() -> None:
    rows = BlindingStatusCueExtractor().extract(
        [{"paper_id": "p1", "methods": "This was a double-blind randomized trial."}]
    )
    assert rows[0].flagged is True
    assert rows[0].blinding_level == "double_blind"
    assert "double_blind" in rows[0].cues


def test_open_label() -> None:
    rows = BlindingStatusCueExtractor().extract(
        [{"id": "p2", "abstract": "An open-label extension followed the RCT."}]
    )
    assert rows[0].flagged is True
    assert rows[0].blinding_level == "open_label"


def test_clear() -> None:
    rows = BlindingStatusCueExtractor().extract(
        [{"paper_id": "clear", "abstract": "We report observational cohort findings."}]
    )
    assert rows[0].flagged is False
