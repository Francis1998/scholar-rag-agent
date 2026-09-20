"""Tests for IntentionToTreatCueExtractor."""

from retrieval.intention_to_treat_cues import IntentionToTreatCueExtractor


def test_empty_ok() -> None:
    assert IntentionToTreatCueExtractor().extract([]) == ()


def test_itt() -> None:
    rows = IntentionToTreatCueExtractor().extract(
        [{"paper_id": "p1", "methods": "Analyses followed the intention-to-treat principle."}]
    )
    assert rows[0].flagged is True
    assert "intention_to_treat" in rows[0].cues


def test_per_protocol() -> None:
    rows = IntentionToTreatCueExtractor().extract(
        [{"id": "p2", "abstract": "Sensitivity used a per-protocol population."}]
    )
    assert rows[0].flagged is True
    assert "per_protocol" in rows[0].cues


def test_clear() -> None:
    rows = IntentionToTreatCueExtractor().extract(
        [{"paper_id": "clear", "abstract": "We report a case series of five patients."}]
    )
    assert rows[0].flagged is False
