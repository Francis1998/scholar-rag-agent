"""Tests for AdverseEventCueExtractor."""

from retrieval.adverse_event_cues import AdverseEventCueExtractor


def test_empty_ok() -> None:
    assert AdverseEventCueExtractor().extract([]) == ()


def test_sae() -> None:
    rows = AdverseEventCueExtractor().extract(
        [{"paper_id": "p1", "results": "Three serious adverse events were reported."}]
    )
    assert rows[0].flagged is True
    assert rows[0].adverse_kind == "serious_adverse_event"


def test_ae() -> None:
    rows = AdverseEventCueExtractor().extract(
        [{"id": "p2", "abstract": "Adverse events were mild and transient."}]
    )
    assert rows[0].flagged is True
    assert "adverse_event" in rows[0].cues


def test_clear() -> None:
    rows = AdverseEventCueExtractor().extract(
        [{"paper_id": "clear", "abstract": "Efficacy was the primary focus."}]
    )
    assert rows[0].flagged is False
