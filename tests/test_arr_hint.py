"""Tests for AbsoluteRiskReductionHintExtractor."""

from retrieval.arr_hint import AbsoluteRiskReductionHintExtractor


def test_empty_ok() -> None:
    assert AbsoluteRiskReductionHintExtractor().extract([]) == ()


def test_extracts_arr_percent() -> None:
    rows = AbsoluteRiskReductionHintExtractor().extract(
        [{"paper_id": "p1", "abstract": "ARR = 3.2% favoring treatment."}]
    )
    assert rows[0].flagged is True
    assert rows[0].value == 3.2
    assert rows[0].unit == "%"


def test_extracts_risk_difference() -> None:
    rows = AbsoluteRiskReductionHintExtractor().extract(
        [{"id": "p2", "results": "Risk difference: 5 percentage points."}]
    )
    assert rows[0].flagged is True
    assert rows[0].value == 5.0


def test_clear_text() -> None:
    rows = AbsoluteRiskReductionHintExtractor().extract(
        [{"paper_id": "clear", "abstract": "We report relative risks only."}]
    )
    assert rows[0].flagged is False
