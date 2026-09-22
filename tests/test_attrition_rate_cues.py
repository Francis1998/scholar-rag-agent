"""Tests for AttritionRateCueExtractor."""

from retrieval.attrition_rate_cues import AttritionRateCueExtractor


def test_empty_ok() -> None:
    assert AttritionRateCueExtractor().extract([]) == ()


def test_dropout() -> None:
    rows = AttritionRateCueExtractor().extract(
        [{"paper_id": "p1", "results": "Twelve dropouts occurred before week 12."}]
    )
    assert rows[0].flagged is True
    assert rows[0].attrition_kind == "dropout"


def test_withdrawal() -> None:
    rows = AttritionRateCueExtractor().extract(
        [{"id": "p2", "methods": "Participants who withdrew were excluded."}]
    )
    assert rows[0].flagged is True
    assert "withdrawal" in rows[0].cues


def test_clear() -> None:
    rows = AttritionRateCueExtractor().extract(
        [{"paper_id": "clear", "abstract": "We report observational cohort findings."}]
    )
    assert rows[0].flagged is False
