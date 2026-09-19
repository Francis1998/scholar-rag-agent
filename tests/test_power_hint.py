"""Tests for StatisticalPowerHintExtractor."""

from retrieval.power_hint import StatisticalPowerHintExtractor


def test_empty_ok() -> None:
    assert StatisticalPowerHintExtractor().extract([]) == ()


def test_power_percent() -> None:
    rows = StatisticalPowerHintExtractor().extract(
        [{"paper_id": "p1", "methods": "The study had 80% power to detect a difference."}]
    )
    assert rows[0].flagged is True
    assert rows[0].power_percent == 80
    assert "power_percent" in rows[0].cues


def test_power_calculation_cue() -> None:
    rows = StatisticalPowerHintExtractor().extract(
        [{"id": "p2", "abstract": "A priori power calculation guided enrollment."}]
    )
    assert rows[0].flagged is True
    assert "power_calculation" in rows[0].cues


def test_clear() -> None:
    rows = StatisticalPowerHintExtractor().extract(
        [{"paper_id": "clear", "abstract": "We report descriptive statistics only."}]
    )
    assert rows[0].flagged is False
