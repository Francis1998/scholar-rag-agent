"""Unit tests for CrossoverDesignCueExtractor."""

from __future__ import annotations

from retrieval.crossover_design_cues import CrossoverDesignCueExtractor


def test_crossover_flagged() -> None:
    """Crossover trial language is flagged."""

    rows = CrossoverDesignCueExtractor().extract(
        [{"paper_id": "p1", "abstract": "A randomized crossover trial of therapy."}]
    )
    assert rows[0].flagged is True
    assert rows[0].cue_kind == "crossover_trial"


def test_latin_square_flagged() -> None:
    """Latin square language is flagged."""

    rows = CrossoverDesignCueExtractor().extract(
        [{"paper_id": "p2", "methods": "We used a Latin square for period effects."}]
    )
    assert rows[0].flagged is True
    assert "latin_square" in rows[0].cues


def test_no_cue() -> None:
    """Parallel-group abstract is not flagged."""

    rows = CrossoverDesignCueExtractor().extract(
        [{"paper_id": "p3", "abstract": "A parallel-group randomized trial."}]
    )
    assert rows[0].flagged is False


def test_empty_input() -> None:
    """Empty paper list returns empty tuple."""

    assert CrossoverDesignCueExtractor().extract([]) == ()
