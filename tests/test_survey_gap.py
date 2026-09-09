"""Tests for SurveyGapFinder."""

import pytest

from retrieval.survey_gap import SurveyGapFinder


def test_rejects_invalid_coverage_threshold() -> None:
    with pytest.raises(ValueError, match="coverage_threshold"):
        SurveyGapFinder(coverage_threshold=1.5)
    with pytest.raises(ValueError, match="coverage_threshold"):
        SurveyGapFinder(coverage_threshold=float("nan"))


def test_empty_papers_mark_all_themes_missing() -> None:
    gaps = SurveyGapFinder().find(
        [],
        expected_themes=["retrieval augmented generation", "citation graphs"],
    )
    assert len(gaps) == 2
    assert all(gap.missing for gap in gaps)
    assert all(gap.coverage_score == 0.0 for gap in gaps)
    assert gaps[0].theme == "retrieval augmented generation"
    assert gaps[1].matching_titles == ()


def test_covered_theme_is_not_missing() -> None:
    papers = [
        {
            "title": "Retrieval Augmented Generation for QA",
            "abstract": "We study retrieval augmented generation over scientific corpora.",
            "year": 2024,
        },
        {
            "title": "Sparse Attention Transformers",
            "abstract": "Efficient attention reduces quadratic cost.",
            "year": 2023,
        },
    ]
    gaps = SurveyGapFinder(coverage_threshold=0.1).find(
        papers,
        expected_themes=[
            "retrieval augmented generation",
            "knowledge graph reasoning",
        ],
    )
    by_theme = {gap.theme: gap for gap in gaps}
    covered = by_theme["retrieval augmented generation"]
    assert covered.missing is False
    assert covered.coverage_score >= 0.1
    assert covered.matching_titles
    missing = by_theme["knowledge graph reasoning"]
    assert missing.missing is True
    assert missing.coverage_score < 0.1


def test_blank_themes_skipped_and_docstring_mentions_gap() -> None:
    gaps = SurveyGapFinder().find(
        [{"title": "Graph Retrieval", "abstract": "graph retrieval methods"}],
        expected_themes=["", "  ", "graph retrieval"],
    )
    assert len(gaps) == 1
    assert gaps[0].theme == "graph retrieval"
    doc = SurveyGapFinder.__doc__ or ""
    assert "GPT-5.5" in doc
    assert "Claude Sonnet 4.6" in doc
    assert "Gemini 3.x" in doc
    assert "Kimi K2" in doc
    assert "Elicit" in doc or "ResearchRabbit" in doc
    assert "RelatedWorksComposer" in doc or "related" in doc.lower()
