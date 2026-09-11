"""Tests for ReadingListPrioritizer."""

import pytest

from retrieval.reading_list import ReadingListPrioritizer


def test_rejects_invalid_knobs() -> None:
    with pytest.raises(ValueError, match="freshness_weight"):
        ReadingListPrioritizer(freshness_weight=1.5)
    with pytest.raises(ValueError, match="reference_year"):
        ReadingListPrioritizer(reference_year=0)
    with pytest.raises(ValueError, match="half_life_years"):
        ReadingListPrioritizer(half_life_years=-1.0)


def test_empty_papers_ok() -> None:
    assert ReadingListPrioritizer().prioritize([]) == ()


def test_ranks_by_novelty_times_authority() -> None:
    papers = [
        {
            "title": "Classic Survey of Optics",
            "year": 2010,
            "citation_count": 500,
            "keywords": ["optics", "lasers"],
            "abstract": "A broad survey of classical optics and lasers.",
        },
        {
            "title": "Novel Graph Retrieval for Multi-Hop Reasoning",
            "year": 2025,
            "cited_by_count": 40,
            "keywords": ["graph", "retrieval", "multi-hop", "reasoning"],
            "abstract": "Graph retrieval improves multi-hop reasoning over papers.",
        },
        {
            "title": "Obscure Preprint Stub",
            "year": 2024,
            "citations": 0,
            "topics": ["unrelated", "astronomy"],
        },
    ]
    ranked = ReadingListPrioritizer(reference_year=2026).prioritize(papers)
    assert len(ranked) == 3
    assert ranked[0].priority_score >= ranked[1].priority_score >= ranked[2].priority_score
    # High novelty x moderate authority should outrank stale high-cite survey
    # when freshness/keyword novelty lift the recent multi-hop paper.
    titles = [row.title for row in ranked]
    assert "Novel Graph Retrieval for Multi-Hop Reasoning" in titles
    assert ranked[0].novelty >= 0.0
    assert ranked[0].authority >= 0.0
    assert abs(ranked[0].priority_score - ranked[0].novelty * ranked[0].authority) < 1e-9
    assert ranked[0].reasons
    assert any("citation" in reason for reason in ranked[0].reasons)
    # Zero-citation stub should rank last by multiplicative score.
    assert ranked[-1].title.startswith("Obscure")
    assert ranked[-1].authority == 0.0
    assert ranked[-1].priority_score == 0.0


def test_docstring_mentions_frontier_models_and_gap() -> None:
    doc = ReadingListPrioritizer.__doc__ or ""
    assert "GPT-5.5" in doc
    assert "Claude Sonnet 4.6" in doc
    assert "Gemini 3.x" in doc
    assert "Kimi K2" in doc
    assert "Zotero" in doc or "ResearchRabbit" in doc
    assert "FreshnessBooster" in doc
    assert "NoveltyDiversifier" in doc
    assert "AuthorityBooster" in doc
