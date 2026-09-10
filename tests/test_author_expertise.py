"""Tests for AuthorExpertiseRanker."""

import pytest

from retrieval.author_expertise import AuthorExpertiseRanker


def test_rejects_invalid_weights() -> None:
    with pytest.raises(ValueError, match="pub_count_weight"):
        AuthorExpertiseRanker(pub_count_weight=1.5)
    with pytest.raises(ValueError, match="at least one"):
        AuthorExpertiseRanker(pub_count_weight=0.0, topic_weight=0.0)


def test_empty_papers_ok() -> None:
    assert AuthorExpertiseRanker().rank([]) == ()


def test_ranks_by_pub_count_and_topic_overlap() -> None:
    papers = [
        {
            "title": "Graph Neural Networks for Molecules",
            "authors": ["Alice Expert", "Bob Newbie"],
            "topics": ["graph neural networks", "chemistry"],
            "abstract": "We apply graph neural networks to molecules.",
        },
        {
            "title": "Survey of Image Classification",
            "authors": "Carol Other",
            "topics": ["computer vision"],
            "abstract": "A survey of image classifiers.",
        },
    ]
    ranked = AuthorExpertiseRanker().rank(
        papers,
        topic_query="graph neural networks",
        author_pub_counts={"alice expert": 40, "bob newbie": 2, "carol other": 5},
    )
    assert len(ranked) == 2
    assert ranked[0].title.startswith("Graph Neural")
    assert ranked[0].author_pub_count == 40
    assert ranked[0].topic_overlap > ranked[1].topic_overlap
    assert ranked[0].expertise_score >= ranked[1].expertise_score


def test_docstring_mentions_frontier_models_and_gap() -> None:
    doc = AuthorExpertiseRanker.__doc__ or ""
    assert "GPT-5.5" in doc
    assert "Claude Sonnet 4.6" in doc
    assert "Gemini 3.x" in doc
    assert "Kimi K2" in doc
    assert "Semantic Scholar" in doc or "OpenAlex" in doc
