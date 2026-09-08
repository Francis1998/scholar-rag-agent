"""Tests for RelatedWorksComposer."""

import pytest

from retrieval.related_works import RelatedWorksComposer, RelatedWorksTheme


def test_rejects_invalid_overlap_threshold() -> None:
    with pytest.raises(ValueError, match="overlap_threshold"):
        RelatedWorksComposer(overlap_threshold=2.0)


def test_rejects_invalid_max_themes() -> None:
    with pytest.raises(ValueError, match="max_themes"):
        RelatedWorksComposer(max_themes=0)


def test_empty_papers_yield_empty_sections() -> None:
    outline = RelatedWorksComposer().compose([], title="Related Works")
    assert outline.title == "Related Works"
    assert outline.sections == ()


def test_clusters_overlapping_keywords_into_themes() -> None:
    papers = [
        {
            "title": "Graph Retrieval Augmented Generation",
            "abstract": "We survey graph retrieval methods for RAG pipelines.",
            "year": 2024,
        },
        {
            "title": "Knowledge Graph Retrieval for QA",
            "abstract": "Graph retrieval improves question answering over corpora.",
            "year": 2023,
        },
        {
            "title": "Transformer Attention Efficiency",
            "abstract": "Sparse attention reduces quadratic compute in transformers.",
            "year": 2022,
        },
        {
            "title": "Efficient Transformers Survey",
            "abstract": "Attention efficiency techniques for long-context transformers.",
            "year": 2021,
        },
    ]
    outline = RelatedWorksComposer(overlap_threshold=0.12, max_themes=4).compose(papers)
    assert len(outline.sections) >= 2
    sizes = sorted((len(section.papers) for section in outline.sections), reverse=True)
    assert sizes[0] >= 2


def test_accepts_theme_objects_and_sorts_by_year() -> None:
    papers = [
        RelatedWorksTheme(
            title="Older RAG",
            year=2020,
            abstract="retrieval augmented generation baselines",
            keywords=("retrieval", "augmented", "generation", "baselines"),
        ),
        RelatedWorksTheme(
            title="Newer RAG",
            year=2024,
            abstract="retrieval augmented generation advances",
            keywords=("retrieval", "augmented", "generation", "advances"),
        ),
    ]
    outline = RelatedWorksComposer(overlap_threshold=0.2).compose(papers)
    assert len(outline.sections) == 1
    titles = [paper.title for paper in outline.sections[0].papers]
    assert titles[0] == "Newer RAG"


def test_markdown_mentions_gap_and_models() -> None:
    outline = RelatedWorksComposer().compose(
        [
            {
                "title": "Citation Graph Expansion",
                "abstract": "Expand seeds along citation edges.",
                "year": 2023,
            }
        ]
    )
    markdown = outline.to_markdown()
    assert "Elicit/PaperQA related-work generation gap" in markdown
    assert "GPT-5.5" in markdown
    assert "Claude Sonnet 4.6" in markdown
    assert "Gemini 3.x" in markdown
    assert "Kimi K2" in markdown


def test_docstring_mentions_frontier_models_and_gap() -> None:
    doc = RelatedWorksComposer.__doc__ or ""
    assert "GPT-5.5" in doc
    assert "Claude Sonnet 4.6" in doc
    assert "Gemini 3.x" in doc
    assert "Kimi K2" in doc
    assert "Elicit" in doc or "PaperQA" in doc
