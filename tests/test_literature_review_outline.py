"""Tests for LiteratureReviewOutliner."""

import pytest

from retrieval.literature_review_outline import LiteratureReviewOutliner
from retrieval.models import Chunk, Document, SearchResult


def _chunk(
    chunk_id: str,
    document_id: str,
    title: str,
    *,
    section: str = "",
) -> Chunk:
    metadata: dict[str, str] = {}
    if section:
        metadata["section"] = section
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        title=title,
        text="body",
        source="test",
        metadata=metadata,
    )


def _result(chunk: Chunk, score: float) -> SearchResult:
    return SearchResult(chunk=chunk, score=score, retriever="hybrid")


def test_rejects_non_positive_top_k() -> None:
    outliner = LiteratureReviewOutliner()
    with pytest.raises(ValueError, match="top_k"):
        outliner.outline([], top_k=0)


def test_empty_sources_produce_empty_sections() -> None:
    outline = LiteratureReviewOutliner().outline([], topic="RAG")
    assert outline.topic == "RAG"
    assert len(outline.sections) == 5
    assert all(section.items == () for section in outline.sections)


def test_section_metadata_routes_papers() -> None:
    outliner = LiteratureReviewOutliner()
    sources = [
        _result(_chunk("c1", "d1", "Intro Paper", section="introduction"), 0.9),
        _result(_chunk("c2", "d2", "Methods Paper", section="methods"), 0.8),
        _result(_chunk("c3", "d3", "Results Paper", section="results"), 0.7),
    ]
    outline = outliner.outline(sources, topic="GraphRAG")
    by_title = {
        section.title: [item.document_id for item in section.items] for section in outline.sections
    }
    assert by_title["Background & Motivation"] == ["d1"]
    assert by_title["Methods & Approaches"] == ["d2"]
    assert by_title["Key Findings"] == ["d3"]


def test_unassigned_round_robin_by_rank() -> None:
    outliner = LiteratureReviewOutliner()
    sources = [
        _result(_chunk("c1", "a", "A"), 1.0),
        _result(_chunk("c2", "b", "B"), 0.9),
        _result(_chunk("c3", "c", "C"), 0.8),
        _result(_chunk("c4", "d", "D"), 0.7),
        _result(_chunk("c5", "e", "E"), 0.6),
    ]
    outline = outliner.outline(sources)
    ids_per_section = [[item.document_id for item in section.items] for section in outline.sections]
    assert ids_per_section[0] == ["a"]
    assert ids_per_section[1] == ["b"]
    assert ids_per_section[2] == ["c"]
    assert ids_per_section[3] == ["d"]
    assert ids_per_section[4] == ["e"]


def test_dedupes_by_highest_score() -> None:
    outliner = LiteratureReviewOutliner()
    sources = [
        _result(_chunk("low", "doc", "Low", section="methods"), 0.2),
        _result(_chunk("high", "doc", "High", section="methods"), 0.95),
    ]
    outline = outliner.outline(sources)
    methods = next(section for section in outline.sections if section.title.startswith("Methods"))
    assert len(methods.items) == 1
    assert methods.items[0].title == "High"
    assert methods.items[0].score == pytest.approx(0.95)


def test_top_k_limits_unique_documents() -> None:
    outliner = LiteratureReviewOutliner()
    sources = [
        _result(_chunk("c1", "a", "A"), 1.0),
        _result(_chunk("c2", "b", "B"), 0.9),
        _result(_chunk("c3", "c", "C"), 0.8),
    ]
    outline = outliner.outline(sources, top_k=2)
    all_ids = [item.document_id for section in outline.sections for item in section.items]
    assert all_ids == ["a", "b"]


def test_accepts_documents_and_markdown_mentions_gap() -> None:
    outliner = LiteratureReviewOutliner()
    docs = [
        Document(
            document_id="p1",
            title="Paper One",
            text="x",
            source="pdf",
            metadata={"section_type": "conclusion"},
        )
    ]
    outline = outliner.outline(docs, topic="Survey")
    markdown = outline.to_markdown()
    assert "PaperQA-style review synthesis gap" in markdown
    assert "GPT-5.5" in markdown
    assert "Claude Sonnet 4.6" in markdown
    assert "Gemini 3.x" in markdown
    assert "Kimi K2" in markdown
    synthesis = next(section for section in outline.sections if "Synthesis" in section.title)
    assert synthesis.items[0].document_id == "p1"


def test_docstring_mentions_frontier_models() -> None:
    doc = LiteratureReviewOutliner.__doc__ or ""
    assert "GPT-5.5" in doc
    assert "Claude Sonnet 4.6" in doc
    assert "Gemini 3.x" in doc
    assert "Kimi K2" in doc
    assert "PaperQA" in doc
