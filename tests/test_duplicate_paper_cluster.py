"""Tests for DuplicatePaperClusterer."""

import pytest

from retrieval.duplicate_paper_cluster import (
    DuplicatePaperClusterer,
    PaperRef,
)
from retrieval.models import Chunk, Document, SearchResult


def test_rejects_invalid_title_threshold() -> None:
    with pytest.raises(ValueError, match="title_threshold"):
        DuplicatePaperClusterer(title_threshold=1.5)


def test_empty_input_returns_empty() -> None:
    assert DuplicatePaperClusterer().cluster([]) == []


def test_clusters_by_exact_doi() -> None:
    papers = [
        PaperRef(document_id="a", title="Alpha Study", doi="10.1000/xyz"),
        PaperRef(document_id="b", title="Different Title", doi="https://doi.org/10.1000/XYZ"),
        PaperRef(document_id="c", title="Unrelated", doi="10.1000/other"),
    ]
    clusters = DuplicatePaperClusterer().duplicate_clusters(papers)
    assert len(clusters) == 1
    assert clusters[0].size == 2
    assert "doi" in clusters[0].match_reasons
    ids = {paper.document_id for paper in clusters[0].papers}
    assert ids == {"a", "b"}


def test_clusters_by_document_id() -> None:
    papers = [
        PaperRef(document_id="same-doc", title="Version A"),
        PaperRef(document_id="same-doc", title="Version B"),
        PaperRef(document_id="other", title="Other"),
    ]
    clusters = DuplicatePaperClusterer().duplicate_clusters(papers)
    assert len(clusters) == 1
    assert clusters[0].size == 2
    assert "document_id" in clusters[0].match_reasons


def test_clusters_by_fuzzy_title() -> None:
    papers = [
        PaperRef(document_id="1", title="Graph Retrieval Augmented Generation Survey"),
        PaperRef(document_id="2", title="Graph Retrieval-Augmented Generation Survey"),
        PaperRef(document_id="3", title="Unrelated Quantum Optics Paper"),
    ]
    clusters = DuplicatePaperClusterer(title_threshold=0.85).duplicate_clusters(papers)
    assert len(clusters) == 1
    assert clusters[0].size == 2
    assert any(reason.startswith("title:") for reason in clusters[0].match_reasons)


def test_singletons_included_in_full_partition() -> None:
    papers = [
        PaperRef(document_id="solo", title="Only Paper"),
        PaperRef(document_id="a", title="Shared DOI", doi="10.1/x"),
        PaperRef(document_id="b", title="Also Shared", doi="10.1/x"),
    ]
    clusters = DuplicatePaperClusterer().cluster(papers)
    assert len(clusters) == 2
    sizes = sorted(cluster.size for cluster in clusters)
    assert sizes == [1, 2]


def test_accepts_documents_chunks_and_search_results() -> None:
    doc = Document(
        document_id="d1",
        title="Paper One",
        text="body",
        source="pdf",
        metadata={"doi": "10.2/abc"},
    )
    chunk = Chunk(
        chunk_id="c1",
        document_id="d2",
        title="Paper Two",
        text="body",
        source="arxiv",
        metadata={"doi": "10.2/ABC"},
    )
    result = SearchResult(chunk=chunk, score=0.9, retriever="hybrid")
    clusters = DuplicatePaperClusterer().duplicate_clusters([doc, result])
    assert len(clusters) == 1
    assert clusters[0].size == 2


def test_docstring_mentions_gap_and_frontier_models() -> None:
    doc = DuplicatePaperClusterer.__doc__ or ""
    assert "GPT-5.5" in doc
    assert "Claude Sonnet 4.6" in doc
    assert "Gemini 3.x" in doc
    assert "Kimi K2" in doc
    assert "PaperQA" in doc or "LocalGPT" in doc
    assert "NearDuplicate" in doc or "paper-level" in doc.lower()
