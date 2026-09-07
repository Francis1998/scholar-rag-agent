"""Tests for BibTeXExporter."""

from retrieval.bibtex_export import BibTeXExporter
from retrieval.models import Chunk, Document, SearchResult


def _document(**kwargs: str) -> Document:
    metadata = {key: value for key, value in kwargs.items() if key not in {"document_id", "title"}}
    return Document(
        document_id=kwargs.get("document_id", "doc-1"),
        title=kwargs.get("title", "Sample Title"),
        text="body",
        source="test",
        metadata=metadata,
    )


def _chunk(**kwargs: str) -> Chunk:
    metadata = {
        key: value
        for key, value in kwargs.items()
        if key not in {"chunk_id", "document_id", "title"}
    }
    return Chunk(
        chunk_id=kwargs.get("chunk_id", "c1"),
        document_id=kwargs.get("document_id", "doc-1"),
        title=kwargs.get("title", "Sample Title"),
        text="body",
        source="test",
        metadata=metadata,
    )


def test_export_document_article_with_doi() -> None:
    exporter = BibTeXExporter()
    entry = exporter.export_document(
        _document(
            document_id="paper-a",
            title="GraphRAG for Science",
            authors="Ada Lovelace; Alan Turing",
            year="2024",
            journal="Nature Methods",
            doi="10.1000/xyz",
        )
    )
    assert entry.startswith("@article{10_1000_xyz,")
    assert "title = {GraphRAG for Science}" in entry
    assert "author = {Ada Lovelace and Alan Turing}" in entry
    assert "year = {2024}" in entry
    assert "journal = {Nature Methods}" in entry
    assert "doi = {10.1000/xyz}" in entry


def test_export_chunk_escapes_braces() -> None:
    exporter = BibTeXExporter()
    entry = exporter.export_chunk(_chunk(title="A {braced} title", document_id="doc-brace"))
    assert "title = {A \\{braced\\} title}" in entry


def test_export_results_dedupes_by_highest_score() -> None:
    exporter = BibTeXExporter()
    low = SearchResult(
        chunk=_chunk(chunk_id="a", document_id="doc-a", title="Low"),
        score=0.2,
        retriever="hybrid",
    )
    high = SearchResult(
        chunk=_chunk(chunk_id="b", document_id="doc-a", title="High", year="2023"),
        score=0.9,
        retriever="hybrid",
    )
    other = SearchResult(
        chunk=_chunk(chunk_id="c", document_id="doc-b", title="Other", year="2022"),
        score=0.5,
        retriever="hybrid",
    )
    bib = exporter.export_results([low, other, high])
    assert bib.count("@") == 2
    assert "title = {High}" in bib
    assert "title = {Low}" not in bib
    assert bib.index("High") < bib.index("Other")


def test_export_documents_skips_duplicate_ids() -> None:
    exporter = BibTeXExporter()
    first = _document(document_id="same", title="First")
    second = _document(document_id="same", title="Second")
    bib = exporter.export_documents([first, second])
    assert bib.count("@") == 1
    assert "First" in bib
    assert "Second" not in bib


def test_inproceedings_uses_booktitle() -> None:
    exporter = BibTeXExporter()
    entry = exporter.export_document(
        _document(
            document_id="conf-1",
            title="Poster",
            entry_type="inproceedings",
            venue="NeurIPS",
            year="2021",
        )
    )
    assert entry.startswith("@inproceedings{")
    assert "booktitle = {NeurIPS}" in entry


def test_year_extracted_from_published_at() -> None:
    exporter = BibTeXExporter()
    entry = exporter.export_chunk(_chunk(document_id="d1", published_at="2020-05-01"))
    assert "year = {2020}" in entry


def test_empty_collections_return_empty_string() -> None:
    exporter = BibTeXExporter()
    assert exporter.export_documents([]) == ""
    assert exporter.export_chunks([]) == ""
    assert exporter.export_results([]) == ""


def test_docstring_mentions_frontier_models() -> None:
    doc = BibTeXExporter.__doc__ or ""
    assert "GPT-5.5" in doc
    assert "Claude Sonnet 4.6" in doc
    assert "Gemini 3.x" in doc
    assert "Kimi K2" in doc
