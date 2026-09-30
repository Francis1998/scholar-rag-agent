"""Tests for BibTeXExporter."""

import re
from typing import Literal

import pytest

from retrieval.bibtex_export import BibTeXExporter
from retrieval.models import Chunk, Document, SearchResult

CollectionKind = Literal["documents", "chunks", "results"]


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


def _export_collection(kind: CollectionKind, documents: list[Document]) -> str:
    exporter = BibTeXExporter()
    if kind == "documents":
        return exporter.export_documents(documents)
    chunks = [
        Chunk(
            chunk_id=f"chunk-{index}",
            document_id=document.document_id,
            title=document.title,
            text=document.text,
            source=document.source,
            metadata=document.metadata,
        )
        for index, document in enumerate(documents)
    ]
    if kind == "chunks":
        return exporter.export_chunks(chunks)
    return exporter.export_results(
        [SearchResult(chunk=chunk, score=1.0, retriever="fixture") for chunk in chunks]
    )


def _entry_keys(bibtex: str) -> list[str]:
    return re.findall(r"^@\w+\{([^,\n]+),", bibtex, re.MULTILINE)


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


def test_export_document_requires_no_provider_metadata() -> None:
    assert BibTeXExporter().export_document(_document()) == (
        "@misc{doc_1,\n  title = {Sample Title}\n}"
    )


@pytest.mark.parametrize("kind", ["documents", "chunks", "results"])
def test_collections_preserve_exact_document_identity(kind: CollectionKind) -> None:
    documents = [
        _document(document_id=identity, title=f"Title {index}")
        for index, identity in enumerate(("Paper", "paper", " paper", "paper "))
    ]
    before = [document.model_dump() for document in documents]

    bibliography = _export_collection(kind, documents)

    assert len(_entry_keys(bibliography)) == len(documents)
    assert len(set(_entry_keys(bibliography))) == len(documents)
    for index in range(len(documents)):
        assert f"title = {{Title {index}}}" in bibliography
    assert bibliography == _export_collection(kind, documents)
    assert [document.model_dump() for document in documents] == before


@pytest.mark.parametrize("kind", ["documents", "chunks", "results"])
def test_collections_disambiguate_normalized_keys(kind: CollectionKind) -> None:
    documents = [_document(document_id=identity) for identity in ("paper-a", "paper_a", "paper a")]

    assert _entry_keys(_export_collection(kind, documents)) == [
        "paper_a",
        "paper_a_2",
        "paper_a_3",
    ]


@pytest.mark.parametrize("kind", ["documents", "chunks", "results"])
@pytest.mark.parametrize("reverse", [False, True])
def test_collections_reserve_natural_suffix_keys(kind: CollectionKind, reverse: bool) -> None:
    identities = ["paper-a", "paper_a", "paper_a_2", "paper a", "paper_a_3"]
    if reverse:
        identities.reverse()
    documents = [_document(document_id=identity, title=identity) for identity in identities]

    bibliography = _export_collection(kind, documents)
    keys = _entry_keys(bibliography)

    assert len(keys) == len(set(keys)) == len(documents)
    assert keys[identities.index("paper_a_2")] == "paper_a_2"
    assert keys[identities.index("paper_a_3")] == "paper_a_3"
    assert set(keys) == {f"paper_a{suffix}" for suffix in ("", "_2", "_3", "_4", "_5")}


@pytest.mark.parametrize("kind", ["documents", "chunks", "results"])
def test_collections_preserve_documents_with_the_same_doi(kind: CollectionKind) -> None:
    documents = [
        _document(document_id=f"paper-{index}", title=f"Title {index}", doi="10.123/shared")
        for index in range(3)
    ]

    bibliography = _export_collection(kind, documents)

    assert _entry_keys(bibliography) == [
        "10_123_shared",
        "10_123_shared_2",
        "10_123_shared_3",
    ]
    assert bibliography.count("doi = {10.123/shared}") == 3


@pytest.mark.parametrize("kind", ["documents", "chunks", "results"])
def test_collections_keep_first_exact_duplicate_on_equal_scores(kind: CollectionKind) -> None:
    documents = [
        _document(document_id="same", title="First"),
        _document(document_id="same", title="Ignored duplicate"),
        _document(document_id="SAME", title="Distinct identity"),
    ]

    bibliography = _export_collection(kind, documents)

    assert _entry_keys(bibliography) == ["same", "same_2"]
    assert "title = {First}" in bibliography
    assert "title = {Distinct identity}" in bibliography
    assert "Ignored duplicate" not in bibliography


def test_results_choose_highest_score_only_within_exact_identity() -> None:
    results = [
        SearchResult(
            chunk=_chunk(chunk_id="c1", document_id="Paper", title="Low"),
            score=1.0,
            retriever="fixture",
        ),
        SearchResult(
            chunk=_chunk(chunk_id="c2", document_id="paper", title="Distinct"),
            score=2.0,
            retriever="fixture",
        ),
        SearchResult(
            chunk=_chunk(chunk_id="c3", document_id="Paper", title="High"),
            score=3.0,
            retriever="fixture",
        ),
    ]
    before = [result.model_dump() for result in results]

    bibliography = BibTeXExporter().export_results(results)

    assert _entry_keys(bibliography) == ["paper", "paper_2"]
    assert "title = {Low}" not in bibliography
    assert bibliography.index("High") < bibliography.index("Distinct")
    assert [result.model_dump() for result in results] == before
