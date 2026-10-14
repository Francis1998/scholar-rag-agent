"""Tests for DuplicatePaperClusterer."""

from copy import deepcopy

import pytest

from retrieval.duplicate_paper_cluster import (
    DuplicatePaperClusterer,
    PaperCluster,
    PaperRef,
)
from retrieval.models import Chunk, Document, SearchResult

PaperInput = PaperRef | Document | Chunk | SearchResult


def _as_input(paper: PaperRef, input_type: type[PaperInput]) -> PaperInput:
    if input_type is PaperRef:
        return paper
    if input_type is Document:
        return Document(
            document_id=paper.document_id,
            title=paper.title,
            text="Synthetic document body.",
            source=paper.source,
            metadata={"doi": paper.doi, "origin": "fixture"},
        )
    chunk = Chunk(
        chunk_id=f"chunk:{paper.document_id}",
        document_id=paper.document_id,
        title=paper.title,
        text="Synthetic chunk body.",
        source=paper.source,
        metadata={"doi": paper.doi, "origin": "fixture"},
    )
    if input_type is Chunk:
        return chunk
    if input_type is SearchResult:
        return SearchResult(chunk=chunk, score=0.9, retriever="fixture", path=["synthetic"])
    raise AssertionError(f"Unsupported input type: {input_type}")


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


@pytest.mark.parametrize("input_type", [PaperRef, Document, Chunk, SearchResult])
@pytest.mark.parametrize(
    ("left_id", "right_id"),
    [
        ("Study-A", "study-a"),
        ("Study-A", " Study-A"),
        ("Study-A", "Study-A "),
        (" Study-A", "Study-A "),
        ("Study-A", "\tStudy-A\n"),
    ],
    ids=["case", "leading-space", "trailing-space", "different-padding", "tab-newline"],
)
def test_document_id_matches_are_literal(
    input_type: type[PaperInput], left_id: str, right_id: str
) -> None:
    refs = [
        PaperRef(document_id=left_id, title="Completely different title one", source="fixture"),
        PaperRef(document_id=right_id, title="Unrelated quantum results", source="fixture"),
    ]
    papers = [_as_input(ref, input_type) for ref in refs]
    before = deepcopy(papers)
    clusterer = DuplicatePaperClusterer(title_threshold=1.0)

    assert clusterer.cluster(papers) == [
        PaperCluster(cluster_id="cluster-0000", papers=(refs[0],), match_reasons=("singleton",)),
        PaperCluster(cluster_id="cluster-0001", papers=(refs[1],), match_reasons=("singleton",)),
    ]
    assert clusterer.duplicate_clusters(papers) == []
    assert papers == before


@pytest.mark.parametrize("input_type", [PaperRef, Document, Chunk, SearchResult])
@pytest.mark.parametrize("document_id", ["Study-A", " Study-A ", "\tStudy-A\n"])
def test_repeated_exact_document_ids_merge(input_type: type[PaperInput], document_id: str) -> None:
    refs = [
        PaperRef(document_id=document_id, title="Completely different title one", source="first"),
        PaperRef(document_id=document_id, title="Unrelated quantum results", source="second"),
    ]
    papers = [_as_input(ref, input_type) for ref in refs]
    before = deepcopy(papers)
    clusterer = DuplicatePaperClusterer(title_threshold=1.0)
    expected = [
        PaperCluster(cluster_id="cluster-0000", papers=tuple(refs), match_reasons=("document_id",))
    ]

    assert clusterer.cluster(papers) == expected
    assert clusterer.duplicate_clusters(papers) == expected
    assert papers == before


@pytest.mark.parametrize("input_type", [PaperRef, Document, Chunk, SearchResult])
@pytest.mark.parametrize("document_id", ["", " ", "\t\n"])
def test_blank_document_ids_do_not_match_or_supply_data(
    input_type: type[PaperInput], document_id: str
) -> None:
    refs = [
        PaperRef(document_id=document_id, title="Completely different title one"),
        PaperRef(document_id=document_id, title="Unrelated quantum results"),
    ]
    papers = [_as_input(ref, input_type) for ref in refs]
    clusterer = DuplicatePaperClusterer(title_threshold=1.0)

    assert clusterer.cluster(papers) == [
        PaperCluster(cluster_id="cluster-0000", papers=(refs[0],), match_reasons=("singleton",)),
        PaperCluster(cluster_id="cluster-0001", papers=(refs[1],), match_reasons=("singleton",)),
    ]
    assert clusterer.duplicate_clusters(papers) == []
    assert (
        clusterer.cluster([_as_input(PaperRef(document_id=document_id, title=" \t\n"), input_type)])
        == []
    )


@pytest.mark.parametrize("input_type", [PaperRef, Document, Chunk, SearchResult])
def test_blank_document_ids_still_allow_doi_and_title_matches(
    input_type: type[PaperInput],
) -> None:
    refs = [
        PaperRef(document_id="", title="", doi="10.2/abc"),
        PaperRef(document_id=" \t", title="Unrelated quantum results", doi="doi:10.2/ABC"),
        PaperRef(document_id="", title="Shared title"),
        PaperRef(document_id=" \t", title="  SHARED   TITLE  "),
    ]
    papers = [_as_input(ref, input_type) for ref in refs]

    assert DuplicatePaperClusterer(title_threshold=1.0).cluster(papers) == [
        PaperCluster(cluster_id="cluster-0000", papers=tuple(refs[:2]), match_reasons=("doi",)),
        PaperCluster(
            cluster_id="cluster-0002", papers=tuple(refs[2:]), match_reasons=("title:1.000",)
        ),
    ]


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


def test_transitive_heuristics_preserve_literal_ids_order_and_inputs() -> None:
    refs = [
        PaperRef(document_id="solo", title="Lunar atmosphere"),
        PaperRef(document_id="Study-A", title="Completely different title one", doi="10.2/link"),
        PaperRef(document_id="Study-A", title="Unrelated quantum results"),
        PaperRef(
            document_id="another", title="Shared bridge title", doi="https://doi.org/10.2/LINK"
        ),
        PaperRef(document_id="fourth", title="  SHARED   BRIDGE TITLE  "),
        PaperRef(document_id="study-a", title="Independent marine ecology"),
        PaperRef(document_id=" Study-A ", title="Distinct thermodynamic analysis"),
    ]
    input_types = [PaperRef, PaperRef, Document, Chunk, SearchResult, Document, SearchResult]
    papers = [_as_input(ref, input_type) for ref, input_type in zip(refs, input_types, strict=True)]
    before = deepcopy(papers)
    clusterer = DuplicatePaperClusterer(title_threshold=1.0)
    expected = [
        PaperCluster(
            cluster_id="cluster-0001",
            papers=tuple(refs[1:5]),
            match_reasons=("document_id", "doi", "title:1.000"),
        ),
        PaperCluster(cluster_id="cluster-0000", papers=(refs[0],), match_reasons=("singleton",)),
        PaperCluster(cluster_id="cluster-0005", papers=(refs[5],), match_reasons=("singleton",)),
        PaperCluster(cluster_id="cluster-0006", papers=(refs[6],), match_reasons=("singleton",)),
    ]

    assert clusterer.cluster(papers) == expected
    assert clusterer.cluster(papers) == expected
    assert clusterer.duplicate_clusters(papers) == expected[:1]
    assert papers == before
