"""Tests for CitationGraphIndex and CitationGraphExpander."""

import math

import pytest

from retrieval.citation_graph import CitationGraphExpander, CitationGraphIndex
from retrieval.models import Chunk, SearchResult


def _chunk(
    chunk_id: str,
    document_id: str,
    *,
    doi: str = "",
    references: str = "",
    cited_by: str = "",
    text: str = "body",
) -> Chunk:
    metadata: dict[str, str] = {}
    if doi:
        metadata["doi"] = doi
    if references:
        metadata["references"] = references
    if cited_by:
        metadata["cited_by"] = cited_by
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        title=chunk_id,
        text=text,
        source="test",
        metadata=metadata,
    )


def _result(chunk: Chunk, score: float = 1.0) -> SearchResult:
    return SearchResult(chunk=chunk, score=score, retriever="hybrid", path=["hybrid"])


def test_rejects_non_positive_hop_decay() -> None:
    index = CitationGraphIndex()
    with pytest.raises(ValueError, match="hop_decay"):
        CitationGraphExpander(index, hop_decay=0.0)


def test_rejects_non_finite_hop_decay() -> None:
    index = CitationGraphIndex()
    with pytest.raises(ValueError, match="hop_decay"):
        CitationGraphExpander(index, hop_decay=float("nan"))


def test_rejects_invalid_direction() -> None:
    index = CitationGraphIndex()
    with pytest.raises(ValueError, match="direction"):
        CitationGraphExpander(index, direction="sideways")


def test_empty_results() -> None:
    index = CitationGraphIndex()
    expander = CitationGraphExpander(index)
    assert expander.expand([]) == []


def test_top_k_zero_returns_empty() -> None:
    seed = _chunk("a", "doc-a", doi="10.1/a", references="10.1/b")
    index = CitationGraphIndex()
    index.index_chunks([seed, _chunk("b", "doc-b", doi="10.1/b")])
    expander = CitationGraphExpander(index)
    assert expander.expand([_result(seed)], top_k=0) == []


def test_expands_cited_reference_one_hop() -> None:
    seed = _chunk("seed", "doc-seed", doi="10.1/seed", references="10.1/ref")
    ref = _chunk("ref", "doc-ref", doi="10.1/ref", text="cited methods")
    index = CitationGraphIndex()
    index.index_chunks([seed, ref])
    expander = CitationGraphExpander(index, hop_decay=0.5, direction="cites")
    kept = expander.expand([_result(seed, 1.0)], max_hops=1)
    ids = [item.chunk.chunk_id for item in kept]
    assert ids == ["seed", "ref"]
    assert kept[1].score == pytest.approx(0.5)
    assert kept[1].retriever == "citation_graph"
    assert kept[1].path == ["10.1/seed", "10.1/ref"]


def test_expands_cited_by_incoming_edge() -> None:
    classic = _chunk("classic", "doc-classic", doi="10.1/classic")
    newer = _chunk(
        "newer",
        "doc-newer",
        doi="10.1/newer",
        references="10.1/classic",
    )
    index = CitationGraphIndex()
    index.index_chunks([classic, newer])
    expander = CitationGraphExpander(index, hop_decay=0.5, direction="cited_by")
    kept = expander.expand([_result(classic, 1.0)], max_hops=1)
    ids = [item.chunk.chunk_id for item in kept]
    assert "newer" in ids
    newer_hit = next(item for item in kept if item.chunk.chunk_id == "newer")
    assert newer_hit.score == pytest.approx(0.5)


def test_normalizes_doi_prefixes() -> None:
    seed = _chunk(
        "seed",
        "doc-seed",
        doi="DOI:10.1/SEED",
        references="https://doi.org/10.1/ref",
    )
    ref = _chunk("ref", "doc-ref", doi="10.1/ref")
    index = CitationGraphIndex()
    index.index_chunks([seed, ref])
    expander = CitationGraphExpander(index, direction="cites")
    kept = expander.expand([_result(seed)], max_hops=1)
    assert any(item.chunk.chunk_id == "ref" for item in kept)


def test_two_hops_apply_compound_decay() -> None:
    a = _chunk("a", "doc-a", doi="10.1/a", references="10.1/b")
    b = _chunk("b", "doc-b", doi="10.1/b", references="10.1/c")
    c = _chunk("c", "doc-c", doi="10.1/c")
    index = CitationGraphIndex()
    index.index_chunks([a, b, c])
    expander = CitationGraphExpander(index, hop_decay=0.5, direction="cites")
    kept = expander.expand([_result(a, 1.0)], max_hops=2)
    by_id = {item.chunk.chunk_id: item.score for item in kept}
    assert by_id["a"] == pytest.approx(1.0)
    assert by_id["b"] == pytest.approx(0.5)
    assert by_id["c"] == pytest.approx(0.25)


def test_exclude_seeds_returns_only_neighbours() -> None:
    seed = _chunk("seed", "doc-seed", doi="10.1/seed", references="10.1/ref")
    ref = _chunk("ref", "doc-ref", doi="10.1/ref")
    index = CitationGraphIndex()
    index.index_chunks([seed, ref])
    expander = CitationGraphExpander(index, direction="cites")
    kept = expander.expand([_result(seed)], include_seeds=False)
    assert [item.chunk.chunk_id for item in kept] == ["ref"]


def test_does_not_mutate_inputs() -> None:
    seed = _chunk("seed", "doc-seed", doi="10.1/seed", references="10.1/ref")
    ref = _chunk("ref", "doc-ref", doi="10.1/ref")
    index = CitationGraphIndex()
    index.index_chunks([seed, ref])
    original = [_result(seed, 0.9)]
    path_before = list(original[0].path)
    CitationGraphExpander(index).expand(original)
    assert original[0].path == path_before
    assert original[0].retriever == "hybrid"
    assert original[0].score == 0.9


def test_docstring_mentions_frontier_models() -> None:
    doc = CitationGraphExpander.__doc__ or ""
    assert "GPT-5.5" in doc
    assert "Claude Sonnet 4.6" in doc
    assert "Gemini 3.x" in doc
    assert "Kimi K2" in doc


def test_hop_decay_halves_after_one_hop() -> None:
    seed = _chunk("seed", "doc-seed", doi="10.1/seed", references="10.1/ref")
    ref = _chunk("ref", "doc-ref", doi="10.1/ref")
    index = CitationGraphIndex()
    index.index_chunks([seed, ref])
    expander = CitationGraphExpander(index, hop_decay=0.5, direction="cites")
    kept = expander.expand([_result(seed, 1.0)], include_seeds=False)
    assert kept[0].score == pytest.approx(0.5)
    assert math.isfinite(kept[0].score)
