"""Tests for SourceAuthorityGate postprocessor."""

import pytest

from retrieval.models import Chunk, SearchResult
from retrieval.source_authority_gate import SourceAuthorityGate


def _result(
    chunk_id: str,
    score: float = 1.0,
    metadata: dict[str, str] | None = None,
) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id=f"doc-{chunk_id}",
            title=chunk_id,
            text="body",
            source="test",
            metadata=metadata or {},
        ),
        score=score,
        retriever="bm25",
        path=["hybrid"],
    )


def test_rejects_invalid_alpha() -> None:
    with pytest.raises(ValueError, match="alpha"):
        SourceAuthorityGate(alpha=1.5)


def test_rejects_invalid_min_authority() -> None:
    with pytest.raises(ValueError, match="min_authority"):
        SourceAuthorityGate(min_authority=-0.1)


def test_empty_results() -> None:
    assert SourceAuthorityGate().gate([]) == []


def test_top_k_zero_returns_empty() -> None:
    assert SourceAuthorityGate().gate([_result("a")], top_k=0) == []


def test_boosts_high_authority_over_low() -> None:
    hits = [
        _result("low", 1.0, {"source_authority": "low"}),
        _result("high", 0.6, {"source_authority": "high"}),
    ]
    kept = SourceAuthorityGate(alpha=0.8).gate(hits)
    assert [r.chunk.chunk_id for r in kept] == ["high", "low"]


def test_numeric_source_authority() -> None:
    kept = SourceAuthorityGate(alpha=1.0).gate([_result("a", 0.2, {"source_authority": "0.9"})])
    assert kept[0].score == pytest.approx(0.9)


def test_filters_below_min_authority() -> None:
    hits = [
        _result("drop", 1.0, {"source_authority": "low"}),
        _result("keep", 0.5, {"source_authority": "high"}),
    ]
    kept = SourceAuthorityGate(min_authority=0.5).gate(hits)
    assert [r.chunk.chunk_id for r in kept] == ["keep"]


def test_venue_tier_map_lookup() -> None:
    gate = SourceAuthorityGate(
        alpha=1.0,
        venue_tiers={"Nature": "high", "blog": "low"},
    )
    kept = gate.gate(
        [
            _result("blog", 1.0, {"venue": "blog"}),
            _result("nature", 0.4, {"journal": "Nature"}),
        ]
    )
    assert [r.chunk.chunk_id for r in kept] == ["nature", "blog"]


def test_provenance_rewritten() -> None:
    kept = SourceAuthorityGate().gate([_result("a", metadata={"source_authority": "high"})])
    assert kept[0].retriever == "source_authority_gate"
    assert kept[0].path == ["hybrid", "bm25"]


def test_top_k_limits_output() -> None:
    hits = [
        _result("a", 1.0, {"source_authority": "high"}),
        _result("b", 1.0, {"source_authority": "medium"}),
        _result("c", 1.0, {"source_authority": "low"}),
    ]
    kept = SourceAuthorityGate().gate(hits, top_k=2)
    assert len(kept) == 2


def test_does_not_mutate_inputs() -> None:
    original = [_result("a", 1.0, {"source_authority": "high"})]
    path_before = list(original[0].path)
    score_before = original[0].score
    SourceAuthorityGate().gate(original)
    assert original[0].path == path_before
    assert original[0].retriever == "bm25"
    assert original[0].score == score_before


def test_docstring_mentions_frontier_models() -> None:
    assert "GPT-5.5" in (SourceAuthorityGate.__doc__ or "")
