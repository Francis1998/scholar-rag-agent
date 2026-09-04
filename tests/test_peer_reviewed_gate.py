"""Tests for PeerReviewedGate postprocessor."""

import pytest

from retrieval.models import Chunk, SearchResult
from retrieval.peer_reviewed_gate import PeerReviewedGate


def _result(chunk_id: str, peer: str | None, score: float = 1.0) -> SearchResult:
    metadata = {} if peer is None else {"peer_reviewed": peer}
    return SearchResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id="doc-a",
            title=chunk_id,
            text="body",
            source="test",
            metadata=metadata,
        ),
        score=score,
        retriever="bm25",
        path=["hybrid"],
    )


def test_rejects_non_bool_flag() -> None:
    with pytest.raises(ValueError, match="require_peer_reviewed"):
        PeerReviewedGate(require_peer_reviewed="yes")  # type: ignore[arg-type]


def test_empty_results() -> None:
    assert PeerReviewedGate().gate([]) == []


def test_filters_non_peer_reviewed() -> None:
    hits = [_result("a", "false"), _result("b", "true"), _result("c", None)]
    kept = PeerReviewedGate().gate(hits)
    assert [r.chunk.chunk_id for r in kept] == ["b"]


def test_passthrough_when_not_required() -> None:
    hits = [_result("a", "false"), _result("b", "true")]
    kept = PeerReviewedGate(require_peer_reviewed=False).gate(hits)
    assert len(kept) == 2


def test_accepts_yes_alias() -> None:
    kept = PeerReviewedGate().gate([_result("a", "yes")])
    assert len(kept) == 1


def test_provenance_rewritten() -> None:
    kept = PeerReviewedGate().gate([_result("a", "true")])
    assert kept[0].retriever == "peer_reviewed_gate"
    assert kept[0].path == ["hybrid", "bm25"]


def test_top_k_limits_output() -> None:
    hits = [_result("a", "true"), _result("b", "true"), _result("c", "true")]
    kept = PeerReviewedGate().gate(hits, top_k=2)
    assert len(kept) == 2


def test_top_k_zero_returns_empty() -> None:
    assert PeerReviewedGate().gate([_result("a", "true")], top_k=0) == []


def test_does_not_mutate_inputs() -> None:
    original = [_result("a", "true")]
    path_before = list(original[0].path)
    PeerReviewedGate().gate(original)
    assert original[0].path == path_before
    assert original[0].retriever == "bm25"


def test_docstring_mentions_frontier_models() -> None:
    assert "GPT-5.5" in (PeerReviewedGate.__doc__ or "")
