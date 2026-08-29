"""Tests for retracted filter postprocessing."""

import pytest

from retrieval.models import Chunk, SearchResult
from retrieval.retracted_filter import RetractedFilter


def _result(chunk_id: str, score: float, metadata: dict[str, str]) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id=f"doc-{chunk_id}",
            title=chunk_id,
            text=f"text for {chunk_id}",
            source="test",
            metadata=metadata,
        ),
        score=score,
        retriever="bm25",
        path=["hybrid"],
    )


def test_rejects_invalid_mode() -> None:
    with pytest.raises(ValueError, match="mode"):
        RetractedFilter(mode="drop")


def test_filter_removes_retracted_when_survivors_exist() -> None:
    good = _result("g", 0.1, {"status": "published"})
    bad = _result("b", 0.9, {"retracted": "true"})
    filtered = RetractedFilter().filter([bad, good])
    assert [item.chunk.chunk_id for item in filtered] == ["g"]


def test_filter_keeps_all_when_everything_retracted() -> None:
    rows = [_result("a", 0.2, {"status": "retracted"}), _result("b", 0.1, {"is_retracted": "yes"})]
    filtered = RetractedFilter().filter(rows)
    assert len(filtered) == 2


def test_demote_mode_soft_ranks() -> None:
    good = _result("g", 0.0, {"status": "published"})
    bad = _result("b", 1.0, {"publication_status": "Retracted Article"})
    demoted = RetractedFilter(alpha=1.0, mode="demote").filter([bad, good])
    assert [item.chunk.chunk_id for item in demoted] == ["g", "b"]
    assert demoted[0].retriever == "retracted_filter"


def test_top_k_empty() -> None:
    assert RetractedFilter().filter([]) == []
    rows = [_result("a", 0.0, {}), _result("b", 0.0, {"retracted": "1"})]
    assert len(RetractedFilter().filter(rows, top_k=1)) == 1
