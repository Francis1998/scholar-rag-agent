"""Tests for metadata-equals gate postprocessing."""

from retrieval.metadata_equals_gate import MetadataEqualsGate
from retrieval.models import Chunk, SearchResult


def _result(
    chunk_id: str,
    score: float,
    text: str = "",
    metadata: dict[str, str] | None = None,
) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id=f"doc-{chunk_id}",
            title=chunk_id,
            text=text,
            source="test",
            metadata=metadata or {},
        ),
        score=score,
        retriever="bm25",
        path=["hybrid"],
    )


def test_empty_required_is_passthrough() -> None:
    kept = MetadataEqualsGate().gate(
        [_result("a", 1.0, metadata={"lang": "en"}), _result("b", 0.9)]
    )
    assert [item.chunk.chunk_id for item in kept] == ["a", "b"]
    assert kept[0].retriever == "metadata_equals_gate"
    assert kept[0].path == ["hybrid", "bm25"]


def test_filters_on_equality() -> None:
    results = [
        _result("ok", 1.0, metadata={"lang": "en", "venue": "neurips"}),
        _result("bad_venue", 0.9, metadata={"lang": "en", "venue": "icml"}),
        _result("bad_lang", 0.8, metadata={"lang": "fr", "venue": "neurips"}),
    ]
    kept = MetadataEqualsGate(required={"lang": "en", "venue": "neurips"}).gate(results)
    assert [item.chunk.chunk_id for item in kept] == ["ok"]


def test_missing_key_drops() -> None:
    assert MetadataEqualsGate(required={"lang": "en"}).gate([_result("a", 1.0)]) == []


def test_empty_and_top_k() -> None:
    gate = MetadataEqualsGate(required={"lang": "en"})
    assert gate.gate([]) == []
    hits = [
        _result("a", 1.0, metadata={"lang": "en"}),
        _result("b", 0.9, metadata={"lang": "en"}),
    ]
    assert len(gate.gate(hits, top_k=1)) == 1
    assert gate.gate(hits, top_k=0) == []


def test_does_not_mutate_inputs() -> None:
    original = [_result("a", 1.0, metadata={"lang": "en"})]
    path_before = list(original[0].path)
    MetadataEqualsGate(required={"lang": "en"}).gate(original)
    assert original[0].path == path_before


def test_docstring_mentions_frontier_models() -> None:
    assert "Kimi K2" in (MetadataEqualsGate.__doc__ or "")
