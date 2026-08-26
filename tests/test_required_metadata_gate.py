"""Tests for the required-metadata retrieval gate."""

from retrieval.models import Chunk, SearchResult
from retrieval.required_metadata_gate import RequiredMetadataGate


def _result(
    chunk_id: str,
    metadata: dict[str, str] | None = None,
    score: float = 1.0,
) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id=f"doc-{chunk_id}",
            title=chunk_id,
            text=f"text for {chunk_id}",
            source="test",
            metadata={} if metadata is None else metadata,
        ),
        score=score,
        retriever="bm25",
        path=["hybrid"],
    )


def test_empty_required_keys_is_passthrough() -> None:
    gate = RequiredMetadataGate(required_keys=[])
    results = [_result("a", {"year": "2024"}), _result("b", {})]

    filtered = gate.filter(results)

    assert filtered == results
    assert filtered[0] is results[0]


def test_drops_missing_key() -> None:
    gate = RequiredMetadataGate(required_keys=["doi"])
    keep = _result("keep", {"doi": "10.1000/xyz"})
    drop = _result("drop", {"year": "2024"})

    filtered = gate.filter([keep, drop])

    assert filtered == [keep]
    assert filtered[0] is keep


def test_drops_empty_string_value() -> None:
    gate = RequiredMetadataGate(required_keys=["doi"])
    blank = _result("blank", {"doi": ""})
    whitespace = _result("ws", {"doi": "   "})
    ok = _result("ok", {"doi": "10.1000/abc"})

    filtered = gate.filter([blank, whitespace, ok])

    assert filtered == [ok]


def test_requires_all_keys() -> None:
    gate = RequiredMetadataGate(required_keys=["doi", "year"])
    both = _result("both", {"doi": "10.1/x", "year": "2023"})
    only_doi = _result("doi-only", {"doi": "10.1/y"})
    only_year = _result("year-only", {"year": "2022"})

    filtered = gate.filter([both, only_doi, only_year])

    assert [result.chunk.chunk_id for result in filtered] == ["both"]


def test_preserves_input_order_and_identity() -> None:
    gate = RequiredMetadataGate(required_keys=["source_type"])
    first = _result("first", {"source_type": "arxiv"}, score=0.2)
    second = _result("second", {"source_type": "pubmed"}, score=0.9)
    third = _result("third", {}, score=1.0)

    filtered = gate.filter([first, second, third])

    assert filtered == [first, second]
    assert filtered[0] is first
    assert filtered[1] is second


def test_empty_results() -> None:
    gate = RequiredMetadataGate(required_keys=["doi"])
    assert gate.filter([]) == []
    assert RequiredMetadataGate(required_keys=[]).filter([]) == []


def test_does_not_mutate_metadata() -> None:
    gate = RequiredMetadataGate(required_keys=["year"])
    result = _result("x", {"year": "2024"})
    original = dict(result.chunk.metadata)

    filtered = gate.filter([result])

    assert filtered == [result]
    assert result.chunk.metadata == original


def test_tuple_and_list_required_keys_equivalent() -> None:
    results = [
        _result("a", {"doi": "10.1/a", "year": "2020"}),
        _result("b", {"doi": "10.1/b"}),
    ]
    from_list = RequiredMetadataGate(required_keys=["doi", "year"]).filter(results)
    from_tuple = RequiredMetadataGate(required_keys=("doi", "year")).filter(results)

    assert from_list == from_tuple == [results[0]]
