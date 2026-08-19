"""Tests for child-hit to parent-document expansion."""

import pytest

from retrieval.models import Chunk, Document, SearchResult
from retrieval.parent_document import ParentDocumentExpander


def _child(
    chunk_id: str,
    document_id: str,
    score: float,
    text: str = "matching child text",
) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id=document_id,
            title=f"Title {document_id}",
            text=text,
            source=f"source:{document_id}",
            metadata={"section": "methods"},
        ),
        score=score,
        retriever="dense",
        path=["hybrid"],
    )


def test_expander_replaces_child_text_with_raw_parent_text() -> None:
    child = _child("child-1", "doc-1", 0.9)
    expander = ParentDocumentExpander(
        {"doc-1": "Full parent introduction. Full parent methods and results."}
    )

    expanded = expander.expand([child])

    assert len(expanded) == 1
    assert expanded[0].chunk.text == ("Full parent introduction. Full parent methods and results.")
    assert expanded[0].chunk.chunk_id == "parent:doc-1"
    assert expanded[0].chunk.title == "Title doc-1"
    assert expanded[0].score == 0.9
    assert expanded[0].retriever == "parent_document"
    assert expanded[0].path == ["hybrid", "dense"]
    assert child.chunk.text == "matching child text"


def test_expander_supports_normalized_document_and_chunk_parents() -> None:
    document = Document(
        document_id="doc-normalized",
        title="Normalized parent",
        text="Complete normalized document.",
        source="doi:parent",
        metadata={"year": "2026"},
    )
    parent_chunk = Chunk(
        chunk_id="whole:doc-chunk",
        document_id="doc-chunk",
        title="Stored parent chunk",
        text="Complete stored chunk.",
        source="store",
        metadata={"kind": "parent"},
    )
    expander = ParentDocumentExpander({"doc-normalized": document, "doc-chunk": parent_chunk})

    expanded = expander.expand(
        [
            _child("child-a", "doc-normalized", 0.8),
            _child("child-b", "doc-chunk", 0.7),
        ]
    )

    assert expanded[0].chunk.model_dump() == {
        "chunk_id": "parent:doc-normalized",
        "document_id": "doc-normalized",
        "title": "Normalized parent",
        "text": "Complete normalized document.",
        "source": "doi:parent",
        "metadata": {"year": "2026"},
    }
    assert expanded[1].chunk == parent_chunk
    assert expanded[1].chunk is not parent_chunk


def test_expander_deduplicates_parent_and_uses_best_child_score() -> None:
    expander = ParentDocumentExpander(
        {
            "doc-a": "Full A.",
            "doc-b": "Full B.",
        }
    )
    results = [
        _child("a-low", "doc-a", 0.4),
        _child("b", "doc-b", 0.7),
        _child("a-high", "doc-a", 0.9),
    ]

    expanded = expander.expand(results)

    assert [result.chunk.document_id for result in expanded] == ["doc-a", "doc-b"]
    assert [result.score for result in expanded] == [0.9, 0.7]
    assert expanded[0].path == ["hybrid", "dense"]


def test_expander_skips_missing_parents_and_honors_bounds() -> None:
    expander = ParentDocumentExpander({"known": "Known full document."})
    results = [
        _child("missing", "missing", 1.0),
        _child("known", "known", 0.8),
    ]

    assert [result.chunk.document_id for result in expander.expand(results, top_k=1)] == ["known"]
    assert expander.expand(results, top_k=0) == []
    assert expander.expand([]) == []


def test_expander_rejects_unsupported_parent_values() -> None:
    with pytest.raises(TypeError, match="Document, Chunk, or string"):
        ParentDocumentExpander({"doc": 42})  # type: ignore[dict-item]
