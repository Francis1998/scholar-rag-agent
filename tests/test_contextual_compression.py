"""Tests for deterministic lexical contextual compression."""

import pytest

from retrieval.contextual_compression import ContextualCompressor
from retrieval.models import Chunk, SearchResult


def _result(chunk_id: str, text: str, score: float = 1.0) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id=f"doc-{chunk_id}",
            title=chunk_id,
            text=text,
            source="test",
            metadata={"section": "results"},
        ),
        score=score,
        retriever="rrf",
        path=["dense", "bm25"],
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_sentences_per_chunk": 0},
        {"max_chars_per_chunk": -1},
        {"min_overlap": 0},
        {"max_chars_per_chunk": 1.5},
    ],
)
def test_compressor_rejects_invalid_bounds(kwargs: dict[str, int | float]) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        ContextualCompressor(**kwargs)


def test_compressor_extracts_relevant_sentences_and_preserves_source_order() -> None:
    result = _result(
        "paper",
        (
            "The introduction describes prior work. "
            "Molecular datasets require careful splits. "
            "Transformers are useful for language. "
            "Graph neural networks predict molecular properties."
        ),
    )
    compressor = ContextualCompressor(max_sentences_per_chunk=2)

    compressed = compressor.compress("graph neural molecular prediction", [result])

    assert len(compressed) == 1
    assert compressed[0].chunk.text == (
        "Molecular datasets require careful splits. "
        "Graph neural networks predict molecular properties."
    )
    assert compressed[0].chunk.metadata == {"section": "results"}
    assert compressed[0].retriever == "contextual_compression"
    assert compressed[0].path == ["dense", "bm25", "rrf"]
    assert result.chunk.text.startswith("The introduction")


def test_compressor_selects_highest_overlap_before_sentence_position() -> None:
    result = _result(
        "ranking",
        (
            "Graph methods provide a baseline. "
            "Neural graph retrieval improves molecular graph ranking."
        ),
    )
    compressor = ContextualCompressor(max_sentences_per_chunk=1)

    compressed = compressor.compress("neural graph molecular retrieval", [result])

    assert compressed[0].chunk.text == ("Neural graph retrieval improves molecular graph ranking.")


def test_compressor_filters_irrelevant_chunks_and_blank_queries() -> None:
    results = [
        _result("irrelevant", "Ocean temperatures affect coral reefs."),
        _result("relevant", "Sparse retrieval uses lexical term matching."),
    ]
    compressor = ContextualCompressor(min_overlap=2)

    compressed = compressor.compress("lexical sparse retrieval", results)

    assert [result.chunk.chunk_id for result in compressed] == ["relevant"]
    assert compressor.compress("", results) == []
    assert compressor.compress("the and of", results) == []


def test_compressor_enforces_character_and_result_bounds_deterministically() -> None:
    results = [
        _result("first", "Retrieval " + "evidence " * 20),
        _result("second", "Retrieval evidence remains grounded."),
    ]
    compressor = ContextualCompressor(max_chars_per_chunk=24)

    first_run = compressor.compress("retrieval evidence", results, top_k=1)
    second_run = compressor.compress("retrieval evidence", results, top_k=1)

    assert len(first_run) == 1
    assert len(first_run[0].chunk.text) <= 24
    assert first_run[0].chunk.text == second_run[0].chunk.text
    assert compressor.compress("retrieval", results, top_k=0) == []
