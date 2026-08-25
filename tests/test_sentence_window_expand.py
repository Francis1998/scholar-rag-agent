"""Tests for sentence-window expansion of retrieved chunks."""

import pytest

from retrieval.models import Chunk, SearchResult
from retrieval.sentence_window_expand import SentenceWindowExpander


def _result(
    chunk_id: str,
    text: str,
    metadata: dict[str, str] | None = None,
    score: float = 1.0,
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


@pytest.mark.parametrize(
    "window_sentences",
    [-1, 11, True],  # bool must be rejected (isinstance(True, int) is True)
)
def test_rejects_invalid_window_sentences(window_sentences: int) -> None:
    with pytest.raises(ValueError, match="window_sentences"):
        SentenceWindowExpander(window_sentences=window_sentences)


def test_expands_with_neighboring_sentences_from_document_text() -> None:
    document = (
        "Background introduces the problem. "
        "Methods use graph neural networks. "
        "Results show improved accuracy. "
        "Limitations include small datasets."
    )
    chunk = "Methods use graph neural networks."
    original = _result("mid", chunk, metadata={"document_text": document})
    expander = SentenceWindowExpander(window_sentences=1)

    expanded = expander.expand([original])

    assert expanded[0] is not original
    assert expanded[0].chunk is not original.chunk
    assert original.chunk.text == chunk
    assert expanded[0].chunk.text == (
        "Background introduces the problem. "
        "Methods use graph neural networks. "
        "Results show improved accuracy."
    )
    assert expanded[0].retriever == "sentence_window"
    assert expanded[0].path == ["hybrid", "bm25"]
    assert expanded[0].score == 1.0


def test_falls_back_to_full_text_metadata_key() -> None:
    document = "Alpha sentence here. Beta sentence here. Gamma sentence here."
    original = _result(
        "beta",
        "Beta sentence here.",
        metadata={"full_text": document},
    )

    expanded = SentenceWindowExpander(window_sentences=1).expand([original])

    assert expanded[0].chunk.text == document


def test_leaves_text_unchanged_without_document_metadata() -> None:
    original = _result("solo", "Only the retrieved sentence.")

    expanded = SentenceWindowExpander(window_sentences=2).expand([original])

    assert expanded[0].chunk.text == original.chunk.text
    assert expanded[0].retriever == "bm25"
    assert expanded[0].path == ["hybrid"]
    assert expanded[0] is not original


def test_window_zero_does_not_add_neighbors() -> None:
    document = "One. Two. Three."
    original = _result("two", "Two.", metadata={"document_text": document})

    expanded = SentenceWindowExpander(window_sentences=0).expand([original])

    assert expanded[0].chunk.text == "Two."
    assert expanded[0].retriever == "bm25"


def test_clamps_window_at_document_boundaries() -> None:
    document = "First sentence. Second sentence. Third sentence."
    original = _result("first", "First sentence.", metadata={"document_text": document})

    expanded = SentenceWindowExpander(window_sentences=2).expand([original])

    assert expanded[0].chunk.text == document


def test_unlocated_chunk_text_is_left_unchanged() -> None:
    original = _result(
        "missing",
        "This span is not in the document.",
        metadata={"document_text": "Entirely different content. Still unrelated."},
    )

    expanded = SentenceWindowExpander(window_sentences=1).expand([original])

    assert expanded[0].chunk.text == original.chunk.text
    assert expanded[0].retriever == "bm25"
