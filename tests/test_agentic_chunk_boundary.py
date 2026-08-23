"""Tests for deterministic semantic-boundary text chunking."""

import pytest

from retrieval.agentic_chunk_boundary import AgenticChunkBoundarySplitter
from retrieval.models import Document


def test_rejects_non_positive_max_chars() -> None:
    with pytest.raises(ValueError, match="max_chars"):
        AgenticChunkBoundarySplitter(max_chars=0)
    with pytest.raises(ValueError, match="max_chars"):
        AgenticChunkBoundarySplitter(max_chars=-5)


def test_rejects_invalid_min_chars() -> None:
    with pytest.raises(ValueError, match="min_chars"):
        AgenticChunkBoundarySplitter(max_chars=100, min_chars=-1)
    with pytest.raises(ValueError, match="min_chars"):
        AgenticChunkBoundarySplitter(max_chars=100, min_chars=100)
    with pytest.raises(ValueError, match="min_chars"):
        AgenticChunkBoundarySplitter(max_chars=100, min_chars=150)


def test_empty_or_whitespace_input_returns_empty_list() -> None:
    splitter = AgenticChunkBoundarySplitter(max_chars=100, min_chars=0)

    assert splitter.split("") == []
    assert splitter.split("   \n\n  ") == []


def test_text_within_limit_is_returned_as_single_chunk() -> None:
    splitter = AgenticChunkBoundarySplitter(max_chars=200, min_chars=0)
    text = "Graph neural networks improve molecular property prediction."

    assert splitter.split(text) == [text]


def test_splits_on_markdown_headings_keeping_heading_with_body() -> None:
    text = (
        "# Introduction\n\n"
        "GraphRAG helps connect entities across papers.\n\n"
        "## Methods\n\n"
        "We combine BM25 and dense retrieval with reciprocal rank fusion."
    )
    splitter = AgenticChunkBoundarySplitter(max_chars=200, min_chars=0)

    chunks = splitter.split(text)

    assert len(chunks) == 2
    assert chunks[0].startswith("# Introduction")
    assert "GraphRAG helps connect entities across papers." in chunks[0]
    assert chunks[1].startswith("## Methods")
    assert "reciprocal rank fusion." in chunks[1]


def test_oversized_heading_section_splits_on_blank_line_paragraphs() -> None:
    text = (
        "# Section\n\n"
        "First paragraph with some scientific content about retrieval.\n\n"
        "Second paragraph with different scientific content about ranking."
    )
    splitter = AgenticChunkBoundarySplitter(max_chars=90, min_chars=0)

    chunks = splitter.split(text)

    assert len(chunks) == 2
    assert chunks[0] == "# Section\n\nFirst paragraph with some scientific content about retrieval."
    assert chunks[1] == "Second paragraph with different scientific content about ranking."
    assert all(len(chunk) <= 90 for chunk in chunks)


def test_oversized_paragraph_splits_on_sentence_boundaries() -> None:
    text = (
        "Graph neural networks improve molecular property prediction. "
        "Dense retrieval complements sparse BM25 search well."
    )
    splitter = AgenticChunkBoundarySplitter(max_chars=65, min_chars=0)

    chunks = splitter.split(text)

    assert chunks == [
        "Graph neural networks improve molecular property prediction.",
        "Dense retrieval complements sparse BM25 search well.",
    ]
    assert all(len(chunk) <= 65 for chunk in chunks)


def test_oversized_sentence_splits_on_word_boundaries() -> None:
    text = "This sentence has many words that together exceed the small limit set here."
    splitter = AgenticChunkBoundarySplitter(max_chars=30, min_chars=0)

    chunks = splitter.split(text)

    assert len(chunks) > 1
    assert all(len(chunk) <= 30 for chunk in chunks)
    assert " ".join(chunks) == text


def test_single_oversized_word_falls_back_to_character_slicing() -> None:
    token = "x" * 75
    splitter = AgenticChunkBoundarySplitter(max_chars=30, min_chars=0)

    chunks = splitter.split(token)

    assert chunks == [token[0:30], token[30:60], token[60:75]]
    assert all(len(chunk) <= 30 for chunk in chunks)


def test_no_chunk_ever_exceeds_max_chars_across_all_boundary_levels() -> None:
    text = (
        "# Title\n\n"
        "Short intro.\n\n"
        "A very long paragraph without internal punctuation that just keeps "
        "going and going without any period to break it into sentences at all "
        "so it must fall back to word-level packing eventually here.\n\n"
        "## Next\n\n"
        "Final section."
    )
    splitter = AgenticChunkBoundarySplitter(max_chars=40, min_chars=0)

    chunks = splitter.split(text)

    assert chunks
    assert all(len(chunk) <= 40 for chunk in chunks)


def test_multiple_consecutive_headings_without_body_are_kept_separate() -> None:
    text = "# H1\n## H2\nBody text under second heading."
    splitter = AgenticChunkBoundarySplitter(max_chars=200, min_chars=0)

    chunks = splitter.split(text)

    assert chunks[0] == "# H1"
    assert chunks[1].startswith("## H2")


def test_preamble_before_first_heading_is_its_own_section() -> None:
    text = "Some preamble text before any heading appears.\n\n# Heading\n\nBody content."
    splitter = AgenticChunkBoundarySplitter(max_chars=200, min_chars=0)

    chunks = splitter.split(text)

    assert chunks[0] == "Some preamble text before any heading appears."
    assert chunks[1].startswith("# Heading")


def test_small_adjacent_heading_sections_merge_up_to_max_chars() -> None:
    text = "# A\n\nShort.\n\n# B\n\nAlso short."
    splitter = AgenticChunkBoundarySplitter(max_chars=200, min_chars=50)

    chunks = splitter.split(text)

    assert chunks == [text]


def test_merge_is_skipped_when_it_would_exceed_max_chars() -> None:
    text = "# A\n\nShort.\n\n# B\n\nAlso short."
    splitter = AgenticChunkBoundarySplitter(max_chars=25, min_chars=20)

    chunks = splitter.split(text)

    assert chunks == ["# A\n\nShort.", "# B\n\nAlso short."]
    assert all(len(chunk) <= 25 for chunk in chunks)


def test_small_middle_chunk_merges_with_previous_when_forward_merge_would_overflow() -> None:
    chunk0_body = "x" * 30
    chunk1_body = "x" * 5
    chunk2_body = "x" * 40
    text = f"# H0\n\n{chunk0_body}\n\n# H1\n\n{chunk1_body}\n\n# H2\n\n{chunk2_body}"
    splitter = AgenticChunkBoundarySplitter(max_chars=50, min_chars=15)

    chunks = splitter.split(text)

    assert len(chunks) == 2
    assert chunks[0] == f"# H0\n\n{chunk0_body}\n\n# H1\n\n{chunk1_body}"
    assert chunks[1] == f"# H2\n\n{chunk2_body}"
    assert all(len(chunk) <= 50 for chunk in chunks)


def test_default_bounds_keep_short_headings_together_and_long_text_bounded() -> None:
    splitter = AgenticChunkBoundarySplitter()
    text = "# A\n\nShort section one.\n\n# B\n\nShort section two."

    assert splitter.split(text) == [text]

    long_text = "word " * 400
    chunks = splitter.split(long_text)
    assert all(len(chunk) <= 1200 for chunk in chunks)
    assert " ".join(chunks) == long_text.strip()


def test_chunk_returns_retrieval_chunks_with_deterministic_ids() -> None:
    document = Document(
        document_id="doc-1",
        title="Paper",
        text=(
            "# Intro\n\nGraphRAG helps retrieval.\n\n## Methods\n\nWe use BM25 and dense retrieval."
        ),
        source="fixture",
        metadata={"year": "2024"},
    )
    splitter = AgenticChunkBoundarySplitter(max_chars=50, min_chars=0)

    chunks = splitter.chunk(document)
    chunks_again = splitter.chunk(document)

    assert len(chunks) == 2
    assert [chunk.chunk_id for chunk in chunks] == [chunk.chunk_id for chunk in chunks_again]
    assert all(chunk.document_id == "doc-1" for chunk in chunks)
    assert all(chunk.title == "Paper" for chunk in chunks)
    assert all(chunk.metadata["year"] == "2024" for chunk in chunks)
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == ["0", "1"]


def test_chunk_on_empty_document_text_returns_no_chunks() -> None:
    document = Document(document_id="doc-2", title="Empty", text="   ", source="fixture")
    splitter = AgenticChunkBoundarySplitter(max_chars=50, min_chars=0)

    assert splitter.chunk(document) == []
