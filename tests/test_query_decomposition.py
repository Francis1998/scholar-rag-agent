"""Tests for deterministic multi-query decomposition."""

import pytest

from retrieval.query_decomposition import QueryDecomposer


def test_preserves_original_and_splits_on_conjunctions() -> None:
    parts = QueryDecomposer().decompose(
        "graph neural networks for molecules and sparse retrieval for ranking"
    )

    assert parts[0] == "graph neural networks for molecules and sparse retrieval for ranking"
    assert parts[1] == "graph neural networks for molecules"
    assert parts[2] == "sparse retrieval for ranking"
    assert len(parts) == 3


def test_splits_on_question_marks_and_semicolon_phrases() -> None:
    parts = QueryDecomposer().decompose(
        "What is HyDE? How does RRF fuse ranks; compare MMR novelty."
    )

    assert parts[0] == "What is HyDE? How does RRF fuse ranks; compare MMR novelty."
    assert "What is HyDE" in parts
    assert "How does RRF fuse ranks" in parts
    assert "compare MMR novelty" in parts


def test_deduplicates_case_insensitively_and_keeps_original_first() -> None:
    parts = QueryDecomposer().decompose("Graph retrieval and graph retrieval")

    assert parts == ["Graph retrieval and graph retrieval", "Graph retrieval"]


def test_blank_query_returns_empty_list() -> None:
    assert QueryDecomposer().decompose("   ") == []
    assert QueryDecomposer().decompose("???") == []


def test_max_parts_bounds_returned_subqueries() -> None:
    parts = QueryDecomposer().decompose(
        "background on transformers as well as methods for fine-tuning and findings about latency",
        max_parts=2,
    )

    assert len(parts) == 2
    assert parts[0].startswith("background on transformers")


@pytest.mark.parametrize("limit", [0, -1, 1.5, True])
def test_rejects_invalid_max_parts(limit: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        QueryDecomposer().decompose("graph retrieval and ranking", max_parts=limit)  # type: ignore[arg-type]
