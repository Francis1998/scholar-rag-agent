"""Tests for deterministic lexical query rewriting."""

import pytest

from retrieval.query_rewrite import QueryRewriter


def test_rewriter_expands_synonyms_drops_stopwords_and_normalizes_whitespace() -> None:
    rewriter = QueryRewriter(
        {
            "ai": ["artificial intelligence", "machine intelligence"],
            "tumour": "cancer",
        }
    )

    rewritten = rewriter.rewrite("  Effects   of AI on   tumour  ")

    assert rewritten == ("effects ai artificial intelligence machine intelligence tumour cancer")


def test_rewriter_builds_bounded_distinct_variants_for_fusion() -> None:
    rewriter = QueryRewriter(
        {
            "gnn": ["graph neural network", "graph convolutional network"],
            "retrieval": "information retrieval",
        }
    )

    variants = rewriter.variants("GNN for retrieval", max_variants=5)

    assert variants == [
        "gnn retrieval",
        ("gnn graph neural network graph convolutional network retrieval information retrieval"),
        "graph neural network retrieval",
        "graph convolutional network retrieval",
        "gnn information retrieval",
    ]


def test_rewriter_prefers_longest_phrase_match() -> None:
    rewriter = QueryRewriter(
        {
            "neural": "connectionist",
            "neural network": "deep learning",
        }
    )

    assert rewriter.rewrite("neural network for ranking") == (
        "neural network deep learning ranking"
    )


def test_rewriter_copies_the_map_and_supports_custom_stopwords() -> None:
    expansions = ["representation learning"]
    rewriter = QueryRewriter(
        {"embedding": expansions},
        stopwords={"please", "using"},
    )
    expansions.append("vector encoding")

    assert rewriter.rewrite("please rank using embedding") == (
        "rank embedding representation learning"
    )
    assert rewriter.variants("please") == []


@pytest.mark.parametrize(
    "synonyms",
    [
        {"the": "useful"},
        {"term": "the"},
        {"": "useful"},
        {"term": [42]},
    ],
)
def test_rewriter_rejects_invalid_synonym_entries(synonyms: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        QueryRewriter(synonyms)  # type: ignore[arg-type]


@pytest.mark.parametrize("limit", [0, -1, 1.5, True])
def test_rewriter_rejects_invalid_variant_bounds(limit: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        QueryRewriter({}).variants("graph retrieval", limit)  # type: ignore[arg-type]
