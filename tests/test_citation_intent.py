"""Tests for rule-based citation-intent classification."""

import pytest

from retrieval.citation_intent import CitationIntent, CitationIntentClassifier
from retrieval.models import Chunk, SearchResult


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Give an overview of retrieval augmented generation", CitationIntent.BACKGROUND),
        ("Which methodology and protocol did the authors use?", CitationIntent.METHOD),
        ("What results demonstrate improved accuracy?", CitationIntent.RESULT),
        ("Compare dense retrieval versus BM25", CitationIntent.COMPARISON),
        ("Papers about protein folding", CitationIntent.UNKNOWN),
        ("", CitationIntent.UNKNOWN),
    ],
)
def test_classifier_assigns_supported_intents(query: str, expected: CitationIntent) -> None:
    assert CitationIntentClassifier().classify(query) is expected


def test_classifier_counts_rules_and_uses_stable_priority_for_ties() -> None:
    classifier = CitationIntentClassifier()

    assert classifier.classify("compare method") is CitationIntent.COMPARISON
    assert classifier.classify("method result") is CitationIntent.METHOD
    assert classifier.classify("results accuracy method") is CitationIntent.RESULT


def test_classifier_matches_phrases_without_substring_false_positives() -> None:
    classifier = CitationIntentClassifier()

    assert classifier.classify("Summarize prior work in this field") is CitationIntent.BACKGROUND
    assert classifier.classify("A surprising worldview") is CitationIntent.UNKNOWN


def test_classifier_attaches_metadata_without_mutating_results() -> None:
    result = SearchResult(
        chunk=Chunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            title="Dense and sparse retrieval",
            text="The methods are evaluated on a benchmark.",
            source="test",
            metadata={"section": "methods"},
        ),
        score=0.9,
        retriever="rrf",
        path=["dense", "bm25"],
    )

    annotated = CitationIntentClassifier().attach("Compare the two approaches", [result])

    assert annotated[0].chunk.metadata == {
        "section": "methods",
        "citation_intent": "comparison",
    }
    assert annotated[0].score == 0.9
    assert annotated[0].retriever == "rrf"
    assert annotated[0].path == ["dense", "bm25"]
    assert result.chunk.metadata == {"section": "methods"}
    assert annotated[0] is not result
    assert annotated[0].chunk is not result.chunk


def test_classifier_supports_a_custom_metadata_key_and_unknown_intent() -> None:
    result = SearchResult(
        chunk=Chunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            title="Paper",
            text="Evidence",
            source="test",
        ),
        score=1.0,
        retriever="dense",
    )

    annotated = CitationIntentClassifier("ranking_intent").attach("protein folding", [result])

    assert annotated[0].chunk.metadata["ranking_intent"] == "unknown"


def test_classifier_rejects_blank_metadata_keys() -> None:
    with pytest.raises(ValueError, match="metadata_key"):
        CitationIntentClassifier("   ")
