"""Tests for deterministic corrective-RAG relevance gating."""

import pytest

from retrieval.corrective_rag import (
    CorrectiveRagGate,
    CorrectiveRagSignal,
)
from retrieval.models import Chunk, SearchResult


def _result(chunk_id: str, title: str, text: str) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id=f"doc-{chunk_id}",
            title=title,
            text=text,
            source="test",
        ),
        score=0.8,
        retriever="rrf",
        path=["dense", "bm25"],
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"keep_threshold": -0.1},
        {"keep_threshold": float("inf")},
        {"filter_threshold": 1.1},
        {"keep_threshold": 0.2, "filter_threshold": 0.3},
    ],
)
def test_gate_rejects_invalid_thresholds(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        CorrectiveRagGate(**kwargs)


def test_gate_keeps_only_results_meeting_the_strong_threshold() -> None:
    results = [
        _result("strong", "Molecular graph study", "Neural prediction improves accuracy."),
        _result("borderline", "Graph baseline", "A conventional method."),
        _result("irrelevant", "Ocean study", "Coral temperatures increased."),
    ]
    gate = CorrectiveRagGate(keep_threshold=0.75, filter_threshold=0.25)

    decision = gate.evaluate("molecular graph neural prediction", results)

    assert decision.signal is CorrectiveRagSignal.KEEP
    assert [result.chunk.chunk_id for result in decision.results] == ["strong"]
    assert decision.rewrite_hint is None
    assert results[0].path == ["dense", "bm25"]


def test_gate_filters_to_borderline_results_when_none_are_strong() -> None:
    results = [
        _result("borderline", "Graph retrieval", "A baseline evaluation."),
        _result("irrelevant", "Ocean study", "Coral temperatures increased."),
    ]
    gate = CorrectiveRagGate(keep_threshold=0.8, filter_threshold=0.5)

    decision = gate.evaluate("graph retrieval evaluation accuracy", results)

    assert decision.signal == "filter"
    assert [result.chunk.chunk_id for result in decision.results] == ["borderline"]
    assert decision.rewrite_hint is None


def test_gate_requests_rewrite_when_results_have_no_usable_overlap() -> None:
    result = _result("irrelevant", "Ocean study", "Coral temperatures increased.")

    decision = CorrectiveRagGate().evaluate("graph neural retrieval", [result])

    assert decision.signal is CorrectiveRagSignal.RETRY_REWRITE
    assert decision.results == []
    assert decision.rewrite_hint == (
        "Try synonyms or broader terminology for: graph, neural, retrieval."
    )


def test_gate_handles_empty_results_and_queries_without_content_terms() -> None:
    gate = CorrectiveRagGate()

    empty_results = gate.evaluate("sparse retrieval", [])
    empty_query = gate.evaluate("the and of", [_result("paper", "Sparse", "Retrieval")])

    assert empty_results.signal is CorrectiveRagSignal.RETRY_REWRITE
    assert empty_results.rewrite_hint is not None
    assert empty_query.results == []
    assert empty_query.rewrite_hint == "Add specific content terms before retrying retrieval."
