"""Tests for deterministic Self-RAG-style reflection gating."""

import pytest

from retrieval.models import Chunk, SearchResult
from retrieval.self_rag_reflection_gate import (
    SelfRagReflectionGate,
    SelfRagSignal,
)


def _result(chunk_id: str, text: str, title: str = "Study") -> SearchResult:
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
    )


def test_supports_well_covered_evidence_without_conflicts() -> None:
    results = [
        _result(
            "supporting",
            "Graph neural retrieval improves molecular prediction accuracy.",
        )
    ]

    decision = SelfRagReflectionGate().evaluate(
        "graph neural retrieval molecular prediction",
        results,
    )

    assert decision.signal is SelfRagSignal.SUPPORT
    assert decision.term_coverage == 1.0
    assert decision.covered_terms == (
        "graph",
        "molecular",
        "neural",
        "prediction",
        "retrieval",
    )
    assert decision.missing_terms == ()
    assert decision.conflicts == ()


def test_returns_partial_for_usable_but_incomplete_coverage() -> None:
    decision = SelfRagReflectionGate().evaluate(
        "graph neural retrieval molecular prediction",
        [_result("partial", "Graph retrieval baselines were evaluated.")],
    )

    assert decision.signal is SelfRagSignal.PARTIAL
    assert decision.term_coverage == pytest.approx(0.4)
    assert decision.covered_terms == ("graph", "retrieval")
    assert decision.missing_terms == ("molecular", "neural", "prediction")


def test_refuses_empty_irrelevant_or_content_free_evidence() -> None:
    gate = SelfRagReflectionGate()

    empty = gate.evaluate("graph neural retrieval", [])
    irrelevant = gate.evaluate(
        "graph neural retrieval",
        [_result("ocean", "Coral temperatures changed.")],
    )
    content_free = gate.evaluate("the and of", [_result("paper", "Relevant evidence.")])

    assert empty.signal is SelfRagSignal.REFUSE
    assert irrelevant.signal is SelfRagSignal.REFUSE
    assert irrelevant.term_coverage == 0.0
    assert content_free.signal is SelfRagSignal.REFUSE
    assert content_free.reason == "Query contains no content terms to reflect on."
    assert (
        SelfRagReflectionGate(support_threshold=0.0, partial_threshold=0.0)
        .evaluate("graph retrieval", [])
        .signal
        is SelfRagSignal.REFUSE
    )


def test_opposing_direction_cues_downgrade_full_coverage_to_partial() -> None:
    results = [
        _result("positive", "The therapy improves survival outcomes."),
        _result("negative", "The therapy worsens survival outcomes."),
    ]

    decision = SelfRagReflectionGate().evaluate("therapy survival outcomes", results)

    assert decision.signal is SelfRagSignal.PARTIAL
    assert decision.term_coverage == 1.0
    assert len(decision.conflicts) == 1
    assert decision.conflicts[0].left_chunk_id == "positive"
    assert decision.conflicts[0].right_chunk_id == "negative"
    assert decision.conflicts[0].axis == "direction"
    assert "direction" in decision.reason


def test_negated_positive_cue_conflicts_with_positive_evidence() -> None:
    results = [
        _result("significant", "The treatment produced a significant response."),
        _result("not-significant", "The treatment response was not significant."),
    ]

    decision = SelfRagReflectionGate().evaluate("treatment response", results)

    assert decision.signal is SelfRagSignal.PARTIAL
    assert [conflict.axis for conflict in decision.conflicts] == ["significance"]


def test_unrelated_opposing_cues_do_not_create_a_conflict() -> None:
    results = [
        _result("therapy", "The therapy improves survival."),
        _result("dataset", "Dataset quality worsens under compression."),
    ]

    decision = SelfRagReflectionGate().evaluate("therapy survival", results)

    assert decision.signal is SelfRagSignal.SUPPORT
    assert decision.conflicts == ()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"support_threshold": -0.1},
        {"support_threshold": float("nan")},
        {"partial_threshold": 1.1},
        {"support_threshold": 0.2, "partial_threshold": 0.3},
    ],
)
def test_rejects_invalid_thresholds(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        SelfRagReflectionGate(**kwargs)
