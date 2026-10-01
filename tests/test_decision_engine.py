"""Tests for query observation and retrieval planning decisions."""

from __future__ import annotations

import pytest

from agent.models import QueryIntent, QueryObservation
from agent.observer import QueryAnalyzer
from agent.planner import Planner


def test_analyzer_classifies_comparison_intent() -> None:
    """Comparison queries are classified before retrieval planning."""

    observation = QueryAnalyzer().analyze("Compare GraphRAG versus BM25 for cancer papers")

    assert observation.intent == QueryIntent.COMPARISON
    assert "BM25" in observation.entities
    assert any("GraphRAG" in entity for entity in observation.entities)


def test_analyzer_classifies_hypothesis_validation_intent() -> None:
    """Hypothesis validation queries are classified before retrieval planning."""

    observation = QueryAnalyzer().analyze("Validate hypothesis that RAG improves triage")

    assert observation.intent == QueryIntent.HYPOTHESIS_VALIDATION


@pytest.mark.parametrize(
    "query",
    [
        pytest.param("Which GraphRAG features are unsupported?", id="unsupported"),
        pytest.param("Which changes invalidate cached GraphRAG results?", id="invalidate"),
        pytest.param("What causes indifference in decision theory?", id="indifference"),
        pytest.param("What does a comparer return?", id="comparer"),
        pytest.param("How many supporters use GraphRAG?", id="supporters"),
        pytest.param("When was the counterhypothesis first proposed?", id="counterhypothesis"),
        pytest.param("Why is the GraphRAG result unrefuted?", id="unrefuted"),
        pytest.param("Where is presummarized text cached?", id="presummarized"),
        pytest.param("How is resynthesized audio stored?", id="resynthesized"),
        pytest.param("What does literature_count measure?", id="underscore-boundary"),
        pytest.param("What does overview2 identify?", id="digit-boundary"),
        pytest.param("What does \u03b1support\u03b2 identify?", id="unicode-word-boundary"),
    ],
)
def test_analyzer_ignores_embedded_intent_markers(query: str) -> None:
    """Keyword substrings in other words or identifiers do not select a task."""
    assert QueryAnalyzer().analyze(query).intent == QueryIntent.FACTUAL_LOOKUP


@pytest.mark.parametrize(
    ("marker", "intent"),
    [
        *(
            (marker, QueryIntent.COMPARISON)
            for marker in ("compare", "compares", "compared", "versus", "difference", "differences")
        ),
        *(
            (marker, QueryIntent.HYPOTHESIS_VALIDATION)
            for marker in (
                "hypothesis",
                "validate",
                "validates",
                "validated",
                "support",
                "supports",
                "supported",
                "supporting",
                "refute",
                "refutes",
                "refuted",
            )
        ),
        *(
            (marker, QueryIntent.SYNTHESIS)
            for marker in (
                "synthesize",
                "synthesizes",
                "synthesized",
                "summarize",
                "summarizes",
                "summarized",
                "literature",
                "literatures",
                "overview",
                "overviews",
            )
        ),
    ],
)
def test_analyzer_preserves_whole_markers_and_inflections(marker: str, intent: QueryIntent) -> None:
    """Retain the existing marker forms, case insensitivity, and punctuation."""
    assert QueryAnalyzer().analyze(f"GraphRAG: {marker.upper()}?").intent == intent


@pytest.mark.parametrize(
    "query",
    [
        "GraphRAG vs BM25",
        "GraphRAG vs. BM25",
        "GraphRAG(VS.)BM25",
        "GraphRAG\tvs\nBM25",
        "GraphRAG versus BM25",
    ],
)
def test_analyzer_recognizes_comparison_boundaries(query: str) -> None:
    """Comparison separators work with punctuation and non-space whitespace."""
    assert QueryAnalyzer().analyze(query).intent == QueryIntent.COMPARISON


@pytest.mark.parametrize(
    ("query", "intent"),
    [
        ("Compare evidence that supports GraphRAG.", QueryIntent.COMPARISON),
        ("Support GraphRAG and compare it with BM25.", QueryIntent.COMPARISON),
        ("Summarize the difference between GraphRAG and BM25.", QueryIntent.COMPARISON),
        ("Summarize evidence to validate the hypothesis.", QueryIntent.HYPOTHESIS_VALIDATION),
        ("Refute the claim and summarize the literature.", QueryIntent.HYPOTHESIS_VALIDATION),
        ("Give an overview of unsupported GraphRAG features.", QueryIntent.SYNTHESIS),
        ("Does evidence support indifference theory?", QueryIntent.HYPOTHESIS_VALIDATION),
        ("GraphRAG vs. BM25: summarize supporting evidence.", QueryIntent.COMPARISON),
        ("What does GraphRAG connect?", QueryIntent.FACTUAL_LOOKUP),
    ],
)
def test_analyzer_preserves_intent_precedence(query: str, intent: QueryIntent) -> None:
    """Category precedence uses real markers, irrespective of their order."""
    assert QueryAnalyzer().analyze(query).intent == intent


def test_analyzer_preserves_trimmed_query_and_entities() -> None:
    """Classification does not rewrite query content or capitalized entities."""
    query = "\t  What  does GraphRAG connect to New York and BM25; GraphRAG?\n"

    observation = QueryAnalyzer().analyze(query)

    assert observation.original_query == query.strip()
    assert observation.intent == QueryIntent.FACTUAL_LOOKUP
    assert observation.entities == ["BM25", "GraphRAG", "New York", "What"]


def test_planner_emits_two_tasks_for_comparison() -> None:
    """Comparison plans retrieve evidence and contrasting findings."""

    observation = QueryObservation(
        original_query="Compare GraphRAG versus BM25",
        intent=QueryIntent.COMPARISON,
        entities=["GraphRAG", "BM25"],
    )

    plan = Planner().plan(run_id="run-1", observation=observation)

    assert [task.task_id for task in plan.tasks] == [
        "comparison-evidence",
        "contrast-findings",
    ]
    assert [task.max_hops for task in plan.tasks] == [3, 2]


def test_planner_emits_support_and_counter_tasks_for_hypothesis() -> None:
    """Hypothesis plans include supporting and refuting evidence retrieval."""

    observation = QueryObservation(
        original_query="Validate hypothesis that Kimi improves extraction",
        intent=QueryIntent.HYPOTHESIS_VALIDATION,
        entities=["Kimi"],
    )

    plan = Planner().plan(run_id="run-2", observation=observation)

    assert [task.task_id for task in plan.tasks] == ["supporting-evidence", "counter-evidence"]
    assert plan.tasks[0].query.startswith("evidence supporting")
    assert plan.tasks[1].query.startswith("evidence refuting")


def test_plan_rationale_trace_matches_task_ids() -> None:
    """Planner rationale traces remain aligned with emitted retrieval tasks."""

    observation = QueryObservation(
        original_query="Summarize agentic RAG evaluation",
        intent=QueryIntent.SYNTHESIS,
        entities=[],
    )

    plan = Planner().plan(run_id="run-3", observation=observation)

    task_ids = [task.task_id for task in plan.tasks]
    traced_task_ids = [trace.split(":", maxsplit=1)[0] for trace in plan.rationale_trace]
    assert traced_task_ids == task_ids
