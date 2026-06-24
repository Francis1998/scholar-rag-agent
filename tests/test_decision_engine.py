"""Tests for query observation and retrieval planning decisions."""

from __future__ import annotations

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
