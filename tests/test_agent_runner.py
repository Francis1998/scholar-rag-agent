"""Integration tests for the agent runner."""

import asyncio
import sqlite3
from collections.abc import Awaitable
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from agent.executor import Executor
from agent.models import AgentAnswer, AgentState, QueryPlan, StateTransition
from agent.observer import QueryAnalyzer
from agent.planner import Planner
from agent.runner import AgentRunner
from agent.safety import CancellationToken, SafetyLimits, with_timeout
from ingestion.pipeline import IngestionPipeline
from llm.fake import FakeLLMAdapter
from llm.providers import OpenAIAdapter
from retrieval.citations import CitationGrounder
from retrieval.dense import DenseRetriever
from retrieval.graph import GraphRAGBuilder
from retrieval.hybrid import HybridRetriever
from retrieval.hyde import HyDEExpander
from retrieval.models import Document, SearchResult
from retrieval.multihop import MultiHopRetriever
from retrieval.rerank import AdaptiveReranker
from retrieval.sparse import BM25Retriever
from storage.document_store import SQLiteDocumentStore
from storage.event_log import SQLiteEventLog
from storage.graph_store import SQLiteGraphStore


def _build_runner(tmp_path: Path) -> tuple[AgentRunner, SQLiteEventLog]:
    """Build a runner with one indexed GraphRAG fixture document.

    Args:
        tmp_path: Temporary directory for the SQLite database.

    Returns:
        Agent runner and its durable event log.
    """
    database_path = tmp_path / "agent.sqlite3"
    event_log = SQLiteEventLog(database_path)
    document_store = SQLiteDocumentStore(database_path)
    graph_store = SQLiteGraphStore(database_path)
    hybrid = HybridRetriever(DenseRetriever(), BM25Retriever(), HyDEExpander())
    pipeline = IngestionPipeline(document_store, hybrid, GraphRAGBuilder(graph_store))
    pipeline.ingest_documents(
        [
            Document(
                document_id="d1",
                title="GraphRAG Paper",
                text="GraphRAG improves scientific retrieval by following entity relationships.",
                source="fixture",
            )
        ]
    )
    executor = Executor(
        retriever=hybrid,
        multihop_retriever=MultiHopRetriever(graph_store),
        reranker=AdaptiveReranker(),
        llm=FakeLLMAdapter(),
        grounder=CitationGrounder(),
    )
    runner = AgentRunner(
        agent_id="test-agent",
        event_log=event_log,
        analyzer=QueryAnalyzer(),
        planner=Planner(),
        executor=executor,
        safety_limits=SafetyLimits(max_hops=1),
    )
    return runner, event_log


class RecordingExecutor(Executor):
    """Executor test double that records the retrieval result limit."""

    def __init__(self) -> None:
        """Create an executor double with no recorded limit."""
        self.max_results_seen: int | None = None
        self.answer_calls = 0

    async def retrieve(self, plan: QueryPlan, max_results: int = 8) -> list[SearchResult]:
        """Record the max retrieval results requested by the runner."""
        del plan
        self.max_results_seen = max_results
        return []

    async def answer(self, plan: QueryPlan, retrieved: list[SearchResult]) -> AgentAnswer:
        """Return a deterministic answer for runner integration tests."""
        del plan, retrieved
        self.answer_calls += 1
        return AgentAnswer(answer="No documents retrieved.", citations=[], claims=[])


async def test_agent_runner_completes_with_events(tmp_path: Path) -> None:
    """Agent runner completes a grounded query and persists all transitions."""
    runner, event_log = _build_runner(tmp_path)

    result = await runner.run("Summarize literature on GraphRAG for scientific retrieval.")

    assert result.state == AgentState.DONE
    assert result.plan is not None
    assert all(task.max_hops <= 1 for task in result.plan.tasks)
    assert result.answer is not None
    assert event_log.list_events(result.run_id)


async def test_agent_runner_persists_decision_log_and_transition_sequence(
    tmp_path: Path,
) -> None:
    """Successful runs should persist the plan and ordered state transitions."""
    runner, event_log = _build_runner(tmp_path)

    result = await runner.run("Summarize literature on GraphRAG for scientific retrieval.")
    events = event_log.list_events(result.run_id)
    transition_events = [event for event in events if event["event_type"] == "state_transition"]
    decision_events = [event for event in events if event["event_type"] == "decision_log"]

    assert [event["event_type"] for event in events] == [
        "state_transition",
        "decision_log",
        "state_transition",
        "state_transition",
        "evidence_snapshot",
        "generation_record",
        "state_transition",
        "state_transition",
    ]
    assert [
        (event["payload"]["from_state"], event["payload"]["to_state"])
        for event in transition_events
    ] == [
        ("IDLE", "PLANNING"),
        ("PLANNING", "RETRIEVING"),
        ("RETRIEVING", "REASONING"),
        ("REASONING", "ANSWERING"),
        ("ANSWERING", "DONE"),
    ]
    assert result.plan is not None
    assert len(decision_events) == 1
    assert decision_events[0]["payload"]["tasks"][0]["task_id"] == result.plan.tasks[0].task_id
    assert (
        decision_events[0]["payload"]["observation"]["intent"]
        == result.plan.observation.intent.value
    )


async def test_agent_runner_transitions_to_error_when_cancelled(tmp_path: Path) -> None:
    """Pre-cancelled runs should surface cancellation through the event log."""
    runner, event_log = _build_runner(tmp_path)
    token = CancellationToken()
    token.cancel()

    result = await runner.run("Summarize literature on GraphRAG.", token=token)
    events = event_log.list_events(result.run_id)

    assert result.state == AgentState.ERROR
    assert result.error == "agent run was cancelled"
    assert [event["event_type"] for event in events] == ["state_transition"]
    assert events[-1]["payload"]["from_state"] == "IDLE"
    assert events[-1]["payload"]["to_state"] == "ERROR"
    assert events[-1]["payload"]["payload"] == {"error": "agent run was cancelled"}


@pytest.mark.parametrize("phase", ["retrieval", "generation"])
@pytest.mark.parametrize("message", [None, "caller requested shutdown"])
async def test_agent_runner_journals_external_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str, message: str | None
) -> None:
    """Task cancellation is durable without changing or retrying the cancellation."""
    runner, _ = _build_runner(tmp_path)
    started = asyncio.Event()
    never_released = asyncio.Event()
    cancellations: list[asyncio.CancelledError] = []

    async def block_phase(*args: object, **kwargs: object) -> None:
        del args, kwargs
        started.set()
        await never_released.wait()

    async def observe_cancellation(
        awaitable: Awaitable[object], timeout_seconds: float, label: str
    ) -> object:
        try:
            return await with_timeout(awaitable, timeout_seconds, label)
        except asyncio.CancelledError as exc:
            cancellations.append(exc)
            raise

    retrieve = AsyncMock(wraps=runner._executor.retrieve)
    if phase == "retrieval":
        retrieve.side_effect = block_phase
    answer = AsyncMock(wraps=runner._executor.answer)
    provider = OpenAIAdapter(api_key="test-key")
    generate_once = AsyncMock(side_effect=block_phase)
    monkeypatch.setattr(provider, "_generate_once", generate_once)
    monkeypatch.setattr(runner._executor, "_llm", provider)
    monkeypatch.setattr(runner._executor, "retrieve", retrieve)
    monkeypatch.setattr(runner._executor, "answer", answer)
    monkeypatch.setattr("agent.runner.with_timeout", observe_cancellation)

    task = asyncio.create_task(runner.run("Summarize literature on GraphRAG."))
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        assert task.cancel(message)
        with pytest.raises(asyncio.CancelledError) as cancelled:
            await asyncio.wait_for(task, timeout=5)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert task.cancelled()
    assert len(cancellations) == 1
    assert cancelled.value is cancellations[0]
    assert cancelled.value.args == (() if message is None else (message,))
    retrieve.assert_awaited_once()
    if phase == "generation":
        answer.assert_awaited_once()
        generate_once.assert_awaited_once()
    else:
        answer.assert_not_awaited()
        generate_once.assert_not_awaited()

    events = SQLiteEventLog(tmp_path / "agent.sqlite3").list_events()
    assert len({event["run_id"] for event in events}) == 1
    transitions = [
        event["payload"] for event in events if event["event_type"] == "state_transition"
    ]
    expected_transitions = [("IDLE", "PLANNING"), ("PLANNING", "RETRIEVING")]
    expected_events = ["state_transition", "decision_log", "state_transition"]
    if phase == "generation":
        expected_transitions.append(("RETRIEVING", "REASONING"))
        expected_events.extend(["state_transition", "evidence_snapshot"])
    expected_transitions.append(("RETRIEVING" if phase == "retrieval" else "REASONING", "ERROR"))
    assert [
        (event["from_state"], event["to_state"]) for event in transitions
    ] == expected_transitions
    assert [event["event_type"] for event in events] == [*expected_events, "state_transition"]
    reason = "agent run was cancelled"
    if message is not None:
        reason += f": {message}"
    assert events[-1]["payload"]["payload"] == {"error": reason}


async def test_agent_runner_surfaces_cancellation_journal_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed cancellation write surfaces the storage error with cancellation context."""
    runner, event_log = _build_runner(tmp_path)
    started = asyncio.Event()
    never_released = asyncio.Event()
    failure = sqlite3.OperationalError("synthetic cancellation write failure")
    error_transitions: list[StateTransition] = []
    append_transition = event_log.append_transition

    async def block_retrieval(plan: QueryPlan, max_results: int = 8) -> list[SearchResult]:
        del plan, max_results
        started.set()
        await never_released.wait()
        return []

    def fail_error_transition(transition: StateTransition) -> int:
        if transition.to_state == AgentState.ERROR:
            error_transitions.append(transition)
            raise failure
        return append_transition(transition)

    monkeypatch.setattr(runner._executor, "retrieve", block_retrieval)
    monkeypatch.setattr(event_log, "append_transition", fail_error_transition)
    task = asyncio.create_task(runner.run("Summarize literature on GraphRAG."))
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        assert task.cancel("caller requested shutdown")
        with pytest.raises(sqlite3.OperationalError) as caught:
            await asyncio.wait_for(task, timeout=5)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert caught.value is failure
    assert isinstance(failure.__context__, asyncio.CancelledError)
    assert failure.__context__.args == ("caller requested shutdown",)
    assert len(error_transitions) == 1
    assert error_transitions[0].from_state == AgentState.RETRIEVING
    assert event_log.list_events()[-1]["payload"]["to_state"] == "RETRIEVING"


@pytest.mark.parametrize("phase", ["retrieval", "reasoning"])
async def test_agent_runner_keeps_timeouts_distinct_from_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    """A wait_for deadline still returns a labelled ERROR rather than cancellation."""
    runner, event_log = _build_runner(tmp_path)
    never_released = asyncio.Event()

    async def block_phase(*args: object, **kwargs: object) -> None:
        del args, kwargs
        await never_released.wait()

    setattr(runner._safety_limits, f"{phase}_timeout_seconds", 0.1)
    retrieve = AsyncMock(wraps=runner._executor.retrieve)
    answer = AsyncMock(wraps=runner._executor.answer)
    if phase == "retrieval":
        retrieve.side_effect = block_phase
    else:
        answer.side_effect = block_phase
    monkeypatch.setattr(runner._executor, "retrieve", retrieve)
    monkeypatch.setattr(runner._executor, "answer", answer)

    result = await asyncio.wait_for(runner.run("Summarize literature on GraphRAG."), timeout=5)

    assert result.state == AgentState.ERROR
    assert result.error == f"{phase} timed out after 0.1s"
    retrieve.assert_awaited_once()
    if phase == "reasoning":
        answer.assert_awaited_once()
    else:
        answer.assert_not_awaited()
    events = event_log.list_events(result.run_id)
    transitions = [
        event["payload"] for event in events if event["event_type"] == "state_transition"
    ]
    expected_states = ["PLANNING", "RETRIEVING"]
    if phase == "reasoning":
        expected_states.append("REASONING")
    assert [event["to_state"] for event in transitions] == [*expected_states, "ERROR"]
    assert events[-1]["payload"]["payload"] == {"error": result.error}


async def test_agent_runner_uses_configured_source_document_limit(tmp_path: Path) -> None:
    """Agent runner passes configured source-document limits into retrieval."""
    database_path = tmp_path / "agent.sqlite3"
    event_log = SQLiteEventLog(database_path)
    executor = RecordingExecutor()
    runner = AgentRunner(
        agent_id="test-agent",
        event_log=event_log,
        analyzer=QueryAnalyzer(),
        planner=Planner(),
        executor=executor,
        safety_limits=SafetyLimits(max_source_docs=20),
    )

    result = await runner.run("Summarize literature on GraphRAG for scientific retrieval.")

    assert result.state == AgentState.DONE
    assert executor.max_results_seen == 20
    assert executor.answer_calls == 1
