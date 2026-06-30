"""Integration tests for the agent runner."""

from pathlib import Path

from agent.executor import Executor
from agent.models import AgentAnswer, AgentState, QueryPlan
from agent.observer import QueryAnalyzer
from agent.planner import Planner
from agent.runner import AgentRunner
from agent.safety import CancellationToken, SafetyLimits
from ingestion.pipeline import IngestionPipeline
from llm.fake import FakeLLMAdapter
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

    async def retrieve(self, plan: QueryPlan, max_results: int = 8) -> list[SearchResult]:
        """Record the max retrieval results requested by the runner."""
        del plan
        self.max_results_seen = max_results
        return []

    async def answer(self, plan: QueryPlan, retrieved: list[SearchResult]) -> AgentAnswer:
        """Return a deterministic answer for runner integration tests."""
        del plan, retrieved
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
