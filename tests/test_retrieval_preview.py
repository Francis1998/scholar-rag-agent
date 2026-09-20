"""End-to-end contracts for generation-free, non-persistent retrieval inspection."""

import asyncio
import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from scripts.demo_evidence_export import offline_settings

from agent.evidence import CaptureLimits, EvidenceSnapshot
from agent.models import AgentAnswer, QueryPlan
from api.application import create_app
from api.dependencies import AppContainer
from llm.fake import FakeLLMAdapter
from llm.providers import AnthropicAdapter, GeminiAdapter, KimiAdapter, OpenAIAdapter
from llm.router import RoutingLLMAdapter
from retrieval.hyde import HyDEExpander
from retrieval.models import Chunk, Document, SearchResult


def denied(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("Retrieval inspection must not generate, write events, or use HTTP.")


@dataclass
class PreviewAPI:
    application: FastAPI
    container: AppContainer
    client: TestClient
    database_path: Path

    def add(self, document_id: str, text: str) -> Chunk:
        chunk = Chunk(
            chunk_id=f"chunk-{document_id}",
            document_id=document_id,
            title=f"Synthetic {document_id}",
            text=text,
            source="synthetic:preview",
            metadata={"license": "synthetic-only"},
        )
        self.container.document_store.add_documents(
            [Document(**chunk.model_dump(exclude={"chunk_id"}))], [chunk]
        )
        self.container.hybrid_retriever.add_chunks([chunk])
        self.container.graph_builder.index_chunks([chunk])
        return chunk

    def preview(
        self, query: str = "What does GraphRAG retrieve?", **scope: object
    ) -> dict[str, Any]:
        response = self.client.post("/retrieve", json={"query": query, **scope})
        assert response.status_code == 200, response.text
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["x-content-type-options"] == "nosniff"
        result: dict[str, Any] = response.json()
        return result


@pytest.fixture
def preview_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[PreviewAPI]:
    settings = offline_settings(tmp_path / "preview.sqlite3").model_copy(
        update={"max_source_docs": 2, "max_hops": 1}
    )
    application = create_app(settings)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", denied)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", denied)
    with TestClient(application) as client:
        yield PreviewAPI(application, application.state.container, client, settings.database_path)


@pytest.mark.parametrize(
    ("query", "intent", "tasks"),
    [
        ("  What does GraphRAG retrieve?  ", "factual_lookup", 1),
        ("Summarize literature on GraphRAG.", "synthesis", 1),
        ("Compare GraphRAG versus Retrieval.", "comparison", 2),
        ("Validate the hypothesis that GraphRAG supports Retrieval.", "hypothesis_validation", 2),
    ],
)
@pytest.mark.parametrize("scope", [None, ["doc-a"], ["not-ingested"]])
def test_preview_matches_exact_query_snapshot_without_creating_a_run(
    preview_api: PreviewAPI,
    monkeypatch: pytest.MonkeyPatch,
    query: str,
    intent: str,
    tasks: int,
    scope: list[str] | None,
) -> None:
    first = preview_api.add(
        "doc-a", "GraphRAG connects Retrieval. " * 20 + "\nExact Unicode: caf\u00e9, \u7814\u7a76."
    )
    preview_api.add("doc-b", "GraphRAG retrieves EXCLUDED_MARKER evidence.")
    scope_args = {} if scope is None else {"document_ids": scope}
    executor = preview_api.container.runner._executor
    with monkeypatch.context() as patch:
        patch.setattr(executor, "answer", denied)
        patch.setattr(preview_api.container.llm, "generate", denied)
        patch.setattr(preview_api.container.event_log, "append_event", denied)
        patch.setattr(preview_api.container.event_log, "append_transition", denied)
        preview = preview_api.preview(query, **scope_args)
    assert preview_api.container.event_log.list_events() == []
    assert preview_api.client.get("/runs").json() == {"runs": [], "next_cursor": None}
    assert set(preview) == {
        "schema_version",
        "plan",
        "configuration",
        "sources",
        "context",
        "context_sha256",
        "context_format",
        "capture_limits",
    }
    assert "run_id" not in preview["plan"]
    assert preview["schema_version"] == "1.0"
    assert preview["plan"]["observation"]["original_query"] == query.strip()
    assert preview["plan"]["observation"]["intent"] == intent
    assert preview["plan"]["observation"]["document_ids"] == scope
    assert len(preview["plan"]["tasks"]) == tasks
    assert len(preview["plan"]["rationale_trace"]) == tasks
    assert all(task["max_hops"] <= 1 for task in preview["plan"]["tasks"])
    assert len(preview["sources"]) <= 2
    if scope == ["doc-a"]:
        assert preview["sources"][0]["chunk"] == first.model_dump()
        assert "EXCLUDED_MARKER" not in preview["context"]
    elif scope == ["not-ingested"]:
        assert preview["sources"] == []
        assert preview["context"] == ""
    assert preview["context_sha256"] == hashlib.sha256(preview["context"].encode()).hexdigest()

    response = preview_api.client.post("/query", json={"query": query, **scope_args})
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["state"] == "DONE", result
    events = preview_api.container.event_log.list_events(result["run_id"])
    snapshot = next(
        event["payload"] for event in events if event["event_type"] == "evidence_snapshot"
    )
    assert preview["sources"] == snapshot["sources"]
    assert preview["context"] == snapshot["request"]["context"]
    assert preview["context_sha256"] == snapshot["context_sha256"]
    assert preview["context_format"] == snapshot["context_format"]
    assert preview["capture_limits"] == snapshot["limits"]
    assert preview["configuration"] == events[0]["payload"]["payload"]["configuration"]
    assert preview["plan"] == {
        key: value for key, value in result["plan"].items() if key != "run_id"
    }


def test_api_never_calls_any_generator_even_with_all_provider_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = offline_settings(tmp_path / "live-configured.sqlite3").model_copy(
        update={
            "openai_api_key": "synthetic-unused-openai",
            "anthropic_api_key": "synthetic-unused-anthropic",
            "gemini_api_key": "synthetic-unused-gemini",
            "moonshot_api_key": "synthetic-unused-kimi",
        }
    )
    calls = []
    for adapter in (
        FakeLLMAdapter,
        RoutingLLMAdapter,
        OpenAIAdapter,
        AnthropicAdapter,
        GeminiAdapter,
        KimiAdapter,
    ):
        generate = AsyncMock(side_effect=denied)
        monkeypatch.setattr(adapter, "generate", generate)
        calls.append(generate)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", denied)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", denied)
    application = create_app(settings)
    container: AppContainer = application.state.container
    monkeypatch.setattr(container.runner._executor._grounder, "ground", denied)
    with TestClient(application) as client:
        ingested = client.post(
            "/ingest/text",
            json={"title": "Synthetic note", "text": "GraphRAG connects local evidence."},
        )
        assert ingested.status_code == 200
        response = client.post("/retrieve", json={"query": "GraphRAG evidence"})
        assert response.status_code == 200, response.text
        assert response.json()["sources"]
        assert "synthetic-unused" not in response.text
    assert container.event_log.list_events() == []
    for generate in calls:
        generate.assert_not_awaited()


@pytest.mark.parametrize("scope", [None, [], "", [" "], [42], {}, ["x" * 129], ["d"] * 101])
def test_invalid_http_scope_is_422_without_retrieval(
    preview_api: PreviewAPI, monkeypatch: pytest.MonkeyPatch, scope: object
) -> None:
    monkeypatch.setattr(preview_api.container.runner._executor, "retrieve", denied)
    response = preview_api.client.post(
        "/retrieve", json={"query": "GraphRAG", "document_ids": scope}
    )
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"][:2] == ["body", "document_ids"]
    assert preview_api.container.event_log.list_events() == []


@pytest.mark.parametrize("query", ["", "   ", None, 12])
def test_invalid_preview_query_is_422(preview_api: PreviewAPI, query: object) -> None:
    response = preview_api.client.post("/retrieve", json={"query": query})
    assert response.status_code == 422
    assert preview_api.container.event_log.list_events() == []


def test_scope_normalization_and_real_graph_bridge_filtering(
    preview_api: PreviewAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    preview_api.container.runner._safety_limits.max_hops = 3
    preview_api.container.runner._executor._multihop_retriever._max_depth = 3
    preview_api.add("head", "Alpha selected local evidence.")
    preview_api.add("tail", "Beta Gamma selected followup evidence.")
    preview_api.add("bridge", "Alpha Beta EXCLUDED_MARKER bridge.")
    graph = preview_api.container.runner._executor._multihop_retriever
    retrieve = AsyncMock(wraps=graph.retrieve)
    neighbours = Mock(wraps=preview_api.container.graph_store.neighbours)
    monkeypatch.setattr(graph, "retrieve", retrieve)
    monkeypatch.setattr(preview_api.container.graph_store, "neighbours", neighbours)
    result = preview_api.preview("compare Alpha", document_ids=[" head ", "tail", "head"])
    assert result["plan"]["observation"]["document_ids"] == ["head", "tail"]
    assert {item["chunk"]["document_id"] for item in result["sources"]} == {"head", "tail"}
    assert "EXCLUDED_MARKER" not in result["context"]
    assert retrieve.await_count == 2
    assert all(call.kwargs["document_ids"] == ("head", "tail") for call in retrieve.await_args_list)
    assert neighbours.call_count > 0
    assert all(
        call.kwargs["document_ids"] == ("head", "tail") for call in neighbours.call_args_list
    )
    head = next(item for item in result["sources"] if item["chunk"]["document_id"] == "head")
    tail = next(item for item in result["sources"] if item["chunk"]["document_id"] == "tail")
    assert head["path"][-1] == "multihop"
    assert tail["path"][-1] == "rrf"


async def test_python_entrypoint_copies_scope_and_limits_before_await(
    preview_api: PreviewAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    preview_api.add("doc-a", "GraphRAG selected evidence.")
    preview_api.add("doc-b", "GraphRAG excluded evidence.")
    runner = preview_api.container.runner
    executor = runner._executor
    started, release = asyncio.Event(), asyncio.Event()
    retrieve = executor.retrieve

    async def held(
        plan: QueryPlan, max_results: int = 8, *, document_ids: tuple[str, ...] | None = None
    ) -> list[SearchResult]:
        started.set()
        await release.wait()
        assert max_results == 2
        assert document_ids == ("doc-a",)
        assert all(task.max_hops == 1 for task in plan.tasks)
        return await retrieve(plan, max_results, document_ids=document_ids)

    monkeypatch.setattr(executor, "retrieve", held)
    monkeypatch.setattr(preview_api.container.llm, "generate", denied)
    supplied = ["doc-a"]
    pending = asyncio.create_task(runner.preview("Compare GraphRAG", document_ids=supplied))
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        supplied[:] = ["doc-b"]
        runner._safety_limits.max_source_docs = 1
        runner._safety_limits.max_hops = 0
        runner._safety_limits.retrieval_timeout_seconds = 0.0001
        runner._safety_limits.reasoning_timeout_seconds = 0.0001
    finally:
        release.set()
    result = await asyncio.wait_for(pending, timeout=2)
    assert result.plan.observation.document_ids == ("doc-a",)
    assert result.configuration.max_source_docs == 2
    assert result.configuration.max_hops == 1
    assert result.configuration.retrieval_timeout_seconds == 30
    assert result.configuration.reasoning_timeout_seconds == 60
    assert {source.chunk.document_id for source in result.sources} == {"doc-a"}
    assert preview_api.container.event_log.list_events() == []


async def test_python_invalid_scope_fails_before_work(preview_api: PreviewAPI) -> None:
    with pytest.raises(ValueError):
        await preview_api.container.runner.preview("retrieval", document_ids=[])
    assert preview_api.container.event_log.list_events() == []


@pytest.mark.parametrize(
    ("phase", "code"),
    [
        ("planner", "planning_failed"),
        ("retriever", "retrieval_failed"),
        ("reranker", "context_preparation_failed"),
    ],
)
def test_runtime_failure_is_sanitized_http_error_not_empty_success(
    preview_api: PreviewAPI, monkeypatch: pytest.MonkeyPatch, phase: str, code: str
) -> None:
    runner = preview_api.container.runner
    error = RuntimeError("PRIVATE_SOURCE and secret-key must not leak")
    if phase == "planner":
        failing = Mock(side_effect=error)
        monkeypatch.setattr(runner._planner, "plan", failing)
    else:
        failing = AsyncMock(side_effect=error)
        target = runner._executor if phase == "retriever" else runner._executor._reranker
        monkeypatch.setattr(target, "retrieve" if phase == "retriever" else "rerank", failing)
    response = preview_api.client.post("/retrieve", json={"query": "GraphRAG"})
    assert response.status_code == 500
    assert response.json()["detail"]["code"] == code
    assert response.headers["cache-control"] == "no-store"
    assert "PRIVATE_SOURCE" not in response.text
    assert "secret-key" not in response.text
    assert "sources" not in response.json()
    assert failing.call_count == 1
    assert preview_api.container.event_log.list_events() == []


@pytest.mark.parametrize("phase", ["retrieval", "context_preparation"])
def test_preview_timeout_has_504_and_no_events(
    preview_api: PreviewAPI, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    cancelled = []

    async def block(*args: object, **kwargs: object) -> NoReturn:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.append(True)
            raise
        raise AssertionError("timeout did not fire")

    runner = preview_api.container.runner
    runner._safety_limits.retrieval_timeout_seconds = 0.01
    runner._safety_limits.reasoning_timeout_seconds = 0.01
    target = runner._executor if phase == "retrieval" else runner._executor._reranker
    monkeypatch.setattr(target, "retrieve" if phase == "retrieval" else "rerank", block)
    monkeypatch.setattr(preview_api.container.llm, "generate", denied)
    response = preview_api.client.post("/retrieve", json={"query": "GraphRAG"})
    assert response.status_code == 504
    assert response.json()["detail"]["code"] == f"{phase}_timeout"
    assert response.headers["cache-control"] == "no-store"
    assert cancelled == [True]
    assert preview_api.container.event_log.list_events() == []


@pytest.mark.parametrize("stage", ["retrieve", "rerank"])
def test_scope_leak_is_an_error_not_filtered_success(
    preview_api: PreviewAPI, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    excluded = preview_api.add("excluded", "PRIVATE_SOURCE GraphRAG")
    leaked = [SearchResult(chunk=excluded, score=1, retriever="bad-component")]
    executor = preview_api.container.runner._executor
    target = executor if stage == "retrieve" else executor._reranker
    monkeypatch.setattr(target, stage, AsyncMock(return_value=leaked))
    response = preview_api.client.post(
        "/retrieve", json={"query": "GraphRAG", "document_ids": ["selected"]}
    )
    assert response.status_code == 500
    assert "PRIVATE_SOURCE" not in response.text
    assert preview_api.container.event_log.list_events() == []


@pytest.mark.parametrize("overflow", ["source_count", "context_bytes", "metadata_bytes"])
def test_capture_overflow_fails_explicitly_without_truncation(
    preview_api: PreviewAPI, monkeypatch: pytest.MonkeyPatch, overflow: str
) -> None:
    source = preview_api.add("doc-a", "GraphRAG evidence")
    if overflow == "source_count":
        results = [
            SearchResult(
                chunk=source.model_copy(update={"chunk_id": f"chunk-{index}"}),
                score=float(index),
                retriever="too-many",
            )
            for index in range(3)
        ]
    else:
        if overflow == "context_bytes":
            source.text = "x" * CaptureLimits().max_context_bytes
        else:
            source.metadata["large"] = "x" * CaptureLimits().max_snapshot_bytes
        results = [SearchResult(chunk=source, score=1, retriever="oversized")]
    monkeypatch.setattr(
        preview_api.container.runner._executor._reranker, "rerank", AsyncMock(return_value=results)
    )
    response = preview_api.client.post("/retrieve", json={"query": "GraphRAG"})
    assert response.status_code == 500
    assert response.json()["detail"]["code"] == "context_preparation_failed"
    assert preview_api.container.event_log.list_events() == []


def test_generative_hyde_is_rejected_before_it_can_call_a_model(
    preview_api: PreviewAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeLLMAdapter()
    generate = AsyncMock(side_effect=denied)
    monkeypatch.setattr(fake, "generate", generate)
    preview_api.container.hybrid_retriever._hyde_expander = HyDEExpander(fake)
    response = preview_api.client.post("/retrieve", json={"query": "GraphRAG"})
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "generative_retrieval"
    generate.assert_not_awaited()
    assert preview_api.container.event_log.list_events() == []


def test_post_rerank_capture_is_shared_and_detached(
    preview_api: PreviewAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    chunk = preview_api.add("doc-a", "GraphRAG original evidence.")
    executor = preview_api.container.runner._executor
    final = chunk.model_copy(update={"text": "Exact post-rerank passage."})
    rerank = AsyncMock(
        return_value=[SearchResult(chunk=final, score=0.625, retriever="custom", path=["original"])]
    )
    prepare = AsyncMock(wraps=executor.prepare_context)
    monkeypatch.setattr(executor._reranker, "rerank", rerank)
    monkeypatch.setattr(executor, "prepare_context", prepare)
    preview = preview_api.preview()
    assert preview["sources"][0]["chunk"]["text"] == final.text
    assert preview["sources"][0]["rank"] == 1
    assert preview["sources"][0]["score"] == 0.625
    assert preview["sources"][0]["path"] == ["original"]
    result = preview_api.client.post("/query", json={"query": "What does GraphRAG retrieve?"})
    assert result.json()["result"]["state"] == "DONE"
    assert prepare.await_count == 2
    snapshot = next(
        event["payload"]
        for event in preview_api.container.event_log.list_events()
        if event["event_type"] == "evidence_snapshot"
    )
    assert preview["sources"] == snapshot["sources"]
    captured = EvidenceSnapshot.model_validate(snapshot)
    final.text = "Later mutable corpus contents."
    assert captured.sources[0].chunk.text == "Exact post-rerank passage."
    assert preview["sources"][0]["chunk"]["text"] == "Exact post-rerank passage."


def test_factory_restart_and_independent_apps_keep_preview_read_only(
    preview_api: PreviewAPI, tmp_path: Path
) -> None:
    preview_api.add("doc-a", "GraphRAG local evidence.")
    before = preview_api.preview(document_ids=["doc-a"])
    reopened = create_app(
        offline_settings(preview_api.database_path).model_copy(
            update={"max_source_docs": 2, "max_hops": 1}
        )
    )
    separate = create_app(offline_settings(tmp_path / "separate.sqlite3"))
    with TestClient(reopened) as client, TestClient(separate) as empty:
        response = client.post(
            "/retrieve", json={"query": "What does GraphRAG retrieve?", "document_ids": ["doc-a"]}
        )
        assert response.status_code == 200
        assert response.json() == before
        assert client.get("/runs").json()["runs"] == []
        other = empty.post("/retrieve", json={"query": "GraphRAG", "document_ids": ["doc-a"]})
        assert other.status_code == 200
        assert other.json()["sources"] == []
        assert empty.get("/runs").json()["runs"] == []
    paths = reopened.openapi()["paths"]
    assert {"/retrieve", "/query", "/runs", "/runs/{run_id}/export"} <= set(paths)
    responses = paths["/retrieve"]["post"]["responses"]
    assert {"200", "409", "422", "500", "504"} <= set(responses)


def test_query_legacy_executor_overrides_still_run_once(
    preview_api: PreviewAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    async def retrieve(plan: QueryPlan, max_results: int = 8) -> list[SearchResult]:
        calls.append("retrieve")
        return []

    async def answer(plan: QueryPlan, retrieved: list[SearchResult]) -> AgentAnswer:
        calls.append("answer")
        return AgentAnswer(answer="legacy", citations=[], claims=[])

    executor = preview_api.container.runner._executor
    monkeypatch.setattr(executor, "retrieve", retrieve)
    monkeypatch.setattr(executor, "answer", answer)
    response = preview_api.client.post("/query", json={"query": "GraphRAG"})
    assert response.json()["result"]["state"] == "DONE"
    assert response.json()["result"]["answer"]["answer"] == "legacy"
    assert not any(
        event["event_type"] == "evidence_snapshot"
        for event in preview_api.container.event_log.list_events()
    )
    assert calls == ["retrieve", "answer"]


def test_catalog_selected_ids_feed_preview_without_creating_runs(preview_api: PreviewAPI) -> None:
    preview_api.add("selected", "GraphRAG selected evidence.")
    preview_api.add("excluded", "GraphRAG EXCLUDED_MARKER.")
    catalog = preview_api.client.get("/documents", params={"title": "selected", "limit": 1})
    assert catalog.status_code == 200
    document_id = catalog.json()["documents"][0]["document_id"]
    result = preview_api.preview(document_ids=[document_id])
    assert {source["chunk"]["document_id"] for source in result["sources"]} == {"selected"}
    assert "EXCLUDED_MARKER" not in result["context"]
    assert preview_api.client.get("/runs").json()["runs"] == []


@pytest.mark.parametrize("phase", ["retrieval", "context"])
async def test_cancelled_preview_propagates_without_generation_or_journal(
    preview_api: PreviewAPI, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    started = asyncio.Event()

    async def block(*args: object, **kwargs: object) -> NoReturn:
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("Cancelled preview continued")

    runner = preview_api.container.runner
    target = runner._executor if phase == "retrieval" else runner._executor._reranker
    monkeypatch.setattr(target, "retrieve" if phase == "retrieval" else "rerank", block)
    monkeypatch.setattr(preview_api.container.llm, "generate", denied)
    monkeypatch.setattr(preview_api.container.event_log, "append_event", denied)
    monkeypatch.setattr(preview_api.container.event_log, "append_transition", denied)
    pending = asyncio.create_task(runner.preview("GraphRAG"))
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        pending.cancel("inspection cancelled")
        with pytest.raises(asyncio.CancelledError, match="inspection cancelled"):
            await pending
    finally:
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
    assert preview_api.container.event_log.list_events() == []
