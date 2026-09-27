"""Integrated, offline minimum-document checks against the final captured context."""

import asyncio
import json
import sqlite3
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError
from scripts.demo_evidence_export import offline_settings
from scripts.demo_per_paper_evidence_limits import QUERY, seed_corpus

from agent.comparison_models import RunComparison
from agent.evidence import CaptureLimits, EvidenceBundle, EvidenceSnapshot, GenerationRecord
from agent.models import AgentAnswer, AgentState, QueryObservation, QueryPlan
from agent.retrieval_preview import RetrievalPreviewError
from agent.safety import CancellationToken, with_timeout
from api.application import create_app
from api.dependencies import AppContainer
from api.schemas import QueryRequest
from llm.fake import FakeLLMAdapter
from llm.providers import AnthropicAdapter, GeminiAdapter, KimiAdapter, OpenAIAdapter
from llm.router import RoutingLLMAdapter
from retrieval.evidence_policy import EvidenceAssessment, EvidencePolicy
from retrieval.hyde import HyDEExpander
from retrieval.min_unique_sources_gate import MinUniqueSourcesGate, count_evidence_documents
from retrieval.models import Chunk, Document, SearchResult


def denied(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("This operation must not generate, access HTTP, or write preview events.")


@dataclass
class EvidenceAPI:
    application: FastAPI
    container: AppContainer
    client: TestClient
    database_path: Path

    def post(self, endpoint: str, **options: object) -> dict[str, Any]:
        response = self.client.post(endpoint, json={"query": QUERY, **options})
        assert response.status_code == 200, response.text
        body: dict[str, Any] = response.json()
        return body["result"] if endpoint == "/query" else body


@pytest.fixture
def evidence_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[EvidenceAPI]:
    database_path = tmp_path / "minimum-evidence.sqlite3"
    application = create_app(
        offline_settings(database_path).model_copy(update={"max_source_docs": 6})
    )
    container: AppContainer = application.state.container
    seed_corpus(container)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", denied)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", denied)
    with TestClient(application) as client:
        yield EvidenceAPI(application, container, client, database_path)


def test_insufficient_query_saves_exact_evidence_and_diagnostic_without_generation(
    evidence_api: EvidenceAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = evidence_api.post("/retrieve", document_ids=["paper-a"])
    generate = AsyncMock(wraps=evidence_api.container.llm.generate)
    monkeypatch.setattr(evidence_api.container.llm, "generate", generate)
    result = evidence_api.post("/query", document_ids=["paper-a"], min_evidence_documents=2)
    assert result["state"] == "ERROR", result
    assert result["answer"] is None
    assert "required 2 distinct evidence documents, observed 1" in result["error"]
    generate.assert_not_awaited()
    events = evidence_api.container.event_log.list_events(result["run_id"])
    assert [event["event_type"] for event in events] == [
        "state_transition",
        "decision_log",
        "state_transition",
        "state_transition",
        "evidence_snapshot",
        "state_transition",
    ]
    assert events[4]["payload"]["sources"] == baseline["sources"]
    assert events[4]["payload"]["request"]["context"] == baseline["context"]
    assert events[4]["payload"]["context_sha256"] == baseline["context_sha256"]
    assert events[-1]["payload"]["to_state"] == "ERROR"
    assert events[-1]["payload"]["payload"] == {
        "error": result["error"],
        "code": "insufficient_evidence_documents",
        "evidence_assessment": {
            "required_documents": 2,
            "observed_documents": 1,
            "passed": False,
        },
    }
    assert evidence_api.client.get(f"/runs/{result['run_id']}/export").status_code == 409


@pytest.mark.parametrize(("minimum", "passed"), [(1, True), (3, True), (4, False), (50, False)])
def test_preview_counts_actual_documents_and_preserves_final_context(
    evidence_api: EvidenceAPI,
    monkeypatch: pytest.MonkeyPatch,
    minimum: int,
    passed: bool,
) -> None:
    baseline = evidence_api.post("/retrieve", max_chunks_per_document=1)
    with monkeypatch.context() as patch:
        patch.setattr(evidence_api.container.llm, "generate", denied)
        patch.setattr(evidence_api.container.runner._executor, "answer", denied)
        patch.setattr(evidence_api.container.event_log, "append_event", denied)
        patch.setattr(evidence_api.container.event_log, "append_transition", denied)
        preview = evidence_api.post(
            "/retrieve", max_chunks_per_document=1, min_evidence_documents=minimum
        )
    assert preview["evidence_assessment"] == {
        "required_documents": minimum,
        "observed_documents": 3,
        "passed": passed,
    }
    for key in ("sources", "context", "context_sha256", "configuration", "capture_limits"):
        assert preview[key] == baseline[key]
    assert preview["plan"]["observation"]["evidence_policy"] == {
        "max_chunks_per_document": 1,
        "min_evidence_documents": minimum,
    }
    assert Counter(source["chunk"]["document_id"] for source in preview["sources"]) == {
        "paper-a": 1,
        "paper-b": 1,
        "paper-c": 1,
    }
    assert evidence_api.container.event_log.list_events() == []


@pytest.mark.parametrize("value", [True, False, "2", 1.0, 1.5, 0, -1, 51, None])
def test_invalid_http_minimum_is_rejected_before_work(
    evidence_api: EvidenceAPI, monkeypatch: pytest.MonkeyPatch, value: object
) -> None:
    monkeypatch.setattr(evidence_api.container.runner, "run", denied)
    monkeypatch.setattr(evidence_api.container.runner, "preview", denied)
    for endpoint in ("/query", "/retrieve"):
        response = evidence_api.client.post(
            endpoint, json={"query": QUERY, "min_evidence_documents": value}
        )
        assert response.status_code == 422, response.text
        assert response.json()["detail"][0]["loc"][:2] == ["body", "min_evidence_documents"]
    assert evidence_api.container.event_log.list_events() == []


@pytest.mark.parametrize("value", [True, False, "2", 1.0, 1.5, 0, -1, 51])
async def test_invalid_python_minimum_is_rejected_before_observation(
    evidence_api: EvidenceAPI, monkeypatch: pytest.MonkeyPatch, value: object
) -> None:
    runner = evidence_api.container.runner
    monkeypatch.setattr(runner._analyzer, "analyze", denied)
    for method in (runner.run, runner.preview):
        with pytest.raises(ValidationError):
            await method(QUERY, min_evidence_documents=value)
    assert evidence_api.container.event_log.list_events() == []


@pytest.mark.parametrize("minimum", [1, 3])
def test_sufficient_query_generates_once_from_the_previewed_context(
    evidence_api: EvidenceAPI, monkeypatch: pytest.MonkeyPatch, minimum: int
) -> None:
    executor = evidence_api.container.runner._executor
    rerank = AsyncMock(wraps=executor._reranker.rerank)
    generate = AsyncMock(wraps=evidence_api.container.llm.generate)
    monkeypatch.setattr(executor._reranker, "rerank", rerank)
    monkeypatch.setattr(evidence_api.container.llm, "generate", generate)
    monkeypatch.setattr(MinUniqueSourcesGate, "gate", denied)
    preview = evidence_api.post("/retrieve", min_evidence_documents=minimum)
    result = evidence_api.post("/query", min_evidence_documents=minimum)
    assert result["state"] == "DONE", result
    generate.assert_awaited_once()
    assert rerank.await_count == 2
    bundle = evidence_api.container.evidence_exporter.export(result["run_id"])
    assert bundle.snapshot.model_dump(mode="json")["sources"] == preview["sources"]
    assert bundle.snapshot.request.context == preview["context"]
    assert bundle.snapshot.context_sha256 == preview["context_sha256"]
    assert generate.await_args.args[0] == bundle.snapshot.request
    policy = {"min_evidence_documents": minimum}
    assert result["plan"]["observation"]["evidence_policy"] == policy
    assert result["observation"]["evidence_policy"] == policy
    events = evidence_api.container.event_log.list_events(result["run_id"])
    assert events[0]["payload"]["payload"]["evidence_policy"] == policy
    assert events[1]["payload"]["observation"]["evidence_policy"] == policy
    assert events[2]["payload"]["payload"]["observation"]["evidence_policy"] == policy
    assert len(events) == 8
    assert events[4]["event_type"] == "evidence_snapshot"
    assert events[5]["event_type"] == "generation_record"
    assert {source.retriever for source in bundle.snapshot.sources} == {"lexical_rerank"}
    markdown = evidence_api.client.get(f"/runs/{result['run_id']}/export?format=markdown")
    assert f"min_evidence_documents={minimum}" in markdown.text
    assert "observed_documents=3; passed=True" in markdown.text
    assert "No per-document quota requested" in markdown.text
    assert "max_chunks_per_document=None" not in markdown.text


def test_schema_policy_and_assessment_are_strict_frozen_and_backward_compatible(
    evidence_api: EvidenceAPI,
) -> None:
    schemas = evidence_api.client.get("/openapi.json").json()["components"]["schemas"]
    for name in ("QueryRequest", "RetrievalRequest"):
        field = schemas[name]["properties"]["min_evidence_documents"]
        assert field["type"] == "integer"
        assert field["minimum"] == 1 and field["maximum"] == 50
        assert "min_evidence_documents" not in schemas[name]["required"]
    request = QueryRequest(query=QUERY, min_evidence_documents=2)
    with pytest.raises(ValidationError, match="frozen"):
        request.min_evidence_documents = 1
    assert QueryRequest(query=QUERY).min_evidence_documents is None
    policy = EvidencePolicy(min_evidence_documents=2)
    with pytest.raises(ValidationError, match="frozen"):
        policy.min_evidence_documents = 1
    observation = QueryObservation.model_validate(
        evidence_api.post("/retrieve", min_evidence_documents=2)["plan"]["observation"]
    )
    with pytest.raises(ValidationError, match="frozen"):
        observation.evidence_policy = EvidencePolicy(min_evidence_documents=1)
    old = EvidencePolicy.model_validate({"max_chunks_per_document": 1})
    assert old.min_evidence_documents is None
    assert old.model_dump() == {"max_chunks_per_document": 1}
    assert policy.model_dump() == {"min_evidence_documents": 2}
    with pytest.raises(ValidationError, match="at least one"):
        EvidencePolicy()
    with pytest.raises(ValidationError, match="document counts"):
        EvidenceAssessment(required_documents=2, observed_documents=1, passed=True)
    assessment = EvidenceAssessment(required_documents=2, observed_documents=1, passed=False)
    with pytest.raises(ValidationError, match="frozen"):
        assessment.passed = True


@pytest.mark.parametrize("scope", [["paper-a"], ["paper-a", "unknown"], ["unknown"]])
def test_only_actual_document_ids_count_not_selected_ids_chunks_titles_or_source_labels(
    evidence_api: EvidenceAPI, monkeypatch: pytest.MonkeyPatch, scope: list[str]
) -> None:
    monkeypatch.setattr(evidence_api.container.llm, "generate", denied)
    preview = evidence_api.post("/retrieve", document_ids=scope, min_evidence_documents=2)
    observed = int("paper-a" in scope)
    assert preview["evidence_assessment"]["observed_documents"] == observed
    assert not preview["evidence_assessment"]["passed"]
    assert len(preview["sources"]) == 3 * observed
    result = evidence_api.post("/query", document_ids=scope, min_evidence_documents=2)
    assert result["state"] == "ERROR", result
    assert result["plan"]["observation"]["document_ids"] == scope
    events = evidence_api.container.event_log.list_events(result["run_id"])
    assert events[-1]["payload"]["payload"]["evidence_assessment"] == preview["evidence_assessment"]
    assert not any(event["event_type"] == "generation_record" for event in events)


def test_empty_corpus_minimum_is_inspectable_and_prevents_every_provider_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    settings = offline_settings(tmp_path / "empty.sqlite3").model_copy(
        update={
            "openai_api_key": "synthetic-unused-openai",
            "anthropic_api_key": "synthetic-unused-anthropic",
            "gemini_api_key": "synthetic-unused-gemini",
            "moonshot_api_key": "synthetic-unused-kimi",
        }
    )
    application = create_app(settings)
    with TestClient(application) as client:
        payload = {"query": QUERY, "min_evidence_documents": 1}
        preview = client.post("/retrieve", json=payload)
        assert preview.status_code == 200
        assert preview.json()["sources"] == []
        assert preview.json()["context"] == ""
        assert preview.json()["evidence_assessment"] == {
            "required_documents": 1,
            "observed_documents": 0,
            "passed": False,
        }
        assert application.state.container.event_log.list_events() == []
        result = client.post("/query", json=payload).json()["result"]
        assert result["state"] == "ERROR"
        assert "observed 0" in result["error"]
        assert "synthetic-unused" not in json.dumps(result)
    for generate in calls:
        generate.assert_not_awaited()


def test_maximum_minimum_accepts_exactly_fifty_actual_documents(tmp_path: Path) -> None:
    application = create_app(
        offline_settings(tmp_path / "fifty.sqlite3").model_copy(update={"max_source_docs": 50})
    )
    container: AppContainer = application.state.container
    chunks = [
        Chunk(
            chunk_id=f"chunk-{i}",
            document_id=f"doc-{i}",
            title="Same synthetic title",
            text="retrieval evidence for a synthetic count boundary.",
            source="synthetic:same",
        )
        for i in range(50)
    ]
    container.document_store.add_documents(
        [Document(**chunk.model_dump(exclude={"chunk_id"})) for chunk in chunks], chunks
    )
    container.hybrid_retriever.add_chunks(chunks)
    with TestClient(application) as client:
        payload = {"query": QUERY, "min_evidence_documents": 50, "max_chunks_per_document": 1}
        preview = client.post("/retrieve", json=payload)
        assert preview.status_code == 200, preview.text
        assert preview.json()["evidence_assessment"] == {
            "required_documents": 50,
            "observed_documents": 50,
            "passed": True,
        }
        result = client.post("/query", json=payload).json()["result"]
        assert result["state"] == "DONE", result
        bundle = container.evidence_exporter.export(result["run_id"])
        assert len(bundle.snapshot.sources) == 50
        assert len({source.chunk.document_id for source in bundle.snapshot.sources}) == 50


def test_count_runs_after_reranking_quota_and_capture_not_against_raw_hits(
    evidence_api: EvidenceAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor = evidence_api.container.runner._executor
    raw_counts = []

    async def only_one_paper(query: str, retrieved: list[SearchResult]) -> list[SearchResult]:
        raw_counts.append(count_evidence_documents(retrieved))
        return [result for result in retrieved if result.chunk.document_id == "paper-a"]

    monkeypatch.setattr(executor._reranker, "rerank", only_one_paper)
    monkeypatch.setattr(evidence_api.container.llm, "generate", denied)
    baseline = evidence_api.post("/retrieve", max_chunks_per_document=1)
    preview = evidence_api.post("/retrieve", max_chunks_per_document=1, min_evidence_documents=2)
    result = evidence_api.post("/query", max_chunks_per_document=1, min_evidence_documents=2)
    assert raw_counts == [3, 3, 3]
    assert result["state"] == "ERROR"
    assert preview["evidence_assessment"]["observed_documents"] == 1
    assert len(preview["sources"]) == 1
    assert preview["sources"] == baseline["sources"]
    assert preview["sources"][0]["retriever"] == "diversity_cap_gate"
    events = evidence_api.container.event_log.list_events(result["run_id"])
    assert events[4]["payload"]["sources"] == preview["sources"]


def test_source_limit_is_not_increased_to_satisfy_a_minimum(
    evidence_api: EvidenceAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = evidence_api.container.runner
    runner._safety_limits.max_source_docs = 1
    retrieve = AsyncMock(wraps=runner._executor.retrieve)
    monkeypatch.setattr(runner._executor, "retrieve", retrieve)
    monkeypatch.setattr(evidence_api.container.llm, "generate", denied)
    preview = evidence_api.post("/retrieve", min_evidence_documents=50)
    result = evidence_api.post("/query", min_evidence_documents=50)
    assert preview["configuration"]["max_source_docs"] == 1
    assert preview["evidence_assessment"]["observed_documents"] == 1
    assert result["state"] == "ERROR" and "required 50" in result["error"]
    assert retrieve.await_count == 2
    assert all(call.args[1] == 1 for call in retrieve.await_args_list)
    assert runner._safety_limits.max_source_docs == 1


def test_collection_resolution_and_empty_unknown_scope_never_widen_for_a_minimum(
    evidence_api: EvidenceAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved = evidence_api.client.post(
        "/collections", json={"name": "Synthetic selection", "document_ids": ["paper-a", "paper-b"]}
    ).json()
    options = {
        "collection_id": saved["collection_id"],
        "max_chunks_per_document": 1,
        "min_evidence_documents": 2,
    }
    preview = evidence_api.post("/retrieve", **options)
    result = evidence_api.post("/query", **options)
    assert result["state"] == "DONE"
    assert preview["evidence_assessment"]["passed"]
    assert preview["plan"]["observation"]["document_ids"] == ["paper-a", "paper-b"]
    assert {source["chunk"]["document_id"] for source in preview["sources"]} == {
        "paper-a",
        "paper-b",
    }
    assert "paper-c" not in json.dumps(preview)
    with sqlite3.connect(evidence_api.database_path) as connection:
        connection.execute("DELETE FROM documents WHERE document_id = ?", ("paper-b",))
    before = evidence_api.container.event_log.list_events()
    monkeypatch.setattr(evidence_api.container.runner, "run", denied)
    monkeypatch.setattr(evidence_api.container.runner, "preview", denied)
    for endpoint in ("/query", "/retrieve"):
        for scope, status in (
            ({"collection_id": saved["collection_id"]}, 409),
            ({"collection_id": "col_" + "0" * 32}, 404),
            ({"document_ids": []}, 422),
            ({"document_ids": None}, 422),
            ({"document_ids": ["paper-a"], "collection_id": saved["collection_id"]}, 422),
        ):
            response = evidence_api.client.post(
                endpoint, json={"query": QUERY, "min_evidence_documents": 1, **scope}
            )
            assert response.status_code == status, response.text
    assert evidence_api.container.event_log.list_events() == before


@pytest.mark.parametrize("stage", ["answer", "prepare_context", "keyword_answer", "missing_hook"])
def test_unsupported_custom_executors_fail_explicitly_before_answer_generation(
    evidence_api: EvidenceAPI, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    executor = evidence_api.container.runner._executor
    calls = []
    original_answer, original_prepare = executor.answer, executor.prepare_context

    async def old_answer(plan: QueryPlan, retrieved: list[SearchResult]) -> AgentAnswer:
        calls.append("answer")
        return await original_answer(plan, retrieved)

    async def old_prepare(plan: QueryPlan, retrieved: list[SearchResult]) -> EvidenceSnapshot:
        calls.append("prepare")
        return await original_prepare(plan, retrieved)

    async def ignores_keywords(*args: object, **kwargs: object) -> AgentAnswer:
        calls.append("ignored")
        return AgentAnswer(answer="ignored requested policy", citations=[], claims=[])

    if stage == "answer":
        monkeypatch.setattr(executor, "answer", old_answer)
    elif stage == "prepare_context":
        monkeypatch.setattr(executor, "prepare_context", old_prepare)
    elif stage == "keyword_answer":
        monkeypatch.setattr(executor, "answer", ignores_keywords)
    else:
        monkeypatch.setattr(executor, "answer_prepared", None)
    if stage != "missing_hook":
        assert evidence_api.post("/query")["state"] == "DONE"
    before = len(calls)
    monkeypatch.setattr(evidence_api.container.llm, "generate", denied)
    result = evidence_api.post("/query", min_evidence_documents=1)
    assert result["state"] == "ERROR", result
    assert ("evidence_policy" if stage == "prepare_context" else "answer_prepared") in result[
        "error"
    ]
    assert len(calls) == before
    events = evidence_api.container.event_log.list_events(result["run_id"])
    assert not any(event["event_type"] == "generation_record" for event in events)
    if stage == "prepare_context":
        before_events = evidence_api.container.event_log.list_events()
        response = evidence_api.client.post(
            "/retrieve", json={"query": QUERY, "min_evidence_documents": 1}
        )
        assert response.status_code == 500
        assert response.json()["detail"]["code"] == "context_preparation_failed"
        assert evidence_api.container.event_log.list_events() == before_events


async def test_explicit_prepared_answer_hook_only_receives_detached_sufficient_context(
    evidence_api: EvidenceAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = evidence_api.container.runner
    executor = runner._executor
    original_prepare, original_answer = executor.prepare_context, executor.answer_prepared
    captured = []
    entered, release = asyncio.Event(), asyncio.Event()

    async def prepare(
        plan: QueryPlan, retrieved: list[SearchResult], *, evidence_policy: EvidencePolicy
    ) -> EvidenceSnapshot:
        snapshot = await original_prepare(plan, retrieved, evidence_policy=evidence_policy)
        captured.append(snapshot)
        return snapshot

    async def answer_prepared(
        plan: QueryPlan,
        snapshot: EvidenceSnapshot,
        *,
        on_generation: Callable[[GenerationRecord], None],
    ) -> AgentAnswer:
        entered.set()
        await release.wait()
        assert len({source.chunk.document_id for source in snapshot.sources}) == 3
        return await original_answer(plan, snapshot, on_generation=on_generation)

    monkeypatch.setattr(executor, "prepare_context", prepare)
    monkeypatch.setattr(executor, "answer", denied)
    hook = AsyncMock(side_effect=answer_prepared)
    monkeypatch.setattr(executor, "answer_prepared", hook)
    failed = await runner.run(QUERY, min_evidence_documents=4)
    assert failed.state == AgentState.ERROR
    hook.assert_not_awaited()
    pending = asyncio.create_task(runner.run(QUERY, min_evidence_documents=3))
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        for source in captured[-1].sources:
            source.chunk.document_id = "later-mutated"
            source.chunk.text = "later-mutated"
        captured[-1].request.context = "later-mutated"
    finally:
        release.set()
    result = await asyncio.wait_for(pending, timeout=2)
    assert result.state == AgentState.DONE, result.error
    hook.assert_awaited_once()
    bundle = evidence_api.container.evidence_exporter.export(result.run_id)
    assert len({source.chunk.document_id for source in bundle.snapshot.sources}) == 3
    assert "later-mutated" not in bundle.snapshot.request.context


def test_prepared_hook_must_record_generation(
    evidence_api: EvidenceAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    hook = AsyncMock(return_value=AgentAnswer(answer="missing record", citations=[], claims=[]))
    monkeypatch.setattr(evidence_api.container.runner._executor, "answer_prepared", hook)
    result = evidence_api.post("/query", min_evidence_documents=1)
    assert result["state"] == "ERROR"
    assert "did not record generation" in result["error"]
    assert result["answer"] is None
    hook.assert_awaited_once()


def test_opt_in_generative_retrieval_is_rejected_without_algorithm_substitution(
    evidence_api: EvidenceAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeLLMAdapter()
    generate = AsyncMock(side_effect=denied)
    monkeypatch.setattr(fake, "generate", generate)
    expander = HyDEExpander(fake)
    evidence_api.container.hybrid_retriever._hyde_expander = expander
    monkeypatch.setattr(evidence_api.container.llm, "generate", denied)
    result = evidence_api.post("/query", min_evidence_documents=2)
    assert result["state"] == "ERROR"
    assert "requires non-generative retrieval" in result["error"]
    before = evidence_api.container.event_log.list_events()
    response = evidence_api.client.post(
        "/retrieve", json={"query": QUERY, "min_evidence_documents": 2}
    )
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "generative_retrieval"
    assert evidence_api.container.event_log.list_events() == before
    assert evidence_api.container.hybrid_retriever._hyde_expander is expander
    generate.assert_not_awaited()


@pytest.mark.parametrize("phase", ["planning", "retrieval", "context_preparation"])
def test_custom_components_cannot_drop_the_frozen_minimum(
    evidence_api: EvidenceAPI, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    runner = evidence_api.container.runner
    plan_method = runner._planner.plan
    retrieve, prepare = runner._executor.retrieve, runner._executor.prepare_context

    def drop(plan: QueryPlan) -> QueryPlan:
        plan.observation = plan.observation.model_copy(
            update={"evidence_policy": EvidencePolicy(max_chunks_per_document=1)}
        )
        return plan

    def dropping_plan(run_id: str, observation: QueryObservation) -> QueryPlan:
        return drop(plan_method(run_id, observation))

    async def dropping_retrieve(plan: QueryPlan, max_results: int = 8) -> list[SearchResult]:
        retrieved = await retrieve(plan, max_results)
        drop(plan)
        return retrieved

    async def dropping_prepare(
        plan: QueryPlan, retrieved: list[SearchResult], *, evidence_policy: EvidencePolicy
    ) -> EvidenceSnapshot:
        snapshot = await prepare(plan, retrieved, evidence_policy=evidence_policy)
        drop(plan)
        return snapshot

    if phase == "planning":
        monkeypatch.setattr(runner._planner, "plan", dropping_plan)
    elif phase == "retrieval":
        monkeypatch.setattr(runner._executor, "retrieve", dropping_retrieve)
    else:
        monkeypatch.setattr(runner._executor, "prepare_context", dropping_prepare)
    monkeypatch.setattr(evidence_api.container.llm, "generate", denied)
    options = {"min_evidence_documents": 2, "max_chunks_per_document": 1}
    result = evidence_api.post("/query", **options)
    assert result["state"] == "ERROR"
    assert "does not match min_evidence_documents" in result["error"]
    events = evidence_api.container.event_log.list_events(result["run_id"])
    assert events[0]["payload"]["payload"]["evidence_policy"] == options
    response = evidence_api.client.post("/retrieve", json={"query": QUERY, **options})
    assert response.status_code == 500
    # Preview revalidates the plan after preparation, even if retrieval changed it.
    expected_phase = "context_preparation" if phase == "retrieval" else phase
    assert response.json()["detail"]["code"] == f"{expected_phase}_failed"


@pytest.mark.parametrize(
    "damage", ["effective_sources", "capture_sources", "context_bytes", "metadata_bytes", "digest"]
)
def test_minimum_does_not_relax_any_capture_validation(
    evidence_api: EvidenceAPI, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    executor = evidence_api.container.runner._executor
    original_prepare = executor.prepare_context
    chunk = evidence_api.container.document_store.list_chunks()[0]
    if damage == "digest":

        async def corrupt(
            plan: QueryPlan, retrieved: list[SearchResult], *, evidence_policy: EvidencePolicy
        ) -> EvidenceSnapshot:
            snapshot = await original_prepare(plan, retrieved, evidence_policy=evidence_policy)
            return snapshot.model_copy(update={"context_sha256": "0" * 64})

        monkeypatch.setattr(executor, "prepare_context", corrupt)
    else:
        count = 7 if damage == "effective_sources" else 51 if damage == "capture_sources" else 1
        if damage == "context_bytes":
            chunk.text = "x" * CaptureLimits().max_context_bytes
        elif damage == "metadata_bytes":
            chunk.metadata["oversized"] = "x" * CaptureLimits().max_snapshot_bytes
        results = [
            SearchResult(
                chunk=chunk.model_copy(update={"chunk_id": f"overflow-{i}"}),
                score=1,
                retriever="synthetic",
            )
            for i in range(count)
        ]
        monkeypatch.setattr(executor._reranker, "rerank", AsyncMock(return_value=results))
    monkeypatch.setattr(evidence_api.container.llm, "generate", denied)
    result = evidence_api.post("/query", min_evidence_documents=1)
    assert result["state"] == "ERROR"
    events = evidence_api.container.event_log.list_events(result["run_id"])
    assert not any(event["event_type"] == "evidence_snapshot" for event in events)
    response = evidence_api.client.post(
        "/retrieve", json={"query": QUERY, "min_evidence_documents": 1}
    )
    assert response.status_code == 500
    assert response.json()["detail"]["code"] == "context_preparation_failed"
    assert "sources" not in response.json()


@pytest.mark.parametrize("phase", ["retrieval", "context", "generation"])
async def test_guard_keeps_external_cancellation_and_phase_deadlines(
    evidence_api: EvidenceAPI, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    runner = evidence_api.container.runner
    entered = asyncio.Event()
    target = evidence_api.container.llm if phase == "generation" else runner._executor
    method = {"retrieval": "retrieve", "context": "prepare_context", "generation": "generate"}[
        phase
    ]

    async def block(*args: object, **kwargs: object) -> NoReturn:
        entered.set()
        await asyncio.Event().wait()
        raise AssertionError("Blocked phase resumed")

    blocked = AsyncMock(side_effect=block)
    monkeypatch.setattr(target, method, blocked)
    if phase != "generation":
        monkeypatch.setattr(evidence_api.container.llm, "generate", denied)
    pending = asyncio.create_task(runner.run(QUERY, min_evidence_documents=1))
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        pending.cancel("guard cancelled")
        with pytest.raises(asyncio.CancelledError, match="guard cancelled"):
            await pending
    finally:
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
    events = evidence_api.container.event_log.list_events()
    assert events[-1]["payload"]["to_state"] == "ERROR"
    assert events[-1]["payload"]["payload"] == {"error": "agent run was cancelled: guard cancelled"}
    assert sum(event["event_type"] == "evidence_snapshot" for event in events) == int(
        phase == "generation"
    )
    assert not any(event["event_type"] == "generation_record" for event in events)
    blocked.assert_awaited_once()

    runner._safety_limits.retrieval_timeout_seconds = 0.01
    runner._safety_limits.reasoning_timeout_seconds = 0.01
    result = await asyncio.wait_for(runner.run(QUERY, min_evidence_documents=1), timeout=2)
    label = "retrieval" if phase == "retrieval" else "reasoning"
    assert result.state == AgentState.ERROR
    assert result.error == f"{label} timed out after 0.0s"
    assert blocked.await_count == 2


async def test_preparation_consumes_the_same_reasoning_budget_as_generation(
    evidence_api: EvidenceAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = evidence_api.container.runner
    prepare = runner._executor.prepare_context
    runner._safety_limits.reasoning_timeout_seconds = 0.2
    timed = AsyncMock(wraps=with_timeout)
    cancelled_generation = []

    async def slow_prepare(
        plan: QueryPlan, retrieved: list[SearchResult], *, evidence_policy: EvidencePolicy
    ) -> EvidenceSnapshot:
        await asyncio.sleep(0.12)
        return await prepare(plan, retrieved, evidence_policy=evidence_policy)

    async def slow_generate(*args: object, **kwargs: object) -> NoReturn:
        try:
            await asyncio.sleep(0.12)
        except asyncio.CancelledError:
            cancelled_generation.append(True)
            raise
        raise AssertionError("Generation incorrectly received its own full reasoning budget")

    generate = AsyncMock(side_effect=slow_generate)
    monkeypatch.setattr(runner._executor, "prepare_context", slow_prepare)
    monkeypatch.setattr(evidence_api.container.llm, "generate", generate)
    monkeypatch.setattr("agent.runner.with_timeout", timed)
    result = await runner.run(QUERY, min_evidence_documents=1)
    assert result.state == AgentState.ERROR
    assert result.error == "reasoning timed out after 0.2s"
    assert [(call.args[2], call.args[1]) for call in timed.await_args_list] == [
        ("retrieval", 30),
        ("reasoning", 0.2),
    ]
    generate.assert_awaited_once()
    assert cancelled_generation == [True]
    events = evidence_api.container.event_log.list_events(result.run_id)
    assert events[-2]["event_type"] == "evidence_snapshot"
    assert events[-1]["payload"]["payload"] == {"error": result.error}


@pytest.mark.parametrize("phase", ["retrieval", "context"])
async def test_guarded_preview_cancellation_and_timeout_never_journal(
    evidence_api: EvidenceAPI, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    runner = evidence_api.container.runner
    entered = asyncio.Event()

    async def block(*args: object, **kwargs: object) -> NoReturn:
        entered.set()
        await asyncio.Event().wait()
        raise AssertionError("Blocked preview resumed")

    monkeypatch.setattr(
        runner._executor, "retrieve" if phase == "retrieval" else "prepare_context", block
    )
    monkeypatch.setattr(evidence_api.container.llm, "generate", denied)
    monkeypatch.setattr(evidence_api.container.event_log, "append_event", denied)
    pending = asyncio.create_task(runner.preview(QUERY, min_evidence_documents=1))
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        pending.cancel("preview cancelled")
        with pytest.raises(asyncio.CancelledError, match="preview cancelled"):
            await pending
    finally:
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
    runner._safety_limits.retrieval_timeout_seconds = 0.01
    runner._safety_limits.reasoning_timeout_seconds = 0.01
    with pytest.raises(RetrievalPreviewError) as failure:
        await runner.preview(QUERY, min_evidence_documents=1)
    assert failure.value.status_code == 504
    assert failure.value.code == (
        "retrieval_timeout" if phase == "retrieval" else "context_preparation_timeout"
    )
    assert evidence_api.container.event_log.list_events() == []


async def test_cooperative_cancellation_before_and_during_preparation_stops_generation(
    evidence_api: EvidenceAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = evidence_api.container.runner
    token = CancellationToken()
    token.cancel()
    monkeypatch.setattr(evidence_api.container.llm, "generate", denied)
    cancelled = await runner.run(QUERY, token, min_evidence_documents=1)
    assert cancelled.error == "agent run was cancelled"
    assert len(evidence_api.container.event_log.list_events(cancelled.run_id)) == 1
    token = CancellationToken()
    prepare = runner._executor.prepare_context

    async def cancelling(
        plan: QueryPlan, retrieved: list[SearchResult], *, evidence_policy: EvidencePolicy
    ) -> EvidenceSnapshot:
        snapshot = await prepare(plan, retrieved, evidence_policy=evidence_policy)
        token.cancel()
        return snapshot

    monkeypatch.setattr(runner._executor, "prepare_context", cancelling)
    result = await runner.run(QUERY, token, min_evidence_documents=1)
    assert result.state == AgentState.ERROR and result.error == "agent run was cancelled"
    events = evidence_api.container.event_log.list_events(result.run_id)
    assert events[-2]["event_type"] == "evidence_snapshot"
    assert events[-1]["payload"]["payload"] == {"error": result.error}


async def test_minimum_scope_and_limits_are_frozen_before_awaits_and_isolated_between_calls(
    evidence_api: EvidenceAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = evidence_api.container.runner
    shared = runner._analyzer.analyze(QUERY)
    monkeypatch.setattr(runner._analyzer, "analyze", Mock(return_value=shared))
    retrieve = runner._executor.retrieve
    entered, release = asyncio.Event(), asyncio.Event()

    async def held(
        plan: QueryPlan, max_results: int = 8, *, document_ids: tuple[str, ...] | None = None
    ) -> list[SearchResult]:
        if plan.observation.evidence_policy == EvidencePolicy(min_evidence_documents=2):
            entered.set()
            await release.wait()
            assert max_results == 6
            assert document_ids == ("paper-a",)
        return await retrieve(plan, max_results, document_ids=document_ids)

    monkeypatch.setattr(runner._executor, "retrieve", held)
    supplied = ["paper-a"]
    pending = asyncio.create_task(
        runner.run(QUERY, document_ids=supplied, min_evidence_documents=2)
    )
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        supplied[:] = ["paper-b", "paper-c"]
        preview, default, explicit_none = await asyncio.gather(
            runner.preview(QUERY, min_evidence_documents=3, max_chunks_per_document=1),
            runner.run(QUERY),
            runner.preview(QUERY, min_evidence_documents=None),
        )
        runner._safety_limits.max_source_docs = 1
        runner._safety_limits.reasoning_timeout_seconds = 0.00001
        runner._safety_limits.retrieval_timeout_seconds = 0.00001
        assert not pending.done()
    finally:
        release.set()
    failed = await asyncio.wait_for(pending, timeout=2)
    assert failed.state == AgentState.ERROR and "observed 1" in (failed.error or "")
    assert failed.plan is not None and failed.plan.observation.document_ids == ("paper-a",)
    assert default.state == AgentState.DONE
    assert shared.evidence_policy is None and shared.document_ids is None
    assert preview.evidence_assessment is not None and preview.evidence_assessment.passed
    assert explicit_none.evidence_assessment is None
    assert "evidence_assessment" not in explicit_none.model_dump()
    assert explicit_none.plan.observation.evidence_policy is None
    events = evidence_api.container.event_log.list_events(failed.run_id)
    config = events[0]["payload"]["payload"]["configuration"]
    assert config["max_source_docs"] == 6
    assert config["reasoning_timeout_seconds"] == 60
    assert config["retrieval_timeout_seconds"] == 30
    assert len(events[-2]["payload"]["sources"]) == 3


def test_minimum_policy_only_changes_survive_export_comparison_and_restart(
    evidence_api: EvidenceAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = evidence_api.post("/query", min_evidence_documents=2)
    candidate = evidence_api.post("/query", min_evidence_documents=3)
    failed = evidence_api.post("/query", min_evidence_documents=4)
    before_events = evidence_api.container.event_log.list_events()
    export_path = f"/runs/{baseline['run_id']}/export"
    comparison_path = f"/runs/{baseline['run_id']}/compare/{candidate['run_id']}"
    paths = (
        export_path,
        export_path + "?format=markdown",
        comparison_path,
        f"/runs/{failed['run_id']}/events",
        "/runs",
    )
    saved = {path: evidence_api.client.get(path).content for path in paths}
    bundle = EvidenceBundle.model_validate_json(saved[export_path])
    comparison = RunComparison.model_validate_json(saved[comparison_path])
    assert comparison.baseline.evidence_policy == EvidencePolicy(min_evidence_documents=2)
    assert comparison.candidate.evidence_policy == EvidencePolicy(min_evidence_documents=3)
    assert comparison.any_changes
    assert [name for name, changed in comparison.changes.model_dump().items() if changed] == [
        "evidence_policy_changed"
    ]
    assert comparison.baseline.context_sha256 == comparison.candidate.context_sha256
    assert any("policies differ" in notice for notice in comparison.notices)
    assert bundle.plan.observation.evidence_policy == EvidencePolicy(min_evidence_documents=2)
    with sqlite3.connect(evidence_api.database_path) as connection:
        connection.execute("DELETE FROM chunks")
        connection.execute("DELETE FROM documents")
    reopened = AppContainer(offline_settings(evidence_api.database_path))
    evidence_api.application.state.container = reopened
    monkeypatch.setattr(reopened.llm, "generate", denied)
    monkeypatch.setattr(reopened.runner._executor, "retrieve", denied)
    monkeypatch.setattr(reopened.document_store, "list_chunks", denied)
    monkeypatch.setattr(reopened.event_log, "append_event", denied)
    for path, content in saved.items():
        response = evidence_api.client.get(path)
        assert response.status_code == 200, response.text
        assert response.content == content
    assert reopened.event_log.list_events() == before_events
    failed_export = evidence_api.client.get(f"/runs/{failed['run_id']}/export")
    assert failed_export.status_code == 409
    assert failed_export.json()["detail"]["code"] == "run_failed"
    failed_comparison = evidence_api.client.get(
        f"/runs/{baseline['run_id']}/compare/{failed['run_id']}"
    )
    assert failed_comparison.status_code == 409
    assert failed_comparison.json()["detail"]["side"] == "candidate"


@pytest.mark.parametrize("damage", ["initial", "plan", "boolean", "counts", "false_success"])
def test_saved_minimum_provenance_and_success_are_revalidated(
    evidence_api: EvidenceAPI, damage: str
) -> None:
    result = evidence_api.post("/query", min_evidence_documents=2)
    events = evidence_api.container.event_log.list_events(result["run_id"])
    if damage == "initial":
        events[0]["payload"]["payload"]["evidence_policy"] = None
    elif damage == "plan":
        events[1]["payload"]["observation"]["evidence_policy"]["min_evidence_documents"] = 1
    elif damage == "boolean":
        events[0]["payload"]["payload"]["evidence_policy"]["min_evidence_documents"] = True
    elif damage == "counts":
        for source in events[4]["payload"]["sources"]:
            source["chunk"]["document_id"] = "same-document"
    else:
        for target in (
            events[0]["payload"]["payload"],
            events[1]["payload"]["observation"],
            events[2]["payload"]["payload"]["observation"],
        ):
            target["evidence_policy"]["min_evidence_documents"] = 4
    with sqlite3.connect(evidence_api.database_path) as connection:
        for event in events:
            connection.execute(
                "UPDATE agent_events SET payload = ? WHERE id = ?",
                (json.dumps(event["payload"]), event["id"]),
            )
    for path in (
        f"/runs/{result['run_id']}/export",
        f"/runs/{result['run_id']}/export?format=markdown",
        f"/runs/{result['run_id']}/compare/{result['run_id']}",
    ):
        response = evidence_api.client.get(path)
        assert response.status_code == 409, response.text
        assert response.json()["detail"]["code"] == "invalid_run_record"


def test_omitted_minimum_keeps_legacy_runner_signatures_and_bundle_shapes(
    evidence_api: EvidenceAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = evidence_api.container.runner
    original_run, original_preview = runner.run, runner.preview

    async def legacy_run(query: str) -> object:
        return await original_run(query)

    async def legacy_preview(query: str) -> object:
        return await original_preview(query)

    monkeypatch.setattr(runner, "run", legacy_run)
    monkeypatch.setattr(runner, "preview", legacy_preview)
    preview = evidence_api.post("/retrieve")
    assert "evidence_assessment" not in preview
    result = evidence_api.post("/query")
    assert result["state"] == "DONE"
    assert result["plan"]["observation"]["evidence_policy"] is None
    exported = evidence_api.client.get(f"/runs/{result['run_id']}/export").json()
    exported["plan"]["observation"].pop("evidence_policy")
    legacy = EvidenceBundle.model_validate(exported)
    assert legacy.plan.observation.evidence_policy is None


@pytest.mark.parametrize("explicit_none", [False, True])
async def test_unconfigured_empty_evidence_still_generates_as_before(
    evidence_api: EvidenceAPI, monkeypatch: pytest.MonkeyPatch, explicit_none: bool
) -> None:
    generate = AsyncMock(wraps=evidence_api.container.llm.generate)
    monkeypatch.setattr(evidence_api.container.llm, "generate", generate)
    runner = evidence_api.container.runner
    if explicit_none:
        result = await runner.run(QUERY, document_ids=["unknown"], min_evidence_documents=None)
    else:
        result = await runner.run(QUERY, document_ids=["unknown"])
    assert result.state == AgentState.DONE, result.error
    generate.assert_awaited_once()
    assert generate.await_args.args[0].context == ""
    bundle = evidence_api.container.evidence_exporter.export(result.run_id)
    assert bundle.snapshot.sources == []
    assert bundle.plan.observation.evidence_policy is None


@pytest.mark.parametrize("phase", ["retrieval", "reranking"])
def test_minimum_never_hides_an_out_of_scope_source(
    evidence_api: EvidenceAPI, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    executor = evidence_api.container.runner._executor
    excluded = next(
        chunk
        for chunk in evidence_api.container.document_store.list_chunks()
        if chunk.document_id == "paper-c"
    )
    leaked = AsyncMock(return_value=[SearchResult(chunk=excluded, score=1, retriever="leak")])
    if phase == "retrieval":
        monkeypatch.setattr(executor, "retrieve", leaked)
    else:
        monkeypatch.setattr(executor._reranker, "rerank", leaked)
    monkeypatch.setattr(evidence_api.container.llm, "generate", denied)
    payload = {
        "query": QUERY,
        "document_ids": ["paper-a", "paper-b"],
        "max_chunks_per_document": 1,
        "min_evidence_documents": 1,
    }
    result = evidence_api.client.post("/query", json=payload).json()["result"]
    assert result["state"] == "ERROR"
    assert "outside document_ids" in result["error"]
    events = evidence_api.container.event_log.list_events(result["run_id"])
    assert not any(event["event_type"] == "evidence_snapshot" for event in events)
    response = evidence_api.client.post("/retrieve", json=payload)
    assert response.status_code == 500
    expected = "retrieval_failed" if phase == "retrieval" else "context_preparation_failed"
    assert response.json()["detail"]["code"] == expected
