"""Opt-in lexical evidence collapse through real offline query and preview routes."""

import asyncio
import json
import sqlite3
from collections.abc import Iterator
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
from scripts.demo_near_duplicate_evidence import QUERY, seed_corpus

from agent.comparison_models import RunComparison
from agent.evidence import EvidenceBundle, EvidenceSnapshot
from agent.models import AgentAnswer, AgentState, QueryObservation, QueryPlan
from api.application import create_app
from api.dependencies import AppContainer
from api.schemas import QueryRequest
from retrieval.diversity_cap_gate import DiversityCapGate
from retrieval.evidence_policy import EvidencePolicy
from retrieval.models import SearchResult
from retrieval.near_duplicate_collapse import NearDuplicateCollapser


def denied(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("Unexpected model, network, or preview event operation.")


@dataclass
class CollapseAPI:
    application: FastAPI
    container: AppContainer
    client: TestClient
    database_path: Path

    def post(self, endpoint: str, **options: object) -> dict[str, Any]:
        response = self.client.post(endpoint, json={"query": QUERY, **options})
        assert response.status_code == 200, response.text
        body: dict[str, Any] = response.json()
        return body["result"] if endpoint == "/query" else body

    def bundle(self, **options: object) -> EvidenceBundle:
        result = self.post("/query", **options)
        assert result["state"] == "DONE", result
        return self.container.evidence_exporter.export(result["run_id"])


@pytest.fixture
def collapse_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[CollapseAPI]:
    database_path = tmp_path / "collapse.sqlite3"
    application = create_app(
        offline_settings(database_path).model_copy(update={"max_source_docs": 5})
    )
    container: AppContainer = application.state.container
    seed_corpus(container)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", denied)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", denied)
    with TestClient(application) as client:
        yield CollapseAPI(application, container, client, database_path)


@pytest.mark.parametrize("threshold", [0.8, 1, 1.0])
def test_real_requests_collapse_after_reranking_with_exact_survivor_provenance(
    collapse_api: CollapseAPI, monkeypatch: pytest.MonkeyPatch, threshold: float
) -> None:
    container = collapse_api.container
    executor = container.runner._executor
    retrieve = AsyncMock(wraps=executor.retrieve)
    monkeypatch.setattr(executor, "retrieve", retrieve)
    baseline = collapse_api.post("/retrieve")
    originals = [SearchResult.model_validate(source) for source in baseline["sources"]]
    expected = NearDuplicateCollapser(threshold).collapse(originals)
    assert len(expected) == (3 if threshold == 0.8 else 4)
    calls_before = list(retrieve.await_args_list)
    retrieve.reset_mock()
    corpus_before = container.document_store.list_chunks()
    with monkeypatch.context() as patch:
        patch.setattr(container.llm, "generate", denied)
        patch.setattr(executor, "answer", denied)
        patch.setattr(executor._grounder, "ground", denied)
        patch.setattr(container.event_log, "append_event", denied)
        patch.setattr(container.event_log, "append_transition", denied)
        preview = collapse_api.post("/retrieve", near_duplicate_threshold=threshold)
    assert [call.args[1:] for call in retrieve.await_args_list] == [
        call.args[1:] for call in calls_before
    ]
    assert container.event_log.list_events() == []
    generate = AsyncMock(wraps=container.llm.generate)
    monkeypatch.setattr(container.llm, "generate", generate)
    bundle = collapse_api.bundle(near_duplicate_threshold=threshold)
    generate.assert_awaited_once()
    assert generate.await_args.args[0] == bundle.snapshot.request
    assert bundle.snapshot.model_dump(mode="json")["sources"] == preview["sources"]
    assert bundle.snapshot.request.context == preview["context"]
    assert bundle.snapshot.context_sha256 == preview["context_sha256"]
    assert len(preview["sources"]) == len(expected)
    for rank, (source, original) in enumerate(
        zip(bundle.snapshot.sources, expected, strict=True), 1
    ):
        assert source.rank == rank
        assert source.chunk == original.chunk
        assert source.score == original.score
        assert source.retriever == "near_duplicate_collapse"
        assert source.path == [*original.path, original.retriever]
    policy = {"near_duplicate_threshold": float(threshold)}
    assert preview["plan"]["observation"]["evidence_policy"] == policy
    events = container.event_log.list_events(bundle.run_id)
    assert events[0]["payload"]["payload"]["evidence_policy"] == policy
    assert events[1]["payload"]["observation"]["evidence_policy"] == policy
    assert events[2]["payload"]["payload"]["observation"]["evidence_policy"] == policy
    assert container.document_store.list_chunks() == corpus_before


@pytest.mark.parametrize("value", [True, False, "0.8", "NaN", 0, -0.1, 1.01, None, [], {}])
def test_invalid_http_threshold_fails_before_runner_work(
    collapse_api: CollapseAPI, monkeypatch: pytest.MonkeyPatch, value: object
) -> None:
    monkeypatch.setattr(collapse_api.container.runner, "run", denied)
    monkeypatch.setattr(collapse_api.container.runner, "preview", denied)
    for endpoint in ("/query", "/retrieve"):
        response = collapse_api.client.post(
            endpoint, json={"query": QUERY, "near_duplicate_threshold": value}
        )
        assert response.status_code == 422, response.text
        assert response.json()["detail"][0]["loc"][:2] == ["body", "near_duplicate_threshold"]
    assert collapse_api.container.event_log.list_events() == []


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity", "1e999", "[NaN]"])
def test_nonfinite_http_threshold_returns_encodable_422_not_a_server_error(
    collapse_api: CollapseAPI, monkeypatch: pytest.MonkeyPatch, literal: str
) -> None:
    monkeypatch.setattr(collapse_api.container.runner, "run", denied)
    monkeypatch.setattr(collapse_api.container.runner, "preview", denied)
    for endpoint in ("/query", "/retrieve"):
        response = collapse_api.client.post(
            endpoint,
            content=f'{{"query": "retrieval evidence", "near_duplicate_threshold": {literal}}}',
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422, response.text
        errors = response.json()["detail"]
        assert errors[0]["loc"][:2] == ["body", "near_duplicate_threshold"]
        assert all("input" not in error for error in errors)
        json.dumps(response.json(), allow_nan=False)
    assert collapse_api.container.event_log.list_events() == []


@pytest.mark.parametrize(
    "value", [True, False, "0.8", 0, -0.1, 1.01, float("nan"), float("inf"), -float("inf"), []]
)
async def test_invalid_python_threshold_is_rejected_before_observation(
    collapse_api: CollapseAPI, monkeypatch: pytest.MonkeyPatch, value: object
) -> None:
    runner = collapse_api.container.runner
    monkeypatch.setattr(runner._analyzer, "analyze", denied)
    for method in (runner.run, runner.preview):
        with pytest.raises(ValidationError):
            await method(QUERY, near_duplicate_threshold=value)
    assert collapse_api.container.event_log.list_events() == []


def test_threshold_schema_policy_and_requests_are_frozen(collapse_api: CollapseAPI) -> None:
    schemas = collapse_api.client.get("/openapi.json").json()["components"]["schemas"]
    for name in ("QueryRequest", "RetrievalRequest"):
        model = schemas[name]
        field = model["properties"]["near_duplicate_threshold"]
        assert field["type"] == "number"
        assert field["exclusiveMinimum"] == 0 and field["maximum"] == 1
        assert "near_duplicate_threshold" not in model["required"]
    request = QueryRequest(query=QUERY, near_duplicate_threshold=0.8)
    with pytest.raises(ValidationError, match="frozen"):
        request.near_duplicate_threshold = 1
    policy = EvidencePolicy(near_duplicate_threshold=0.8)
    with pytest.raises(ValidationError, match="frozen"):
        policy.near_duplicate_threshold = 1
    assert policy.model_dump() == {"near_duplicate_threshold": 0.8}
    assert EvidencePolicy(max_chunks_per_document=1).model_dump() == {"max_chunks_per_document": 1}
    assert QueryRequest(query=QUERY).near_duplicate_threshold is None


def test_omission_never_invokes_collapser(
    collapse_api: CollapseAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    collapse = Mock(side_effect=denied)
    monkeypatch.setattr(NearDuplicateCollapser, "collapse", collapse)
    preview = collapse_api.post("/retrieve")
    bundle = collapse_api.bundle()
    assert len(preview["sources"]) == 5
    assert bundle.snapshot.model_dump(mode="json")["sources"] == preview["sources"]
    assert preview["plan"]["observation"]["evidence_policy"] is None
    assert {source.retriever for source in bundle.snapshot.sources} == {"lexical_rerank"}
    collapse.assert_not_called()


@pytest.mark.parametrize("threshold", [5e-324, 0.8, 1])
async def test_python_threshold_and_explicit_none(
    collapse_api: CollapseAPI, threshold: float
) -> None:
    runner = collapse_api.container.runner
    baseline = await runner.preview(QUERY)
    explicit_none = await runner.preview(QUERY, near_duplicate_threshold=None)
    assert explicit_none == baseline
    preview = await runner.preview(QUERY, near_duplicate_threshold=threshold)
    result = await runner.run(QUERY, near_duplicate_threshold=threshold)
    assert result.state == AgentState.DONE, result.error
    bundle = collapse_api.container.evidence_exporter.export(result.run_id)
    assert preview.sources == bundle.snapshot.sources
    assert preview.plan.observation.evidence_policy == EvidencePolicy(
        near_duplicate_threshold=threshold
    )
    assert len(preview.sources) <= len(baseline.sources)


def test_collapse_precedes_quota_and_minimum_without_refill_or_generation(
    collapse_api: CollapseAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor = collapse_api.container.runner._executor
    chunks = {
        chunk.chunk_id: chunk for chunk in collapse_api.container.document_store.list_chunks()
    }
    order = ("a-distinct", "a-original", "b-copy", "c-variant", "d-distinct")
    ranked = [
        SearchResult(
            chunk=chunks[key], score=10 - index, retriever="test-rerank", path=["upstream"]
        )
        for index, key in enumerate(order)
    ]
    raw = [result.model_dump() for result in ranked]
    rerank = AsyncMock(return_value=ranked)
    retrieve = AsyncMock(wraps=executor.retrieve)
    gate = Mock(wraps=DiversityCapGate)
    monkeypatch.setattr(executor, "retrieve", retrieve)
    monkeypatch.setattr(executor._reranker, "rerank", rerank)
    monkeypatch.setattr("agent.executor.DiversityCapGate", gate)
    baseline = collapse_api.post("/retrieve", max_chunks_per_document=1, min_evidence_documents=3)
    assert baseline["evidence_assessment"]["observed_documents"] == 4
    assert baseline["evidence_assessment"]["passed"]
    options = {
        "near_duplicate_threshold": 0.8,
        "max_chunks_per_document": 1,
        "min_evidence_documents": 3,
    }
    preview = collapse_api.post("/retrieve", **options)
    assert [source["chunk"]["chunk_id"] for source in preview["sources"]] == [
        "a-distinct",
        "d-distinct",
    ]
    assert preview["evidence_assessment"] == {
        "required_documents": 3,
        "observed_documents": 2,
        "passed": False,
    }
    for source in preview["sources"]:
        assert source["retriever"] == "diversity_cap_gate"
        assert source["path"] == ["upstream", "test-rerank", "near_duplicate_collapse"]
    generate = AsyncMock(side_effect=denied)
    monkeypatch.setattr(collapse_api.container.llm, "generate", generate)
    result = collapse_api.post("/query", **options)
    assert result["state"] == "ERROR" and result["answer"] is None
    assert "required 3 distinct evidence documents, observed 2" in result["error"]
    generate.assert_not_awaited()
    events = collapse_api.container.event_log.list_events(result["run_id"])
    assert len(events) == 6
    assert events[-2]["event_type"] == "evidence_snapshot"
    assert events[-2]["payload"]["sources"] == preview["sources"]
    assert events[-2]["payload"]["request"]["context"] == preview["context"]
    assert events[-1]["payload"]["payload"]["code"] == "insufficient_evidence_documents"
    assert events[-1]["payload"]["payload"]["evidence_assessment"] == preview["evidence_assessment"]
    assert gate.call_count == rerank.await_count == retrieve.await_count == 3
    assert all(call.args[1] == 5 for call in retrieve.await_args_list)
    assert [result.model_dump() for result in ranked] == raw
    assert collapse_api.client.get(f"/runs/{result['run_id']}/export").status_code == 409


@pytest.mark.parametrize("scope_kind", ["document_ids", "collection_id"])
def test_collapse_respects_selected_papers_and_empty_unknown_scope(
    collapse_api: CollapseAPI, scope_kind: str
) -> None:
    selected = ["paper-c", "paper-b"]
    scope: dict[str, object] = {"document_ids": [" paper-c ", "paper-b", "paper-c"]}
    if scope_kind == "collection_id":
        response = collapse_api.client.post(
            "/collections", json={"name": "Synthetic selection", "document_ids": selected}
        )
        assert response.status_code == 201
        scope = {"collection_id": response.json()["collection_id"]}
    preview = collapse_api.post("/retrieve", near_duplicate_threshold=0.8, **scope)
    bundle = collapse_api.bundle(near_duplicate_threshold=0.8, **scope)
    assert len(preview["sources"]) == 1
    assert preview["plan"]["observation"]["document_ids"] == selected
    assert bundle.plan.observation.document_ids == tuple(selected)
    assert all(source.chunk.document_id in selected for source in bundle.snapshot.sources)
    assert bundle.snapshot.model_dump(mode="json")["sources"] == preview["sources"]
    for endpoint in ("/retrieve", "/query"):
        result = collapse_api.post(endpoint, document_ids=["unknown"], near_duplicate_threshold=0.8)
        if endpoint == "/query":
            assert result["state"] == "DONE"
            result = collapse_api.container.evidence_exporter.export(result["run_id"]).model_dump(
                mode="json"
            )["snapshot"]
        assert result["sources"] == []


@pytest.mark.parametrize("phase", ["retrieval", "reranking"])
def test_scope_violation_cannot_be_hidden_by_collapsing_a_duplicate(
    collapse_api: CollapseAPI, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    executor = collapse_api.container.runner._executor
    chunks = collapse_api.container.document_store.list_chunks()
    duplicate = next(chunk for chunk in chunks if chunk.chunk_id == "b-copy")
    leaked = [SearchResult(chunk=duplicate, score=1, retriever="leak")]
    if phase == "retrieval":
        monkeypatch.setattr(executor, "retrieve", AsyncMock(return_value=leaked))
    else:
        monkeypatch.setattr(executor._reranker, "rerank", AsyncMock(return_value=leaked))
    monkeypatch.setattr("agent.executor.NearDuplicateCollapser", denied)
    monkeypatch.setattr(collapse_api.container.llm, "generate", denied)
    options = {"near_duplicate_threshold": 0.8, "document_ids": ["paper-a"]}
    result = collapse_api.post("/query", **options)
    assert result["state"] == "ERROR" and "outside document_ids" in result["error"]
    before = collapse_api.container.event_log.list_events()
    response = collapse_api.client.post("/retrieve", json={"query": QUERY, **options})
    assert response.status_code == 500
    assert response.json()["detail"]["code"] == (
        "retrieval_failed" if phase == "retrieval" else "context_preparation_failed"
    )
    assert collapse_api.container.event_log.list_events() == before
    assert not any(event["event_type"] == "evidence_snapshot" for event in before)


async def test_request_policy_scope_and_limits_are_isolated_before_awaits(
    collapse_api: CollapseAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = collapse_api.container.runner
    shared = runner._analyzer.analyze(QUERY)
    monkeypatch.setattr(runner._analyzer, "analyze", Mock(return_value=shared))
    original_retrieve = runner._executor.retrieve
    entered, release = asyncio.Event(), asyncio.Event()

    async def held(
        plan: QueryPlan, max_results: int = 8, *, document_ids: tuple[str, ...] | None = None
    ) -> list[SearchResult]:
        if plan.observation.evidence_policy == EvidencePolicy(near_duplicate_threshold=0.8):
            entered.set()
            await release.wait()
            assert max_results == 5
            assert document_ids == ("paper-a", "paper-b", "paper-c")
        return await original_retrieve(plan, max_results, document_ids=document_ids)

    monkeypatch.setattr(runner._executor, "retrieve", held)
    scope = ["paper-a", "paper-b", "paper-c"]
    pending = asyncio.create_task(
        runner.run(QUERY, document_ids=scope, near_duplicate_threshold=0.8)
    )
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        scope[:] = ["paper-d"]
        preview, default = await asyncio.gather(
            runner.preview(QUERY, near_duplicate_threshold=1),
            runner.run(QUERY, near_duplicate_threshold=None),
        )
        runner._safety_limits.max_source_docs = 1
        assert not pending.done()
    finally:
        release.set()
    result = await asyncio.wait_for(pending, timeout=2)
    assert result.state == default.state == AgentState.DONE
    bundle = collapse_api.container.evidence_exporter.export(result.run_id)
    default_bundle = collapse_api.container.evidence_exporter.export(default.run_id)
    assert bundle.configuration.max_source_docs == 5
    assert bundle.plan.observation.document_ids == ("paper-a", "paper-b", "paper-c")
    assert bundle.plan.observation.evidence_policy == EvidencePolicy(near_duplicate_threshold=0.8)
    assert len(bundle.snapshot.sources) == 2
    assert len(preview.sources) == 4
    assert len(default_bundle.snapshot.sources) == 5
    assert default_bundle.plan.observation.evidence_policy is None
    assert shared.document_ids is None and shared.evidence_policy is None


def test_legacy_runner_signatures_receive_no_new_keywords_when_omitted(
    collapse_api: CollapseAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = collapse_api.container.runner
    original_run, original_preview = runner.run, runner.preview

    async def run(query: str) -> object:
        return await original_run(query)

    async def preview(query: str) -> object:
        return await original_preview(query)

    monkeypatch.setattr(runner, "run", run)
    monkeypatch.setattr(runner, "preview", preview)
    assert collapse_api.post("/query")["state"] == "DONE"
    assert len(collapse_api.post("/retrieve")["sources"]) == 5


@pytest.mark.parametrize("hook", ["answer", "prepare_context"])
def test_legacy_executor_works_by_default_but_opt_in_requires_explicit_support(
    collapse_api: CollapseAPI, monkeypatch: pytest.MonkeyPatch, hook: str
) -> None:
    executor = collapse_api.container.runner._executor
    original_answer, original_prepare = executor.answer, executor.prepare_context
    calls = []

    async def answer(plan: QueryPlan, retrieved: list[SearchResult]) -> AgentAnswer:
        calls.append("answer")
        return await original_answer(plan, retrieved)

    async def prepare(plan: QueryPlan, retrieved: list[SearchResult]) -> EvidenceSnapshot:
        calls.append("prepare")
        return await original_prepare(plan, retrieved)

    monkeypatch.setattr(executor, hook, answer if hook == "answer" else prepare)
    assert collapse_api.post("/query")["state"] == "DONE"
    assert len(collapse_api.post("/retrieve")["sources"]) == 5
    before = list(calls)
    monkeypatch.setattr(collapse_api.container.llm, "generate", denied)
    result = collapse_api.post("/query", near_duplicate_threshold=0.8)
    assert result["state"] == "ERROR" and "evidence_policy" in result["error"]
    assert calls == before
    if hook == "prepare_context":
        response = collapse_api.client.post(
            "/retrieve", json={"query": QUERY, "near_duplicate_threshold": 0.8}
        )
        assert response.status_code == 500
        assert response.json()["detail"]["code"] == "context_preparation_failed"
        assert calls == before


@pytest.mark.parametrize("fake_provenance", [False, True])
def test_custom_preparation_cannot_ignore_collapse_or_only_relabel_duplicates(
    collapse_api: CollapseAPI, monkeypatch: pytest.MonkeyPatch, fake_provenance: bool
) -> None:
    async def bypass(
        plan: QueryPlan, retrieved: list[SearchResult], *, evidence_policy: EvidencePolicy
    ) -> EvidenceSnapshot:
        if fake_provenance:
            retrieved = [
                result.model_copy(
                    update={"retriever": "near_duplicate_collapse", "path": [result.retriever]}
                )
                for result in retrieved
            ]
        return EvidenceSnapshot.capture(plan.observation.original_query, retrieved)

    monkeypatch.setattr(collapse_api.container.runner._executor, "prepare_context", bypass)
    monkeypatch.setattr(collapse_api.container.llm, "generate", denied)
    result = collapse_api.post("/query", near_duplicate_threshold=0.8)
    assert result["state"] == "ERROR"
    assert ("violates near_duplicate_threshold" if fake_provenance else "provenance") in result[
        "error"
    ]
    before = collapse_api.container.event_log.list_events()
    response = collapse_api.client.post(
        "/retrieve", json={"query": QUERY, "near_duplicate_threshold": 0.8}
    )
    assert response.status_code == 500
    assert response.json()["detail"]["code"] == "context_preparation_failed"
    assert collapse_api.container.event_log.list_events() == before
    assert not any(event["event_type"] == "evidence_snapshot" for event in before)


@pytest.mark.parametrize("phase", ["planning", "retrieval", "context"])
def test_extensions_cannot_drop_the_requested_threshold(
    collapse_api: CollapseAPI, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    runner = collapse_api.container.runner
    plan_method = runner._planner.plan
    retrieve, prepare = runner._executor.retrieve, runner._executor.prepare_context

    def drop(plan: QueryPlan) -> QueryPlan:
        plan.observation = plan.observation.model_copy(update={"evidence_policy": None})
        return plan

    def plan(run_id: str, observation: QueryObservation) -> QueryPlan:
        return drop(plan_method(run_id, observation))

    async def dropping_retrieve(plan: QueryPlan, max_results: int = 8) -> list[SearchResult]:
        results = await retrieve(plan, max_results)
        drop(plan)
        return results

    async def dropping_prepare(
        plan: QueryPlan, results: list[SearchResult], *, evidence_policy: EvidencePolicy
    ) -> EvidenceSnapshot:
        snapshot = await prepare(plan, results, evidence_policy=evidence_policy)
        drop(plan)
        return snapshot

    if phase == "planning":
        monkeypatch.setattr(runner._planner, "plan", plan)
    elif phase == "retrieval":
        monkeypatch.setattr(runner._executor, "retrieve", dropping_retrieve)
    else:
        monkeypatch.setattr(runner._executor, "prepare_context", dropping_prepare)
    monkeypatch.setattr(collapse_api.container.llm, "generate", denied)
    result = collapse_api.post("/query", near_duplicate_threshold=0.8)
    assert result["state"] == "ERROR"
    assert "does not match near_duplicate_threshold" in result["error"]
    response = collapse_api.client.post(
        "/retrieve", json={"query": QUERY, "near_duplicate_threshold": 0.8}
    )
    assert response.status_code == 500
    assert "sources" not in response.json()


def test_collapse_failures_are_explicit_and_do_not_retry_or_generate(
    collapse_api: CollapseAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    collapse = Mock(side_effect=RuntimeError("synthetic collapse failure"))
    monkeypatch.setattr(NearDuplicateCollapser, "collapse", collapse)
    monkeypatch.setattr(collapse_api.container.llm, "generate", denied)
    result = collapse_api.post("/query", near_duplicate_threshold=0.8)
    assert result["state"] == "ERROR" and result["error"] == "synthetic collapse failure"
    assert result["answer"] is None
    assert collapse.call_count == 1
    before = collapse_api.container.event_log.list_events()
    response = collapse_api.client.post(
        "/retrieve", json={"query": QUERY, "near_duplicate_threshold": 0.8}
    )
    assert response.status_code == 500
    assert response.json()["detail"]["code"] == "context_preparation_failed"
    assert "synthetic collapse failure" not in response.text
    assert "sources" not in response.json()
    assert collapse.call_count == 2
    assert collapse_api.container.event_log.list_events() == before


def test_saved_policy_provenance_and_comparison_survive_restart_without_live_work(
    collapse_api: CollapseAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = collapse_api.bundle(near_duplicate_threshold=0.9)
    candidate = collapse_api.bundle(near_duplicate_threshold=1)
    assert baseline.snapshot == candidate.snapshot
    export_path = f"/runs/{candidate.run_id}/export"
    compare_path = f"/runs/{baseline.run_id}/compare/{candidate.run_id}"
    paths = (export_path, export_path + "?format=markdown", compare_path)
    before = {path: collapse_api.client.get(path).content for path in paths}
    comparison = RunComparison.model_validate_json(before[compare_path])
    assert comparison.baseline.evidence_policy == EvidencePolicy(near_duplicate_threshold=0.9)
    assert comparison.candidate.evidence_policy == EvidencePolicy(near_duplicate_threshold=1)
    assert [key for key, changed in comparison.changes.model_dump().items() if changed] == [
        "evidence_policy_changed"
    ]
    assert comparison.any_changes
    assert b"near_duplicate_threshold=1.0" in before[export_path + "?format=markdown"]
    assert b"semantic equivalence" in before[export_path + "?format=markdown"]
    assert EvidenceBundle.model_validate_json(before[export_path]) == candidate
    events_before = collapse_api.container.event_log.list_events()
    with sqlite3.connect(collapse_api.database_path) as connection:
        connection.execute("DELETE FROM chunks")
        connection.execute("DELETE FROM documents")
    reopened = AppContainer(offline_settings(collapse_api.database_path))
    collapse_api.application.state.container = reopened
    monkeypatch.setattr(reopened.llm, "generate", denied)
    monkeypatch.setattr(reopened.runner._executor, "retrieve", denied)
    monkeypatch.setattr(reopened.document_store, "list_chunks", denied)
    monkeypatch.setattr(reopened.event_log, "append_event", denied)
    for path, content in before.items():
        response = collapse_api.client.get(path)
        assert response.status_code == 200, response.text
        assert response.content == content
    assert reopened.event_log.list_events() == events_before


@pytest.mark.parametrize("damage", ["initial", "decision", "retrieval_plan", "boolean", "path"])
def test_export_revalidates_saved_threshold_and_transformation_provenance(
    collapse_api: CollapseAPI, damage: str
) -> None:
    bundle = collapse_api.bundle(near_duplicate_threshold=0.8, max_chunks_per_document=1)
    events = collapse_api.container.event_log.list_events(bundle.run_id)
    if damage in ("initial", "boolean"):
        target = events[0]
        target["payload"]["payload"]["evidence_policy"]["near_duplicate_threshold"] = (
            True if damage == "boolean" else 1
        )
    elif damage == "decision":
        target = events[1]
        target["payload"]["observation"]["evidence_policy"]["near_duplicate_threshold"] = 1
    elif damage == "retrieval_plan":
        target = events[2]
        target["payload"]["payload"]["observation"]["evidence_policy"][
            "near_duplicate_threshold"
        ] = 1
    else:
        target = events[4]
        target["payload"]["sources"][0]["path"].pop()
    with sqlite3.connect(collapse_api.database_path) as connection:
        connection.execute(
            "UPDATE agent_events SET payload = ? WHERE id = ?",
            (json.dumps(target["payload"]), target["id"]),
        )
    for suffix in ("", "?format=markdown"):
        response = collapse_api.client.get(f"/runs/{bundle.run_id}/export{suffix}")
        assert response.status_code == 409
        assert response.json()["detail"]["code"] == "invalid_run_record"


@pytest.mark.parametrize(
    "options", [{}, {"max_chunks_per_document": 1}, {"min_evidence_documents": 1}]
)
def test_version_one_bundles_without_threshold_keep_their_shape_and_remain_readable(
    collapse_api: CollapseAPI, options: dict[str, int]
) -> None:
    bundle = collapse_api.bundle(**options)
    payload = bundle.model_dump(mode="json")
    assert "near_duplicate_threshold" not in json.dumps(payload)
    assert EvidenceBundle.model_validate(payload) == bundle
    assert bundle.schema_version == "1.0"
    if not options:
        payload["plan"]["observation"].pop("evidence_policy")
        assert EvidenceBundle.model_validate(payload).plan.observation.evidence_policy is None


@pytest.mark.parametrize(
    ("first", "second"),
    [("alpha beta alpha", "beta alpha"), ("the and of", "a an the")],
)
def test_threshold_one_collapses_equal_meaningful_term_sets_not_only_identical_text(
    collapse_api: CollapseAPI, monkeypatch: pytest.MonkeyPatch, first: str, second: str
) -> None:
    executor = collapse_api.container.runner._executor
    chunks = collapse_api.container.document_store.list_chunks()
    ranked = [
        SearchResult(
            chunk=chunk.model_copy(update={"text": text}),
            score=2 - index,
            retriever="synthetic-rerank",
        )
        for index, (chunk, text) in enumerate(zip(chunks[:2], (first, second), strict=True))
    ]
    monkeypatch.setattr(executor._reranker, "rerank", AsyncMock(return_value=ranked))
    assert first != second
    preview = collapse_api.post("/retrieve", near_duplicate_threshold=1)
    bundle = collapse_api.bundle(near_duplicate_threshold=1)
    assert len(preview["sources"]) == len(bundle.snapshot.sources) == 1
    assert bundle.snapshot.sources[0].chunk == ranked[0].chunk
    assert bundle.snapshot.sources[0].score == ranked[0].score
