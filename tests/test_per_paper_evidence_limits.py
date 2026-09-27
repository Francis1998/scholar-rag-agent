"""Request-level per-document limits through the real API and shared executor."""

import asyncio
import json
import sqlite3
from collections import Counter
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

from agent.comparison_models import RunComparison
from agent.evidence import EvidenceBundle, EvidenceSnapshot, RunConfiguration, text_digest
from agent.models import AgentAnswer, AgentState, QueryObservation, QueryPlan
from api.application import create_app
from api.dependencies import AppContainer
from api.schemas import QueryRequest
from retrieval.diversity_cap_gate import DiversityCapGate
from retrieval.evidence_policy import EvidencePolicy
from retrieval.models import Chunk, Document, SearchResult

QUERY = "retrieval evidence"


def denied(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("This offline request must not call a provider or write preview events.")


@dataclass
class CapAPI:
    application: FastAPI
    container: AppContainer
    client: TestClient
    database_path: Path

    def post(self, endpoint: str, **options: object) -> dict[str, Any]:
        response = self.client.post(endpoint, json={"query": QUERY, **options})
        assert response.status_code == 200, response.text
        body: dict[str, Any] = response.json()
        if endpoint == "/query":
            result = body["result"]
            assert result["state"] == "DONE", result
            exported = self.client.get(f"/runs/{result['run_id']}/export")
            assert exported.status_code == 200, exported.text
            body = exported.json()
        return body


@pytest.fixture
def cap_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[CapAPI]:
    database_path = tmp_path / "per-paper.sqlite3"
    settings = offline_settings(database_path).model_copy(update={"max_source_docs": 6})
    application = create_app(settings)
    container: AppContainer = application.state.container
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", denied)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", denied)
    for document_id, count in (("paper-a", 3), ("paper-b", 2), ("paper-c", 1)):
        chunks = [
            Chunk(
                chunk_id=f"{document_id}-{index}",
                document_id=document_id,
                title="Identical synthetic title",
                text=f"GraphRAG retrieval evidence {index} for {document_id}; synthetic only.",
                source="synthetic:shared-source",
                metadata={"license": "synthetic-only"},
            )
            for index in range(count)
        ]
        container.document_store.add_documents(
            [
                Document(
                    document_id=document_id,
                    title=chunks[0].title,
                    text="\n".join(chunk.text for chunk in chunks),
                    source=chunks[0].source,
                )
            ],
            chunks,
        )
        container.hybrid_retriever.add_chunks(chunks)
        container.graph_builder.index_chunks(chunks)
    with TestClient(application) as client:
        yield CapAPI(application, container, client, database_path)


@pytest.mark.parametrize("endpoint", ["/query", "/retrieve"])
def test_real_requests_limit_multiple_passages_per_document(cap_api: CapAPI, endpoint: str) -> None:
    baseline = cap_api.post("/retrieve")
    assert Counter(source["chunk"]["document_id"] for source in baseline["sources"]) == {
        "paper-a": 3,
        "paper-b": 2,
        "paper-c": 1,
    }
    limited = cap_api.post(endpoint, max_chunks_per_document=1)
    sources = limited["snapshot"]["sources"] if endpoint == "/query" else limited["sources"]
    assert Counter(source["chunk"]["document_id"] for source in sources) == {
        "paper-a": 1,
        "paper-b": 1,
        "paper-c": 1,
    }


@pytest.mark.parametrize("document_ids", [["paper-c"], ["not-ingested"]])
def test_equal_evidence_still_reports_requested_policy_change(
    cap_api: CapAPI, monkeypatch: pytest.MonkeyPatch, document_ids: list[str]
) -> None:
    baseline = cap_api.post("/query", document_ids=document_ids, max_chunks_per_document=1)
    candidate = cap_api.post("/query", document_ids=document_ids, max_chunks_per_document=2)
    assert baseline["snapshot"] == candidate["snapshot"]
    assert baseline["configuration"] == candidate["configuration"]
    before_events = cap_api.container.event_log.list_events()
    monkeypatch.setattr(cap_api.container.llm, "generate", denied)
    monkeypatch.setattr(cap_api.container.runner._executor, "retrieve", denied)
    monkeypatch.setattr(cap_api.container.event_log, "append_event", denied)
    response = cap_api.client.get(f"/runs/{baseline['run_id']}/compare/{candidate['run_id']}")
    assert response.status_code == 200, response.text
    assert response.json()["any_changes"] is True
    comparison = response.json()
    assert comparison["baseline"]["evidence_policy"] == {"max_chunks_per_document": 1}
    assert comparison["candidate"]["evidence_policy"] == {"max_chunks_per_document": 2}
    assert [key for key, changed in comparison["changes"].items() if changed] == [
        "evidence_policy_changed"
    ]
    assert any("policies differ" in notice for notice in comparison["notices"])
    assert cap_api.container.event_log.list_events() == before_events


def test_policy_and_exact_post_rerank_context_survive_every_recorded_surface(
    cap_api: CapAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    query = "compare GraphRAG versus retrieval"
    container = cap_api.container
    executor = container.runner._executor
    corpus_before = container.document_store.list_chunks()
    factory = Mock(wraps=DiversityCapGate)
    monkeypatch.setattr("agent.executor.DiversityCapGate", factory)
    spies = [
        AsyncMock(wraps=component.retrieve)
        for component in (container.hybrid_retriever, executor._multihop_retriever)
    ]
    for component, spy in zip(
        (container.hybrid_retriever, executor._multihop_retriever), spies, strict=True
    ):
        monkeypatch.setattr(component, "retrieve", spy)
    baseline = cap_api.post("/retrieve", query=query)
    baseline_calls = [list(spy.await_args_list) for spy in spies]
    for spy in spies:
        spy.reset_mock()
    factory.assert_not_called()

    with monkeypatch.context() as patch:
        patch.setattr(executor, "answer", denied)
        patch.setattr(executor._grounder, "ground", denied)
        patch.setattr(container.llm, "generate", denied)
        patch.setattr(container.event_log, "append_event", denied)
        patch.setattr(container.event_log, "append_transition", denied)
        preview = cap_api.post("/retrieve", query=query, max_chunks_per_document=1)
    assert container.event_log.list_events() == []
    assert [list(spy.await_args_list) for spy in spies] == baseline_calls
    assert all(len(calls) == 2 for calls in baseline_calls)
    generate = AsyncMock(wraps=container.llm.generate)
    monkeypatch.setattr(container.llm, "generate", generate)
    bundle = cap_api.post("/query", query=query, max_chunks_per_document=1)
    generate.assert_awaited_once()
    assert factory.call_count == 2
    assert all(call.kwargs == {"max_per_source": 1} for call in factory.call_args_list)
    assert preview["sources"] == bundle["snapshot"]["sources"]
    assert preview["context"] == bundle["snapshot"]["request"]["context"]
    assert preview["context_sha256"] == bundle["snapshot"]["context_sha256"]
    assert preview["context_sha256"] == text_digest(preview["context"])
    assert generate.await_args.args[0].context == preview["context"]
    assert preview["plan"] == {
        key: value for key, value in bundle["plan"].items() if key != "run_id"
    }
    policy = {"max_chunks_per_document": 1}
    assert preview["plan"]["observation"]["evidence_policy"] == policy
    events = container.event_log.list_events(bundle["run_id"])
    assert events[0]["payload"]["payload"]["evidence_policy"] == policy
    assert events[1]["payload"]["observation"]["evidence_policy"] == policy
    assert events[2]["payload"]["payload"]["observation"]["evidence_policy"] == policy
    assert bundle["events"][0]["payload"]["payload"]["evidence_policy"] == policy
    assert preview["configuration"] == bundle["configuration"]
    assert set(bundle["configuration"]) == set(RunConfiguration.model_fields)
    assert len(RunConfiguration.model_fields) == 4

    originals = {source["chunk"]["chunk_id"]: source for source in baseline["sources"]}
    original_ranks = []
    for rank, source in enumerate(preview["sources"], start=1):
        original = originals[source["chunk"]["chunk_id"]]
        original_ranks.append(original["rank"])
        assert source["rank"] == rank
        assert source["chunk"] == original["chunk"]
        assert source["score"] == original["score"]
        assert source["text_sha256"] == original["text_sha256"]
        assert source["retriever"] == "diversity_cap_gate"
        assert source["path"] == [*original["path"], original["retriever"]]
    assert original_ranks == sorted(original_ranks)
    assert container.document_store.list_chunks() == corpus_before
    markdown = cap_api.client.get(f"/runs/{bundle['run_id']}/export?format=markdown").text
    assert "## Per-paper evidence policy" in markdown
    assert "max_chunks_per_document=1" in markdown
    assert "distinct-paper coverage is not guaranteed" in markdown


@pytest.mark.parametrize("limit", [2, 50])
def test_valid_quotas_do_not_refill_or_increase_the_candidate_pool(
    cap_api: CapAPI, limit: int
) -> None:
    for endpoint in ("/query", "/retrieve"):
        result = cap_api.post(endpoint, max_chunks_per_document=limit)
        sources = result["snapshot"]["sources"] if endpoint == "/query" else result["sources"]
        assert Counter(source["chunk"]["document_id"] for source in sources) == {
            "paper-a": min(3, limit),
            "paper-b": min(2, limit),
            "paper-c": 1,
        }
        assert len(sources) <= result["configuration"]["max_source_docs"] == 6


def test_schema_and_observation_expose_strict_optional_frozen_policy(cap_api: CapAPI) -> None:
    schemas = cap_api.client.get("/openapi.json").json()["components"]["schemas"]
    for name in ("QueryRequest", "RetrievalRequest"):
        model = schemas[name]
        field = model["properties"]["max_chunks_per_document"]
        assert field["type"] == "integer"
        assert field["minimum"] == 1 and field["maximum"] == 50
        assert "max_chunks_per_document" not in model["required"]
    assert QueryRequest(query=QUERY).max_chunks_per_document is None
    request = QueryRequest(query=QUERY, max_chunks_per_document=1)
    with pytest.raises(ValidationError, match="frozen"):
        request.max_chunks_per_document = 2
    observation = QueryObservation.model_validate(
        cap_api.post("/retrieve", max_chunks_per_document=1)["plan"]["observation"]
    )
    assert observation.evidence_policy is not None
    with pytest.raises(ValidationError, match="frozen"):
        observation.evidence_policy = EvidencePolicy(max_chunks_per_document=2)
    with pytest.raises(ValidationError, match="frozen"):
        observation.evidence_policy.max_chunks_per_document = 2


@pytest.mark.parametrize("value", [True, "1", 1.0, 1.5, 0, -1, 51, None])
def test_invalid_http_policy_fails_before_any_work(
    cap_api: CapAPI, monkeypatch: pytest.MonkeyPatch, value: object
) -> None:
    monkeypatch.setattr(cap_api.container.runner, "run", denied)
    monkeypatch.setattr(cap_api.container.runner, "preview", denied)
    for endpoint in ("/query", "/retrieve"):
        response = cap_api.client.post(
            endpoint, json={"query": QUERY, "max_chunks_per_document": value}
        )
        assert response.status_code == 422
        assert response.json()["detail"][0]["loc"][:2] == ["body", "max_chunks_per_document"]
    assert cap_api.container.event_log.list_events() == []


@pytest.mark.parametrize("value", [False, "2", 1.0, 1.5, 0, 51])
async def test_invalid_python_policy_fails_before_observation_or_events(
    cap_api: CapAPI, monkeypatch: pytest.MonkeyPatch, value: object
) -> None:
    runner = cap_api.container.runner
    monkeypatch.setattr(runner._analyzer, "analyze", denied)
    for method in (runner.run, runner.preview):
        with pytest.raises(ValidationError):
            await method(QUERY, max_chunks_per_document=value)
    assert cap_api.container.event_log.list_events() == []


@pytest.mark.parametrize("scope_kind", ["document_ids", "collection_id"])
def test_selected_documents_and_collections_keep_the_quota_and_scope(
    cap_api: CapAPI, scope_kind: str
) -> None:
    selected = ["paper-b", "paper-a"]
    if scope_kind == "document_ids":
        scope = {"document_ids": [" paper-b ", "paper-a", "paper-b"]}
    else:
        saved = cap_api.client.post(
            "/collections", json={"name": "Synthetic", "document_ids": selected}
        )
        assert saved.status_code == 201, saved.text
        scope = {"collection_id": saved.json()["collection_id"]}
    for endpoint in ("/query", "/retrieve"):
        before = cap_api.container.event_log.list_events()
        result = cap_api.post(endpoint, max_chunks_per_document=1, **scope)
        sources = result["snapshot"]["sources"] if endpoint == "/query" else result["sources"]
        assert Counter(source["chunk"]["document_id"] for source in sources) == {
            "paper-a": 1,
            "paper-b": 1,
        }
        assert result["plan"]["observation"]["document_ids"] == selected
        assert result["plan"]["observation"]["evidence_policy"] == {"max_chunks_per_document": 1}
        assert "paper-c" not in json.dumps(result)
        if endpoint == "/retrieve":
            assert cap_api.container.event_log.list_events() == before


def test_cap_does_not_widen_unknown_ids_or_invalid_collections(
    cap_api: CapAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    for endpoint in ("/query", "/retrieve"):
        result = cap_api.post(endpoint, document_ids=["not-ingested"], max_chunks_per_document=1)
        sources = result["snapshot"]["sources"] if endpoint == "/query" else result["sources"]
        assert sources == []
        assert result["plan"]["observation"]["document_ids"] == ["not-ingested"]
    saved = cap_api.client.post(
        "/collections", json={"name": "Missing member", "document_ids": ["paper-a"]}
    ).json()
    with sqlite3.connect(cap_api.database_path) as connection:
        connection.execute("DELETE FROM documents WHERE document_id = ?", ("paper-a",))
    before = cap_api.container.event_log.list_events()
    monkeypatch.setattr(cap_api.container.runner, "run", denied)
    monkeypatch.setattr(cap_api.container.runner, "preview", denied)
    for endpoint in ("/query", "/retrieve"):
        for scope, status in (
            ({"collection_id": "col_" + "0" * 32}, 404),
            ({"collection_id": saved["collection_id"]}, 409),
            ({"document_ids": []}, 422),
        ):
            response = cap_api.client.post(
                endpoint, json={"query": QUERY, "max_chunks_per_document": 1, **scope}
            )
            assert response.status_code == status, response.text
    assert cap_api.container.event_log.list_events() == before


async def test_frozen_requests_are_isolated_while_query_and_preview_overlap(
    cap_api: CapAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = cap_api.container.runner
    shared_observation = runner._analyzer.analyze(QUERY)
    monkeypatch.setattr(runner._analyzer, "analyze", Mock(return_value=shared_observation))
    entered, release = asyncio.Event(), asyncio.Event()
    retrieve = runner._executor.retrieve

    async def held(
        plan: QueryPlan, max_results: int = 8, *, document_ids: tuple[str, ...] | None = None
    ) -> list[SearchResult]:
        if plan.observation.evidence_policy == EvidencePolicy(max_chunks_per_document=1):
            entered.set()
            await release.wait()
            assert document_ids == ("paper-a", "paper-b")
            assert plan.observation.evidence_policy.max_chunks_per_document == 1
        return await retrieve(plan, max_results, document_ids=document_ids)

    monkeypatch.setattr(runner._executor, "retrieve", held)
    supplied = ["paper-a", "paper-b"]
    pending = asyncio.create_task(
        runner.run(QUERY, document_ids=supplied, max_chunks_per_document=1)
    )
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        supplied[:] = ["paper-c"]
        preview, default_run, explicit_none = await asyncio.wait_for(
            asyncio.gather(
                runner.preview(QUERY, document_ids=["paper-a"], max_chunks_per_document=2),
                runner.run(QUERY),
                runner.preview(QUERY, max_chunks_per_document=None),
            ),
            timeout=5,
        )
        assert not pending.done()
    finally:
        release.set()
    first = await asyncio.wait_for(pending, timeout=2)
    assert first.state == default_run.state == AgentState.DONE
    first_bundle = cap_api.container.evidence_exporter.export(first.run_id)
    default_bundle = cap_api.container.evidence_exporter.export(default_run.run_id)
    assert Counter(source.chunk.document_id for source in first_bundle.snapshot.sources) == {
        "paper-a": 1,
        "paper-b": 1,
    }
    assert Counter(source.chunk.document_id for source in preview.sources) == {"paper-a": 2}
    assert explicit_none.sources == default_bundle.snapshot.sources
    assert len(default_bundle.snapshot.sources) == 6
    assert default_bundle.plan.observation.evidence_policy is None
    assert first_bundle.plan.observation.document_ids == ("paper-a", "paper-b")
    assert first_bundle.plan.observation.evidence_policy == EvidencePolicy(
        max_chunks_per_document=1
    )
    assert preview.plan.observation.evidence_policy == EvidencePolicy(max_chunks_per_document=2)
    assert shared_observation.evidence_policy is None
    assert shared_observation.document_ids is None


@pytest.mark.parametrize("stage", ["answer", "prepare_context"])
def test_legacy_executor_signatures_work_by_default_and_reject_unsupported_opt_in(
    cap_api: CapAPI, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    executor = cap_api.container.runner._executor
    calls = []
    if stage == "answer":
        answer = executor.answer

        async def legacy_answer(plan: QueryPlan, retrieved: list[SearchResult]) -> AgentAnswer:
            calls.append("answer")
            return await answer(plan, retrieved)

        monkeypatch.setattr(executor, "answer", legacy_answer)
    else:
        prepare = executor.prepare_context

        async def legacy_prepare(
            plan: QueryPlan, retrieved: list[SearchResult]
        ) -> EvidenceSnapshot:
            calls.append("prepare")
            return await prepare(plan, retrieved)

        monkeypatch.setattr(executor, "prepare_context", legacy_prepare)
    default = cap_api.client.post("/query", json={"query": QUERY}).json()["result"]
    assert default["state"] == "DONE", default
    assert cap_api.post("/retrieve")["plan"]["observation"]["evidence_policy"] is None
    before = len(calls)
    monkeypatch.setattr(cap_api.container.llm, "generate", denied)
    opted_in = cap_api.client.post(
        "/query", json={"query": QUERY, "max_chunks_per_document": 1}
    ).json()["result"]
    assert opted_in["state"] == "ERROR"
    assert "evidence_policy" in opted_in["error"]
    assert len(calls) == before
    if stage == "prepare_context":
        preview = cap_api.client.post(
            "/retrieve", json={"query": QUERY, "max_chunks_per_document": 1}
        )
        assert preview.status_code == 500
        assert preview.json()["detail"]["code"] == "context_preparation_failed"
        assert len(calls) == before


def test_custom_preparation_cannot_ignore_policy_or_generate_from_uncapped_context(
    cap_api: CapAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def bypass(
        plan: QueryPlan,
        retrieved: list[SearchResult],
        *,
        evidence_policy: EvidencePolicy | None = None,
    ) -> EvidenceSnapshot:
        return EvidenceSnapshot.capture(plan.observation.original_query, retrieved)

    monkeypatch.setattr(cap_api.container.runner._executor, "prepare_context", bypass)
    monkeypatch.setattr(cap_api.container.llm, "generate", denied)
    query = cap_api.client.post("/query", json={"query": QUERY, "max_chunks_per_document": 1})
    result = query.json()["result"]
    assert result["state"] == "ERROR"
    assert "diversity_cap_gate provenance" in result["error"]
    before = cap_api.container.event_log.list_events()
    preview = cap_api.client.post("/retrieve", json={"query": QUERY, "max_chunks_per_document": 1})
    assert preview.status_code == 500
    assert preview.json()["detail"]["code"] == "context_preparation_failed"
    assert cap_api.container.event_log.list_events() == before
    assert not any(event["event_type"] == "evidence_snapshot" for event in before)


def test_custom_answer_accepting_keywords_must_still_record_opted_in_context(
    cap_api: CapAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer = AsyncMock(return_value=AgentAnswer(answer="unsupported", citations=[], claims=[]))
    monkeypatch.setattr(cap_api.container.runner._executor, "answer", answer)
    response = cap_api.client.post("/query", json={"query": QUERY, "max_chunks_per_document": 1})
    result = response.json()["result"]
    assert result["state"] == "ERROR"
    assert "did not capture evidence" in result["error"]
    answer.assert_awaited_once()
    assert cap_api.client.get(f"/runs/{result['run_id']}/export").status_code == 409


def test_planner_cannot_drop_requested_policy_before_execution(
    cap_api: CapAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = cap_api.container.runner
    original = runner._planner.plan

    def discard_policy(run_id: str, observation: QueryObservation) -> QueryPlan:
        return original(run_id, observation.model_copy(update={"evidence_policy": None}))

    monkeypatch.setattr(runner._planner, "plan", discard_policy)
    monkeypatch.setattr(runner._executor, "retrieve", denied)
    query = cap_api.client.post("/query", json={"query": QUERY, "max_chunks_per_document": 1})
    result = query.json()["result"]
    assert result["state"] == "ERROR"
    assert "does not match the requested quota" in result["error"]
    before = cap_api.container.event_log.list_events()
    assert before[0]["payload"]["payload"]["evidence_policy"] == {"max_chunks_per_document": 1}
    preview = cap_api.client.post("/retrieve", json={"query": QUERY, "max_chunks_per_document": 1})
    assert preview.status_code == 500
    assert preview.json()["detail"]["code"] == "planning_failed"
    assert cap_api.container.event_log.list_events() == before


def test_scope_errors_are_not_hidden_by_the_post_rerank_gate(
    cap_api: CapAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    excluded = next(
        chunk
        for chunk in cap_api.container.document_store.list_chunks()
        if chunk.document_id == "paper-c"
    )
    monkeypatch.setattr(
        cap_api.container.runner._executor._reranker,
        "rerank",
        AsyncMock(
            return_value=[SearchResult(chunk=excluded, score=1, retriever="leaking-reranker")]
        ),
    )
    monkeypatch.setattr("agent.executor.DiversityCapGate", denied)
    monkeypatch.setattr(cap_api.container.llm, "generate", denied)
    payload = {"query": QUERY, "document_ids": ["paper-a"], "max_chunks_per_document": 1}
    result = cap_api.client.post("/query", json=payload).json()["result"]
    assert result["state"] == "ERROR" and "outside document_ids" in result["error"]
    preview = cap_api.client.post("/retrieve", json=payload)
    assert preview.status_code == 500
    assert preview.json()["detail"]["code"] == "context_preparation_failed"
    assert not any(
        event["event_type"] == "evidence_snapshot"
        for event in cap_api.container.event_log.list_events()
    )


@pytest.mark.parametrize(
    "damage", ["initial", "decision", "retrieval_plan", "invalid_policy", "quota", "gate"]
)
def test_export_rejects_inconsistent_policy_and_false_gate_provenance(
    cap_api: CapAPI, damage: str
) -> None:
    bundle = cap_api.post("/query", max_chunks_per_document=1)
    events = cap_api.container.event_log.list_events(bundle["run_id"])
    if damage == "initial":
        target = events[0]
        target["payload"]["payload"].pop("evidence_policy")
    elif damage == "decision":
        target = events[1]
        target["payload"]["observation"]["evidence_policy"]["max_chunks_per_document"] = 2
    elif damage == "retrieval_plan":
        target = events[2]
        target["payload"]["payload"]["observation"]["evidence_policy"][
            "max_chunks_per_document"
        ] = 2
    elif damage == "invalid_policy":
        target = events[0]
        target["payload"]["payload"]["evidence_policy"]["max_chunks_per_document"] = True
    else:
        target = next(event for event in events if event["event_type"] == "evidence_snapshot")
        sources = target["payload"]["sources"]
        if damage == "quota":
            sources[1]["chunk"]["document_id"] = sources[0]["chunk"]["document_id"]
        else:
            sources[0]["retriever"] = "lexical_rerank"
    with sqlite3.connect(cap_api.database_path) as connection:
        connection.execute(
            "UPDATE agent_events SET payload = ? WHERE id = ?",
            (json.dumps(target["payload"]), target["id"]),
        )
    for fmt in ("json", "markdown"):
        response = cap_api.client.get(f"/runs/{bundle['run_id']}/export", params={"format": fmt})
        assert response.status_code == 409, response.text
        assert response.json()["detail"]["code"] == "invalid_run_record"


def test_saved_exports_keep_policy_after_restart_without_live_corpus_or_model_work(
    cap_api: CapAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = cap_api.post("/query", document_ids=["paper-a", "paper-b"], max_chunks_per_document=1)
    export_path = f"/runs/{bundle['run_id']}/export"
    before = {
        fmt: cap_api.client.get(export_path, params={"format": fmt}).content
        for fmt in ("json", "markdown")
    }
    with sqlite3.connect(cap_api.database_path) as connection:
        connection.execute("UPDATE chunks SET text = ?", ("Changed after the saved query",))
    reopened = AppContainer(
        offline_settings(cap_api.database_path).model_copy(update={"max_source_docs": 1})
    )
    cap_api.application.state.container = reopened
    monkeypatch.setattr(reopened.llm, "generate", denied)
    monkeypatch.setattr(reopened.runner._executor, "retrieve", denied)
    monkeypatch.setattr(reopened.document_store, "list_chunks", denied)
    monkeypatch.setattr(reopened.event_log, "append_event", denied)
    for fmt, content in before.items():
        after = cap_api.client.get(export_path, params={"format": fmt})
        assert after.status_code == 200, after.text
        assert after.content == content
    saved = EvidenceBundle.model_validate_json(before["json"])
    assert saved.plan.observation.evidence_policy == EvidencePolicy(max_chunks_per_document=1)
    assert saved.configuration.max_source_docs == 6


def test_pre_policy_saved_runs_and_comparison_json_remain_readable(cap_api: CapAPI) -> None:
    bundle = cap_api.post("/query")
    events = cap_api.container.event_log.list_events(bundle["run_id"])
    with sqlite3.connect(cap_api.database_path) as connection:
        for event in events:
            payload = event["payload"]
            if event["event_type"] == "decision_log":
                payload["observation"].pop("evidence_policy", None)
            elif event["event_type"] == "state_transition":
                payload["payload"].pop("evidence_policy", None)
                if "observation" in payload["payload"]:
                    payload["payload"]["observation"].pop("evidence_policy", None)
            connection.execute(
                "UPDATE agent_events SET payload = ? WHERE id = ?",
                (json.dumps(payload), event["id"]),
            )
    saved = cap_api.container.evidence_exporter.export(bundle["run_id"])
    assert saved.plan.observation.evidence_policy is None
    assert len(saved.snapshot.sources) == 6
    assert saved.schema_version == "1.0"
    legacy_bundle = saved.model_dump(mode="json")
    legacy_bundle["plan"]["observation"].pop("evidence_policy")
    assert EvidenceBundle.model_validate(legacy_bundle).plan.observation.evidence_policy is None
    comparison = cap_api.client.get(f"/runs/{bundle['run_id']}/compare/{bundle['run_id']}").json()
    assert not comparison["any_changes"]
    assert comparison["baseline"]["evidence_policy"] is None
    comparison["baseline"].pop("evidence_policy")
    comparison["candidate"].pop("evidence_policy")
    comparison["changes"].pop("evidence_policy_changed")
    legacy_comparison = RunComparison.model_validate(comparison)
    assert legacy_comparison.baseline.evidence_policy is None
    assert not legacy_comparison.changes.evidence_policy_changed
