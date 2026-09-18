"""Real API-to-evidence scope acceptance, isolation, and compatibility tests."""

import asyncio
import json
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from agent.evidence import EvidenceBundle, RunConfiguration
from agent.models import AgentAnswer, AgentRunResult, AgentState, QueryPlan
from api.dependencies import AppContainer
from api.main import app
from api.schemas import IngestResponse
from retrieval.hyde import HyDEExpander
from retrieval.models import Document, SearchResult
from retrieval.rrf import reciprocal_rank_fusion
from tests.test_evidence_export import no_live_call, offline_settings


@dataclass
class ScopeAPI:
    application: FastAPI
    container: AppContainer
    client: TestClient
    database_path: Path

    def ingest(self, title: str, text: str) -> IngestResponse:
        response = self.client.post(
            "/ingest/text", json={"title": title, "text": text, "source": "synthetic:scope"}
        )
        response.raise_for_status()
        return IngestResponse.model_validate_json(response.content)

    def query(self, query: str, **values: object) -> AgentRunResult:
        response = self.client.post("/query", json={"query": query, **values})
        response.raise_for_status()
        return AgentRunResult.model_validate(response.json()["result"])

    def export(self, run_id: str) -> EvidenceBundle:
        response = self.client.get(f"/runs/{run_id}/export")
        response.raise_for_status()
        return EvidenceBundle.model_validate_json(response.content)


@pytest.fixture
def scope_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[ScopeAPI]:
    database_path = tmp_path / "scope.sqlite3"
    settings = offline_settings(database_path).model_copy(update={"max_source_docs": 2})
    container = AppContainer(settings)
    application = FastAPI()
    application.include_router(app.router)
    application.state.container = container
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", no_live_call)
    with TestClient(application) as client:
        yield ScopeAPI(application, container, client, database_path)


def source_documents(bundle: EvidenceBundle) -> set[str]:
    return {source.chunk.document_id for source in bundle.snapshot.sources}


def test_api_schema_exposes_optional_nonempty_bounded_scope(scope_api: ScopeAPI) -> None:
    schema = scope_api.client.get("/openapi.json").json()["components"]["schemas"]["QueryRequest"]
    field = schema["properties"]["document_ids"]
    assert "document_ids" not in schema["required"]
    assert field["type"] == "array"
    assert field["minItems"] == 1
    assert field["maxItems"] == 100
    assert field["items"]["type"] == "string"
    assert field["items"]["minLength"] == 1
    assert field["items"]["maxLength"] == 128
    assert "pattern" not in field["items"]


def test_real_api_excludes_other_papers_from_context_citations_and_export(
    scope_api: ScopeAPI,
) -> None:
    first = scope_api.ingest("Selected note", "GraphRAG connects selected synthetic evidence.")
    second = scope_api.ingest("Excluded note", "GraphRAG connects EXCLUDED_MARKER evidence.")
    unscoped = scope_api.query("What does GraphRAG connect?")
    assert unscoped.state == AgentState.DONE
    assert unscoped.answer is not None
    assert {c.document_id for c in unscoped.answer.citations} == {
        first.document_id,
        second.document_id,
    }

    scoped = scope_api.query(
        "What does GraphRAG connect?", document_ids=[f" {first.document_id} ", first.document_id]
    )
    assert scoped.state == AgentState.DONE, scoped.error
    assert scoped.answer is not None
    assert {c.document_id for c in scoped.answer.citations} == {first.document_id}
    bundle = scope_api.export(scoped.run_id)
    assert source_documents(bundle) == {first.document_id}
    assert "EXCLUDED_MARKER" not in bundle.snapshot.request.context
    assert bundle.plan.observation.document_ids == (first.document_id,)
    assert scoped.observation is not None
    assert scoped.observation.document_ids == (first.document_id,)
    assert bundle.plan == scoped.plan
    assert bundle.answer == scoped.answer
    assert set(bundle.configuration.model_dump()) == set(RunConfiguration.model_fields)
    assert len(RunConfiguration.model_fields) == 4
    assert all(c.evidence_rank is not None for c in bundle.citation_evidence)
    events = scope_api.client.get(f"/runs/{scoped.run_id}/events").json()
    assert events[0]["payload"]["payload"]["document_ids"] == [first.document_id]
    assert events[1]["payload"]["observation"]["document_ids"] == [first.document_id]
    assert events[2]["payload"]["payload"]["observation"]["document_ids"] == [first.document_id]
    markdown = scope_api.client.get(f"/runs/{scoped.run_id}/export?format=markdown").text
    assert "## Document scope" in markdown
    assert first.document_id in markdown
    assert "EXCLUDED_MARKER" not in markdown
    with pytest.raises(ValidationError, match="frozen"):
        scoped.observation.document_ids = (second.document_id,)


def test_scope_reaches_each_real_subtask_and_each_retrieval_branch(
    scope_api: ScopeAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph_paper = scope_api.ingest(
        "Selected graph note", "GraphRAG connects Retrieval in synthetic evidence."
    )
    lexical_paper = scope_api.ingest(
        "Selected lexical note", "graphrag versus retrieval; synthetic lexical evidence."
    )
    scope_api.ingest("Excluded graph note", "GraphRAG connects Retrieval with EXCLUDED_MARKER.")
    hybrid = scope_api.container.hybrid_retriever
    multi = scope_api.container.runner._executor._multihop_retriever
    calls = []
    for component, method in (
        (hybrid._hyde_expander, "expand"),
        (hybrid._dense_retriever, "retrieve"),
        (hybrid._sparse_retriever, "retrieve"),
        (multi, "retrieve"),
    ):
        spy = AsyncMock(wraps=getattr(component, method))
        monkeypatch.setattr(component, method, spy)
        calls.append(spy)
    rrf = Mock(wraps=reciprocal_rank_fusion)
    monkeypatch.setattr("retrieval.hybrid.reciprocal_rank_fusion", rrf)
    selected = [graph_paper.document_id, lexical_paper.document_id]
    for query in (
        "compare GraphRAG versus Retrieval",
        "validate the hypothesis about GraphRAG versus Retrieval",
    ):
        result = scope_api.query(query, document_ids=selected)
        assert result.state == AgentState.DONE, result.error
        assert result.plan is not None
        assert len(result.plan.tasks) == 2
        bundle = scope_api.export(result.run_id)
        assert source_documents(bundle) == set(selected)
        assert {source.path[-1] for source in bundle.snapshot.sources} == {"rrf", "multihop"}
        assert "EXCLUDED_MARKER" not in bundle.snapshot.request.context
    assert all(spy.await_count == 4 for spy in calls)
    for spy in calls[1:]:
        assert all(call.kwargs["document_ids"] == tuple(selected) for call in spy.await_args_list)
    assert rrf.call_count == 4
    for call in rrf.call_args_list:
        assert all(
            result.chunk.document_id in selected
            for candidates in call.args[0]
            for result in candidates
        )


async def test_api_recovers_allowed_paper_buried_beyond_global_top_k(scope_api: ScopeAPI) -> None:
    expanded = await HyDEExpander().expand("retrieval")
    for index in range(6):
        scope_api.ingest(f"Globally higher rank {index}", expanded)
    selected = scope_api.ingest("Selected low rank", "retrieval " + "unrelated " * 60)
    baseline = scope_api.query("retrieval")
    assert selected.document_id not in source_documents(scope_api.export(baseline.run_id))

    result = scope_api.query("retrieval", document_ids=[selected.document_id])
    assert result.state == AgentState.DONE, result.error
    bundle = scope_api.export(result.run_id)
    assert source_documents(bundle) == {selected.document_id}
    assert bundle.snapshot.sources[0].path[-1] == "rrf"
    assert result.answer is not None
    assert [c.document_id for c in result.answer.citations] == [selected.document_id]


@pytest.mark.parametrize("scope", [None, [], "", [" "], [42], ["x" * 129], ["d"] * 101])
def test_api_invalid_scope_returns_422_without_starting_a_run(
    scope_api: ScopeAPI, scope: object
) -> None:
    scope_api.ingest("Real candidate", "GraphRAG has synthetic evidence.")
    before = scope_api.container.event_log.list_events()
    response = scope_api.client.post(
        "/query", json={"query": "GraphRAG evidence", "document_ids": scope}
    )
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"][:2] == ["body", "document_ids"]
    assert scope_api.container.event_log.list_events() == before


def test_unknown_scope_retains_ids_and_zero_evidence_without_fallback(scope_api: ScopeAPI) -> None:
    selected = scope_api.ingest("Existing note", "GraphRAG connects synthetic evidence.")
    missing = scope_api.query("GraphRAG evidence", document_ids=["not-ingested"])
    assert missing.state == AgentState.DONE, missing.error
    bundle = scope_api.export(missing.run_id)
    assert bundle.plan.observation.document_ids == ("not-ingested",)
    assert bundle.snapshot.sources == []
    assert bundle.snapshot.request.context == ""
    assert bundle.answer.citations == []
    assert bundle.answer.ungrounded
    assert bundle.answer.warnings
    mixed = scope_api.query(
        "GraphRAG evidence", document_ids=["not-ingested", selected.document_id]
    )
    assert source_documents(scope_api.export(mixed.run_id)) == {selected.document_id}
    assert scope_api.export(mixed.run_id).plan.observation.document_ids == (
        "not-ingested",
        selected.document_id,
    )


@pytest.mark.parametrize("document_id", ["paper-\u03b2", "https://doi.org/10.1/'quoted'"])
def test_api_can_scope_imported_unicode_and_quoted_ids(
    scope_api: ScopeAPI, document_id: str
) -> None:
    scope_api.container.ingestion_pipeline.ingest_documents(
        [
            Document(
                document_id=document_id,
                title="Imported synthetic note",
                text="GraphRAG connects selected synthetic evidence.",
                source="synthetic:imported",
            )
        ]
    )
    scope_api.ingest("Excluded", "GraphRAG EXCLUDED_MARKER.")
    result = scope_api.query("GraphRAG evidence", document_ids=[document_id])
    assert result.state == AgentState.DONE, result.error
    bundle = scope_api.export(result.run_id)
    assert source_documents(bundle) == {document_id}
    assert bundle.plan.observation.document_ids == (document_id,)
    assert {c.document_id for c in bundle.answer.citations} == {document_id}


async def test_parallel_api_scopes_and_unscoped_run_remain_independent(
    scope_api: ScopeAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    a = scope_api.ingest("A note", "GraphRAG retrieval evidence for A.")
    b = scope_api.ingest("B note", "GraphRAG retrieval evidence for B.")
    entered = asyncio.Event()
    release = asyncio.Event()
    hyde = scope_api.container.hybrid_retriever._hyde_expander
    expand = hyde.expand

    async def hold_first(query: str) -> str:
        if query == "first retrieval":
            entered.set()
            await release.wait()
        return await expand(query)

    monkeypatch.setattr(hyde, "expand", hold_first)
    transport = httpx.ASGITransport(app=scope_api.application)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first = asyncio.create_task(
            client.post(
                "/query", json={"query": "first retrieval", "document_ids": [a.document_id]}
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), timeout=2)
            others = await asyncio.wait_for(
                asyncio.gather(
                    client.post(
                        "/query",
                        json={"query": "second retrieval", "document_ids": [b.document_id]},
                    ),
                    client.post("/query", json={"query": "all retrieval"}),
                ),
                timeout=5,
            )
            assert not first.done()
        finally:
            release.set()
        first_response = await asyncio.wait_for(first, timeout=2)
    for response, selected in zip(
        [first_response, *others],
        [{a.document_id}, {b.document_id}, {a.document_id, b.document_id}],
        strict=True,
    ):
        response.raise_for_status()
        result = AgentRunResult.model_validate(response.json()["result"])
        assert result.state == AgentState.DONE, result.error
        bundle = scope_api.export(result.run_id)
        assert source_documents(bundle) == selected
        assert {c.document_id for c in bundle.answer.citations} == selected
    assert (
        scope_api.export(others[1].json()["result"]["run_id"]).plan.observation.document_ids is None
    )


async def test_runner_snapshots_caller_scope_before_first_await(
    scope_api: ScopeAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    a = scope_api.ingest("A note", "GraphRAG retrieval evidence for A.")
    b = scope_api.ingest("B note", "GraphRAG retrieval evidence for B.")
    entered = asyncio.Event()
    release = asyncio.Event()
    executor = scope_api.container.runner._executor
    retrieve = executor.retrieve

    async def hold_retrieval(
        plan: QueryPlan, max_results: int = 8, *, document_ids: tuple[str, ...] | None = None
    ) -> list[SearchResult]:
        entered.set()
        await release.wait()
        assert document_ids == (a.document_id,)
        assert plan.observation.document_ids == (a.document_id,)
        return await retrieve(plan, max_results, document_ids=document_ids)

    monkeypatch.setattr(executor, "retrieve", hold_retrieval)
    supplied = [a.document_id]
    pending = asyncio.create_task(
        scope_api.container.runner.run("retrieval", document_ids=supplied)
    )
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        supplied[:] = [b.document_id]
    finally:
        release.set()
    result = await asyncio.wait_for(pending, timeout=2)
    assert result.state == AgentState.DONE, result.error
    bundle = scope_api.export(result.run_id)
    assert bundle.plan.observation.document_ids == (a.document_id,)
    assert source_documents(bundle) == {a.document_id}


@pytest.mark.parametrize("scope", [[], [""], ["d"] * 101, "doc-a", {"doc-a"}])
async def test_runner_invalid_scope_is_rejected_before_events(
    scope_api: ScopeAPI, scope: object
) -> None:
    with pytest.raises(ValueError):
        await scope_api.container.runner.run("retrieval", document_ids=scope)
    assert scope_api.container.event_log.list_events() == []


@pytest.mark.parametrize("component", ["executor", "dense", "bm25", "multihop", "graph_store"])
def test_legacy_components_work_unscoped_but_never_drop_requested_scope(
    scope_api: ScopeAPI, monkeypatch: pytest.MonkeyPatch, component: str
) -> None:
    selected = scope_api.ingest("Selected note", "GraphRAG connects synthetic evidence.")
    executor = scope_api.container.runner._executor
    if component == "executor":
        target = executor
    elif component == "dense":
        target = scope_api.container.hybrid_retriever._dense_retriever
    elif component == "bm25":
        target = scope_api.container.hybrid_retriever._sparse_retriever
    elif component == "multihop":
        target = executor._multihop_retriever
    else:
        target = scope_api.container.graph_store
    method = "chunks_for_entities" if component == "graph_store" else "retrieve"
    original = getattr(target, method)
    called = 0

    def legacy(*args: object, **kwargs: object) -> object:
        nonlocal called
        if "document_ids" in kwargs:
            raise TypeError("legacy component does not support document_ids")
        called += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(target, method, legacy)
    result = scope_api.query("GraphRAG evidence")
    assert result.state == AgentState.DONE, result.error
    assert called > 0
    calls_before_scoped = called
    result = scope_api.query("GraphRAG evidence", document_ids=[selected.document_id])
    assert result.state == AgentState.ERROR
    assert result.error is not None and "document_ids" in result.error
    assert called == calls_before_scoped
    assert scope_api.client.get(f"/runs/{result.run_id}/export").status_code == 409


def test_old_executor_signature_is_called_without_new_keywords_for_unscoped_queries(
    scope_api: ScopeAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def retrieve(plan: QueryPlan, max_results: int = 8) -> list[SearchResult]:
        del plan, max_results
        return []

    async def answer(plan: QueryPlan, retrieved: list[SearchResult]) -> AgentAnswer:
        del plan, retrieved
        return AgentAnswer(answer="legacy", citations=[], claims=[])

    executor = scope_api.container.runner._executor
    monkeypatch.setattr(executor, "retrieve", retrieve)
    monkeypatch.setattr(executor, "answer", answer)
    assert scope_api.query("retrieval").state == AgentState.DONE
    assert scope_api.query("retrieval", document_ids=["selected"]).state == AgentState.ERROR


@pytest.mark.parametrize("component", ["executor", "dense", "reranker"])
def test_out_of_scope_custom_results_fail_before_generation(
    scope_api: ScopeAPI, monkeypatch: pytest.MonkeyPatch, component: str
) -> None:
    selected = scope_api.ingest("Selected", "GraphRAG selected evidence.")
    excluded = scope_api.ingest("Excluded", "GraphRAG excluded evidence.")
    excluded_chunk = next(
        c
        for c in scope_api.container.document_store.list_chunks()
        if c.document_id == excluded.document_id
    )

    async def leak(*args: object, **kwargs: object) -> list[SearchResult]:
        del args, kwargs
        return [SearchResult(chunk=excluded_chunk, score=10.0, retriever="broken-custom")]

    executor = scope_api.container.runner._executor
    target = (
        executor
        if component == "executor"
        else scope_api.container.hybrid_retriever._dense_retriever
        if component == "dense"
        else executor._reranker
    )
    monkeypatch.setattr(target, "rerank" if component == "reranker" else "retrieve", leak)
    monkeypatch.setattr(scope_api.container.llm, "generate", no_live_call)
    result = scope_api.query("GraphRAG evidence", document_ids=[selected.document_id])
    assert result.state == AgentState.ERROR
    assert result.error is not None and "outside document_ids" in result.error
    assert not any(
        e["event_type"] == "evidence_snapshot"
        for e in scope_api.container.event_log.list_events(result.run_id)
    )


def test_scope_reloads_with_corpus_and_frozen_exports_survive_later_changes(
    scope_api: ScopeAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected = scope_api.ingest("Selected note", "GraphRAG selected original evidence.")
    scope_api.ingest("Excluded note", "GraphRAG EXCLUDED_MARKER.")
    first = scope_api.query("GraphRAG evidence", document_ids=[selected.document_id])
    assert first.state == AgentState.DONE, first.error
    export_path = f"/runs/{first.run_id}/export"
    before = {
        fmt: scope_api.client.get(export_path, params={"format": fmt}).content
        for fmt in ("json", "markdown")
    }
    restarted = AppContainer(offline_settings(scope_api.database_path))
    scope_api.application.state.container = restarted
    second = scope_api.query("GraphRAG evidence", document_ids=[selected.document_id])
    assert second.state == AgentState.DONE, second.error
    assert source_documents(scope_api.export(second.run_id)) == {selected.document_id}
    assert scope_api.export(second.run_id).plan.observation.document_ids == (selected.document_id,)
    with sqlite3.connect(scope_api.database_path) as connection:
        connection.execute("UPDATE chunks SET text = ?", ("Changed after both queries",))
        connection.execute("DELETE FROM graph_chunks")
        connection.execute("DELETE FROM entity_mentions")
        connection.execute("DELETE FROM entity_edges")
    reopened = AppContainer(offline_settings(scope_api.database_path))
    scope_api.application.state.container = reopened
    monkeypatch.setattr(reopened.llm, "generate", no_live_call)
    monkeypatch.setattr(reopened.hybrid_retriever, "retrieve", no_live_call)
    for fmt in ("json", "markdown"):
        after = scope_api.client.get(export_path, params={"format": fmt})
        assert after.status_code == 200
        assert after.content == before[fmt]


def test_old_unscoped_evidence_events_remain_exportable(scope_api: ScopeAPI) -> None:
    scope_api.ingest("Older note", "GraphRAG synthetic evidence.")
    result = scope_api.query("GraphRAG evidence")
    events = scope_api.container.event_log.list_events(result.run_id)
    for event in events:
        payload = event["payload"]
        if event["event_type"] == "decision_log":
            payload["observation"].pop("document_ids", None)
        elif event["event_type"] == "state_transition":
            payload["payload"].pop("document_ids", None)
            if "observation" in payload["payload"]:
                payload["payload"]["observation"].pop("document_ids", None)
        with sqlite3.connect(scope_api.database_path) as connection:
            connection.execute(
                "UPDATE agent_events SET payload = ? WHERE id = ?",
                (json.dumps(payload), event["id"]),
            )
    bundle = scope_api.export(result.run_id)
    assert bundle.schema_version == "1.0"
    assert bundle.plan.observation.document_ids is None
    assert len(bundle.snapshot.sources) == 1
    old_bundle = bundle.model_dump(mode="json")
    old_bundle["plan"]["observation"].pop("document_ids", None)
    assert EvidenceBundle.model_validate(old_bundle).plan.observation.document_ids is None


@pytest.mark.parametrize("damage", ["start", "plan", "source", "citation"])
def test_export_rejects_inconsistent_saved_document_scope(scope_api: ScopeAPI, damage: str) -> None:
    selected = scope_api.ingest("Selected", "GraphRAG original evidence.")
    result = scope_api.query("GraphRAG evidence", document_ids=[selected.document_id])
    assert result.state == AgentState.DONE, result.error
    events = scope_api.container.event_log.list_events(result.run_id)
    if damage == "start":
        target = events[0]
        target["payload"]["payload"]["document_ids"] = ["different-document"]
    elif damage == "plan":
        target = events[1]
        target["payload"]["observation"]["document_ids"] = ["different-document"]
    elif damage == "source":
        target = next(e for e in events if e["event_type"] == "evidence_snapshot")
        target["payload"]["sources"][0]["chunk"]["document_id"] = "different-document"
    else:
        target = next(
            e
            for e in events
            if e["event_type"] == "state_transition" and e["payload"]["to_state"] == "ANSWERING"
        )
        target["payload"]["payload"]["citations"][0]["document_id"] = "different-document"
    with sqlite3.connect(scope_api.database_path) as connection:
        connection.execute(
            "UPDATE agent_events SET payload = ? WHERE id = ?",
            (json.dumps(target["payload"]), target["id"]),
        )
    response = scope_api.client.get(f"/runs/{result.run_id}/export")
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "invalid_run_record"
