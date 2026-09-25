"""Acceptance tests for exact, read-only comparisons of completed saved runs."""

import importlib
import json
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn
from urllib.parse import quote

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from scripts.demo_evidence_export import offline_settings

from agent.evidence import EvidenceSnapshot, GenerationRecord, RunConfiguration, text_digest
from agent.models import (
    AgentAnswer,
    AgentState,
    Citation,
    Claim,
    QueryIntent,
    QueryObservation,
    QueryPlan,
    StateTransition,
)
from api.application import create_app
from api.dependencies import AppContainer
from llm.schemas import TaskType
from retrieval.models import Chunk, SearchResult

QUERY = "  Inspect synthetic evidence \u7814\u7a76  "
PRIVATE = "synthetic-private-provider-diagnostic"


def _no_work(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("Comparison must not retrieve, generate, write events, or use the corpus")


@dataclass
class ComparisonAPI:
    application: FastAPI
    client: TestClient
    container: AppContainer
    path: Path


@pytest.fixture
def comparison_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[ComparisonAPI]:
    path = tmp_path / "comparison.sqlite3"
    application = create_app(offline_settings(path))
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _no_work)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", _no_work)
    with TestClient(application) as client:
        yield ComparisonAPI(application, client, application.state.container, path)


def _source(chunk_id: str, document_id: str | None = None, **changes: object) -> SearchResult:
    chunk = Chunk(
        chunk_id=chunk_id,
        document_id=document_id or f"doc-{chunk_id}",
        title=f"Synthetic title {chunk_id}",
        text=f"Synthetic evidence {chunk_id}, not a scientific finding.",
        source="synthetic:fixture",
        metadata={"fixture": "true", "author": "Synthetic Author"},
    )
    result = SearchResult(chunk=chunk, score=0.5, retriever="fixture", path=["hybrid", chunk_id])
    for key, value in changes.items():
        if key in {"score", "retriever", "path"}:
            setattr(result, key, value)
        else:
            setattr(chunk, key, value)
    return result


def _save(
    api: ComparisonAPI,
    run_id: str,
    sources: list[SearchResult] | None = None,
    *,
    query: str = QUERY,
    document_ids: tuple[str, ...] | None = None,
    configuration: RunConfiguration | None = None,
    provider: str = "fake",
    model_name: str | None = None,
    task_type: TaskType = TaskType.REASONING,
    answer: AgentAnswer | None = None,
    proposed: list[list[str]] | None = None,
    diagnostics: bool = False,
) -> None:
    """Persist valid synthetic fixtures through the real capture/event/export contracts."""
    selected = sources if sources is not None else [_source("one")]
    snapshot = EvidenceSnapshot.capture(query.strip(), selected)
    snapshot.request.task_type = task_type
    plan = QueryPlan(
        run_id=run_id,
        observation=QueryObservation(
            original_query=query.strip(),
            intent=QueryIntent.FACTUAL_LOOKUP,
            document_ids=document_ids,
        ),
        tasks=[],
        rationale_trace=["Synthetic fixture, not a live research finding."],
    )
    if answer is None:
        answer = AgentAnswer(
            answer="Synthetic saved answer, not a scientific finding.",
            claims=[
                Claim(
                    text="Synthetic saved claim.",
                    chunk_ids=[source.chunk.chunk_id for source in selected],
                    grounded=bool(selected),
                )
            ],
            citations=[
                Citation(
                    chunk_id=source.chunk.chunk_id,
                    document_id=source.chunk.document_id,
                    title=source.chunk.title,
                    snippet=source.chunk.text[:240],
                )
                for source in selected
            ],
            ungrounded=not selected,
        )
    generation = GenerationRecord(
        provider=provider,
        model_name=model_name,
        task_type=task_type,
        claim_chunk_ids=proposed if proposed is not None else [c.chunk_ids for c in answer.claims],
    )
    config = configuration or RunConfiguration(
        max_source_docs=50,
        max_hops=5,
        retrieval_timeout_seconds=30,
        reasoning_timeout_seconds=60,
    )
    log = api.container.event_log

    def transition(source: AgentState, target: AgentState, payload: dict[str, Any]) -> None:
        log.append_transition(
            StateTransition(
                agent_id="synthetic-agent",
                run_id=run_id,
                from_state=source,
                to_state=target,
                payload=payload,
            )
        )

    transition(
        AgentState.IDLE,
        AgentState.PLANNING,
        {
            "query": query,
            "document_ids": document_ids,
            "configuration": config.model_dump(mode="json"),
        },
    )
    log.append_event("synthetic-agent", run_id, "decision_log", plan.model_dump(mode="json"))
    transition(AgentState.PLANNING, AgentState.RETRIEVING, plan.model_dump(mode="json"))
    transition(
        AgentState.RETRIEVING,
        AgentState.REASONING,
        {"chunk_ids": snapshot.request.citation_chunk_ids},
    )
    log.append_event(
        "synthetic-agent", run_id, "evidence_snapshot", snapshot.model_dump(mode="json")
    )
    log.append_event(
        "synthetic-agent", run_id, "generation_record", generation.model_dump(mode="json")
    )
    if diagnostics:
        log.append_event(
            "synthetic-agent",
            run_id,
            "private_diagnostics",
            {"api_key": PRIVATE, "provider_thinking": PRIVATE, "headers": {"secret": PRIVATE}},
        )
    transition(AgentState.REASONING, AgentState.ANSWERING, answer.model_dump(mode="json"))
    transition(AgentState.ANSWERING, AgentState.DONE, {"ungrounded": answer.ungrounded})
    assert api.container.evidence_exporter.export(run_id).snapshot == snapshot


def _url(baseline: str, candidate: str) -> str:
    return f"/runs/{quote(baseline, safe='')}/compare/{quote(candidate, safe='')}"


def _compare(api: ComparisonAPI, baseline: str, candidate: str) -> dict[str, Any]:
    response = api.client.get(_url(baseline, candidate))
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["content-type"] == "application/json"
    result: dict[str, Any] = response.json()
    assert result["schema_version"] == "1.0"
    return result


def test_same_run_is_a_deterministic_no_change_comparison(comparison_api: ComparisonAPI) -> None:
    api = comparison_api
    _save(api, "baseline")
    before = api.container.event_log.list_events()
    comparison = _compare(api, "baseline", "baseline")
    assert comparison["baseline"] == comparison["candidate"]
    assert comparison["any_changes"] is False
    assert not any(comparison["changes"].values())
    assert comparison["baseline"]["query"]["preview"] == QUERY
    assert comparison["baseline"]["query"]["sha256"] == text_digest(QUERY)
    assert comparison["baseline"]["generation"]["model_name"] is None
    assert comparison["baseline"]["export_url"] == "/runs/baseline/export"
    assert comparison["evidence"]["statistics"] == {
        "baseline_count": 1,
        "candidate_count": 1,
        "shared_count": 1,
        "added_count": 0,
        "removed_count": 0,
        "union_count": 1,
        "identity_jaccard": 1.0,
    }
    assert comparison["evidence"]["added"] == comparison["evidence"]["removed"] == []
    assert not any(comparison["evidence"]["shared"][0]["changes"].values())
    first = api.client.get(_url("baseline", "baseline")).content
    assert api.client.get(_url("baseline", "baseline")).content == first
    assert api.container.event_log.list_events() == before
    assert "fake" in " ".join(comparison["notices"]).lower()
    assert "not" in " ".join(comparison["notices"]).lower()


def test_distinct_runs_with_same_compared_content_ignore_trace_identity(
    comparison_api: ComparisonAPI,
) -> None:
    _save(comparison_api, "first")
    _save(comparison_api, "second")
    result = _compare(comparison_api, "first", "second")
    assert result["any_changes"] is False
    assert result["baseline"]["run_id"] != result["candidate"]["run_id"]


def test_query_scope_config_and_requested_provenance_are_explicit(
    comparison_api: ComparisonAPI,
) -> None:
    api = comparison_api
    _save(api, "baseline")
    configuration = RunConfiguration(
        max_source_docs=12,
        max_hops=2,
        retrieval_timeout_seconds=12,
        reasoning_timeout_seconds=25,
    )
    _save(
        api,
        "candidate",
        query="Another synthetic question",
        document_ids=("doc-one", "unknown-but-selected"),
        configuration=configuration,
        provider="historical-fixture-provider",
        model_name="historical-fixture-model",
        task_type=TaskType.COST,
    )
    result = _compare(api, "baseline", "candidate")
    for field in (
        "query_changed",
        "document_scope_changed",
        "document_scope_membership_changed",
        "configuration_changed",
        "provider_changed",
        "model_changed",
        "task_type_changed",
        "request_changed",
    ):
        assert result["changes"][field] is True, field
    assert result["changes"]["context_changed"] is False
    assert result["baseline"]["document_ids"] is None
    assert result["candidate"]["document_ids"] == ["doc-one", "unknown-but-selected"]
    assert result["candidate"]["configuration"] == configuration.model_dump(mode="json")
    assert result["candidate"]["generation"] == {
        "provider": "historical-fixture-provider",
        "model_name": "historical-fixture-model",
        "task_type": "cost",
    }
    assert any("A/B" in notice for notice in result["notices"])


def test_exact_query_and_scope_order_are_not_normalized_away(comparison_api: ComparisonAPI) -> None:
    _save(comparison_api, "baseline", document_ids=("doc-one", "extra"))
    _save(comparison_api, "candidate", query=QUERY.strip(), document_ids=("extra", "doc-one"))
    result = _compare(comparison_api, "baseline", "candidate")
    assert result["changes"]["query_changed"] is True
    assert result["changes"]["request_changed"] is False
    assert result["changes"]["document_scope_changed"] is True
    assert result["changes"]["document_scope_membership_changed"] is False


@pytest.mark.parametrize(
    ("field", "value", "flag", "context_changed"),
    [
        ("text", "Changed synthetic text \u00e9", "text_changed", True),
        ("title", "Changed synthetic title", "title_changed", True),
        ("metadata", {"fixture": "changed"}, "metadata_changed", False),
        ("source", "synthetic:changed", "source_changed", False),
        ("score", -0.75, "score_changed", False),
        ("path", ["other", "path"], "path_changed", False),
        ("retriever", "other-retriever", "retriever_changed", False),
    ],
)
def test_each_source_field_is_compared_beyond_just_text_digest(
    comparison_api: ComparisonAPI, field: str, value: object, flag: str, context_changed: bool
) -> None:
    api = comparison_api
    _save(api, "baseline")
    _save(api, "candidate", [_source("one", **{field: value})])
    result = _compare(api, "baseline", "candidate")
    assert result["changes"]["evidence_changed"] is True
    assert result["changes"]["context_changed"] is context_changed
    change = result["evidence"]["shared"][0]
    assert change["changes"][flag] is True
    assert change["changes"]["record_changed"] is True
    assert change["baseline"]["record_sha256"] != change["candidate"]["record_sha256"]
    assert change["changes"]["rank_changed"] is False
    assert change["rank_delta"] == 0
    if field != "text":
        assert change["baseline"]["text_sha256"] == change["candidate"]["text_sha256"]
    assert result["any_changes"] is True


def test_source_membership_rank_changes_and_ordering_are_directional(
    comparison_api: ComparisonAPI,
) -> None:
    api = comparison_api
    _save(api, "baseline", [_source(name) for name in ("z-shared", "removed", "a-shared")])
    _save(api, "candidate", [_source(name) for name in ("added", "a-shared", "z-shared")])
    result = _compare(api, "baseline", "candidate")
    evidence = result["evidence"]
    assert [item["chunk_id"] for item in evidence["added"]] == ["added"]
    assert [item["chunk_id"] for item in evidence["removed"]] == ["removed"]
    assert [item["baseline"]["chunk_id"] for item in evidence["shared"]] == [
        "z-shared",
        "a-shared",
    ]
    assert [item["rank_delta"] for item in evidence["shared"]] == [2, -1]
    assert all(item["changes"]["rank_changed"] for item in evidence["shared"])
    assert evidence["statistics"]["identity_jaccard"] == 0.5
    assert result["changes"]["source_membership_changed"] is True
    assert result["changes"]["source_order_changed"] is True
    reverse = _compare(api, "candidate", "baseline")["evidence"]
    assert reverse["added"] == evidence["removed"]
    assert reverse["removed"] == evidence["added"]
    assert [item["rank_delta"] for item in reverse["shared"]] == [1, -2]


def test_chunk_id_reassignment_is_removed_and_added_not_shared(
    comparison_api: ComparisonAPI,
) -> None:
    _save(comparison_api, "baseline", [_source("one", "original-document")])
    _save(comparison_api, "candidate", [_source("one", "different-document")])
    evidence = _compare(comparison_api, "baseline", "candidate")["evidence"]
    assert evidence["shared"] == []
    assert evidence["reassigned_chunk_ids"] == ["one"]
    assert evidence["removed"][0]["document_id"] == "original-document"
    assert evidence["added"][0]["document_id"] == "different-document"
    assert evidence["statistics"]["identity_jaccard"] == 0.0


@pytest.mark.parametrize(("left", "right", "overlap"), [(0, 0, None), (0, 2, 0.0), (2, 0, 0.0)])
def test_empty_evidence_has_no_invented_perfect_overlap(
    comparison_api: ComparisonAPI, left: int, right: int, overlap: float | None
) -> None:
    _save(comparison_api, "baseline", [_source(str(i)) for i in range(left)])
    _save(comparison_api, "candidate", [_source(str(i)) for i in range(right)])
    result = _compare(comparison_api, "baseline", "candidate")
    assert result["evidence"]["statistics"]["identity_jaccard"] == overlap
    assert result["evidence"]["statistics"]["baseline_count"] == left
    assert result["evidence"]["statistics"]["candidate_count"] == right
    json.dumps(result, allow_nan=False)


def test_answer_text_claims_grounding_warnings_and_citations_are_independent(
    comparison_api: ComparisonAPI,
) -> None:
    api = comparison_api
    _save(api, "baseline")
    baseline = api.container.evidence_exporter.export("baseline")
    updates: list[tuple[str, AgentAnswer, list[list[str]] | None, str]] = [
        (
            "text",
            baseline.answer.model_copy(update={"answer": "A changed exact answer."}, deep=True),
            None,
            "answer_text_changed",
        ),
        (
            "claim",
            baseline.answer.model_copy(
                update={"claims": [baseline.answer.claims[0].model_copy(update={"text": "Other"})]},
                deep=True,
            ),
            None,
            "claim_text_changed",
        ),
        (
            "grounding",
            baseline.answer.model_copy(update={"ungrounded": True}, deep=True),
            None,
            "grounding_changed",
        ),
        (
            "warning",
            baseline.answer.model_copy(update={"warnings": ["Saved fixture warning"]}, deep=True),
            None,
            "warnings_changed",
        ),
        (
            "proposed",
            baseline.answer.model_copy(deep=True),
            [["one", "not-in-context"]],
            "citation_associations_changed",
        ),
        (
            "citation",
            baseline.answer.model_copy(
                update={
                    "citations": [
                        baseline.answer.citations[0].model_copy(update={"snippet": "Other"})
                    ]
                },
                deep=True,
            ),
            None,
            "citations_changed",
        ),
        (
            "accepted",
            baseline.answer.model_copy(
                update={"claims": [baseline.answer.claims[0].model_copy(update={"chunk_ids": []})]},
                deep=True,
            ),
            [["one"]],
            "citation_associations_changed",
        ),
    ]
    for label, answer, proposed, flag in updates:
        _save(api, label, answer=answer, proposed=proposed)
        result = _compare(api, "baseline", label)
        assert result["changes"][flag] is True, label
        assert result["changes"]["evidence_changed"] is False, label
        assert result["changes"]["answer_text_changed"] is (label == "text"), label
        assert result["changes"]["answer_changed"] is (label != "proposed"), label
        assert result["any_changes"] is True


def test_rank_only_change_updates_citation_associations(comparison_api: ComparisonAPI) -> None:
    api = comparison_api
    _save(api, "baseline", [_source("one"), _source("two")])
    answer = api.container.evidence_exporter.export("baseline").answer
    _save(api, "candidate", [_source("two"), _source("one")], answer=answer)
    result = _compare(api, "baseline", "candidate")
    assert result["changes"]["answer_changed"] is False
    assert result["changes"]["citation_associations_changed"] is True
    assert result["changes"]["context_changed"] is True


@pytest.mark.parametrize("side", ["baseline", "candidate"])
@pytest.mark.parametrize(
    ("state", "code", "status"),
    [
        (None, "run_not_found", 404),
        (AgentState.REASONING, "run_incomplete", 409),
        (AgentState.ERROR, "run_failed", 409),
        (AgentState.DONE, "snapshot_unavailable", 409),
    ],
)
def test_unexportable_side_fails_whole_comparison(
    comparison_api: ComparisonAPI, side: str, state: AgentState | None, code: str, status: int
) -> None:
    api = comparison_api
    _save(api, "good")
    if state is not None:
        api.container.event_log.append_transition(
            StateTransition(
                agent_id="fixture",
                run_id="bad",
                from_state=AgentState.IDLE,
                to_state=state,
                payload={"error": PRIVATE},
            )
        )
    baseline, candidate = ("bad", "good") if side == "baseline" else ("good", "bad")
    authoritative = api.client.get("/runs/bad/export")
    response = api.client.get(_url(baseline, candidate))
    assert response.status_code == status
    assert response.json() == {"detail": {**authoritative.json()["detail"], "side": side}}
    assert PRIVATE not in response.text
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"


@pytest.mark.parametrize("side", ["baseline", "candidate"])
@pytest.mark.parametrize("damage", ["json", "digest", "version", "score", "sequence"])
def test_corruption_is_a_sanitized_authoritative_conflict(
    comparison_api: ComparisonAPI, side: str, damage: str
) -> None:
    api = comparison_api
    _save(api, "good")
    _save(api, "bad")
    events = api.container.event_log.list_events("bad")
    event = next(event for event in events if event["event_type"] == "evidence_snapshot")
    payload = event["payload"]
    if damage == "digest":
        payload["sources"][0]["chunk"]["text"] = PRIVATE
    elif damage == "version":
        payload["schema_version"] = PRIVATE
    elif damage == "score":
        payload["sources"][0]["score"] = float("nan")
    elif damage == "sequence":
        event = events[-1]
        payload = {**event["payload"], "from_state": "IDLE"}
    with sqlite3.connect(api.path) as connection:
        connection.execute(
            "UPDATE agent_events SET payload = ? WHERE id = ?",
            (PRIVATE if damage == "json" else json.dumps(payload), event["id"]),
        )
    baseline, candidate = ("bad", "good") if side == "baseline" else ("good", "bad")
    response = api.client.get(_url(baseline, candidate))
    assert response.status_code == 409
    assert response.json()["detail"]["side"] == side
    assert response.json()["detail"]["code"] == "invalid_run_record"
    assert PRIVATE not in response.text
    assert "sources" not in response.text


def test_both_invalid_reports_baseline_first(comparison_api: ComparisonAPI) -> None:
    response = comparison_api.client.get(_url("absent-baseline", "absent-candidate"))
    assert response.status_code == 404
    assert response.json()["detail"]["side"] == "baseline"
    assert response.json()["detail"]["code"] == "run_not_found"


def test_pure_comparison_is_detached_and_metadata_key_order_is_not_a_change(
    comparison_api: ComparisonAPI,
) -> None:
    api = comparison_api
    _save(api, "baseline")
    _save(
        api,
        "candidate",
        [_source("one", metadata={"author": "Synthetic Author", "fixture": "true"})],
    )
    module = importlib.import_module("agent.run_comparison")
    baseline = api.container.evidence_exporter.export("baseline")
    candidate = api.container.evidence_exporter.export("candidate")
    before = baseline.model_dump_json(), candidate.model_dump_json()
    result = module.compare_bundles(baseline, candidate)
    assert result.any_changes is False
    assert result.model_dump(mode="json") == _compare(api, "baseline", "candidate")
    assert (baseline.model_dump_json(), candidate.model_dump_json()) == before
    result.baseline.configuration.model_copy(update={"max_hops": 0})
    assert baseline.configuration.max_hops == 5


def test_bounded_summaries_keep_unicode_and_markup_as_json_data(
    comparison_api: ComparisonAPI,
) -> None:
    api = comparison_api
    hostile = "\u7814\u7a76\x00\n`````\n<script>example</script>\n[link](javascript:example)\n"
    query = hostile * 15
    sources = [
        _source(str(i), title=hostile * 10, text="Synthetic long text. " * 100) for i in range(50)
    ]
    _save(api, 'quote"\r\nX-Test: example', sources, query=query, diagnostics=True)
    result = _compare(api, 'quote"\r\nX-Test: example', 'quote"\r\nX-Test: example')
    preview = result["baseline"]["query"]
    assert preview == {
        "preview": query[:240],
        "truncated": True,
        "characters": len(query),
        "utf8_bytes": len(query.encode("utf-8")),
        "sha256": text_digest(query),
    }
    assert len(result["evidence"]["shared"]) == 50
    assert result["baseline"]["export_url"] == ("/runs/quote%22%0D%0AX-Test%3A%20example/export")
    assert PRIVATE not in json.dumps(result)
    assert "provider_thinking" not in json.dumps(result)
    assert "snapshot" not in result and "events" not in result
    summary = result["evidence"]["shared"][0]["baseline"]
    assert "text" not in summary and "metadata" not in summary and "path" not in summary
    assert summary["title"]["preview"] == (hostile * 10)[:240]
    assert result["baseline"]["answer"]["text"]["truncated"] is False
    assert "x-test" not in api.client.get(_url('quote"\r\nX-Test: example', "absent")).headers


def test_real_query_export_history_and_comparison_survive_corpus_deletion_and_restart(
    comparison_api: ComparisonAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = comparison_api
    ingested = api.client.post(
        "/ingest/text",
        json={
            "title": "Synthetic comparison note",
            "text": "Synthetic data only. GraphRAG connects research evidence.",
            "source": "synthetic:comparison",
        },
    )
    assert ingested.status_code == 200
    first = api.client.post("/query", json={"query": "What does GraphRAG connect?"})
    second = api.client.post(
        "/query",
        json={
            "query": "What synthetic evidence does GraphRAG connect?",
            "document_ids": [ingested.json()["document_id"]],
        },
    )
    assert first.status_code == second.status_code == 200
    assert first.json()["result"]["state"] == second.json()["result"]["state"] == "DONE"
    baseline, candidate = first.json()["result"]["run_id"], second.json()["result"]["run_id"]
    exports = [api.client.get(f"/runs/{run}/export").content for run in (baseline, candidate)]
    result = _compare(api, baseline, candidate)
    assert result["changes"]["query_changed"] is True
    assert result["changes"]["document_scope_changed"] is True
    before = api.client.get(_url(baseline, candidate)).content
    with sqlite3.connect(api.path) as connection:
        connection.execute("UPDATE chunks SET text = 'replacement synthetic text'")
    assert api.client.get(_url(baseline, candidate)).content == before
    with sqlite3.connect(api.path) as connection:
        for table in ("entity_edges", "entity_mentions", "graph_chunks", "chunks", "documents"):
            connection.execute(f"DELETE FROM {table}")  # noqa: S608
    restarted = AppContainer(offline_settings(api.path))
    api.application.state.container = restarted
    assert restarted.document_store.list_chunks() == []
    events = restarted.event_log.list_events()
    for component, method in (
        (restarted.llm, "generate"),
        (restarted.hybrid_retriever, "retrieve"),
        (restarted.document_store, "list_chunks"),
        (restarted.graph_store, "chunks_for_entities"),
        (restarted.runner, "run"),
        (restarted.event_log, "append_event"),
        (restarted.event_log, "append_transition"),
    ):
        monkeypatch.setattr(component, method, _no_work)
    assert api.client.get(_url(baseline, candidate)).content == before
    assert restarted.event_log.list_events() == events
    assert [
        api.client.get(f"/runs/{run}/export").content for run in (baseline, candidate)
    ] == exports
    history = api.client.get("/runs?state=DONE")
    assert history.status_code == 200
    assert [run["run_id"] for run in history.json()["runs"]] == [candidate, baseline]
    schema = api.client.get("/openapi.json").json()
    operation = schema["paths"]["/runs/{baseline_run_id}/compare/{candidate_run_id}"]["get"]
    assert operation["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/RunComparison"
    }
