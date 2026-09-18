"""Offline, end-to-end regressions for durable generation-time evidence."""

import asyncio
import hashlib
import json
import re
import sqlite3
import subprocess
import sys
from collections.abc import Awaitable
from pathlib import Path
from typing import Any, NoReturn
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from agent.evidence import EvidenceSnapshot
from agent.models import AgentAnswer, AgentState, Citation, Claim, QueryPlan, StateTransition
from agent.safety import with_timeout
from api.dependencies import AppContainer
from api.main import app
from config import Settings
from llm.schemas import LLMRequest, LLMResponse
from retrieval.models import Chunk, Document, SearchResult

LONG_TEXT = (
    "Synthetic GraphRAG evidence connects research entities. " * 10
    + "Beyond the old snippet: caf\u00e9, \u03b2, \u7814\u7a76, \U0001f52c.\n"
    + "`````\n# Not a report heading\n<script>alert('example')</script>\n"
    + "[not a link](javascript:example)\n"
)
QUERY = "  What does GraphRAG connect? \u7814\u7a76  "


def offline_settings(database_path: Path) -> Settings:
    """Ignore local credentials even on a developer machine with live accounts."""
    return Settings(
        _env_file=None,
        database_path=database_path,
        default_model="fake",
        OPENAI_API_KEY="",
        ANTHROPIC_API_KEY="",
        GEMINI_API_KEY="",
        MOONSHOT_API_KEY="",
    )


def no_live_call(*args: object, **kwargs: object) -> NoReturn:
    """Fail rather than allow accidental network or regeneration during export."""
    raise AssertionError("Export must not call generation, retrieval, or the network")


@pytest.fixture
def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, AppContainer, Path]:
    """Give each test its own durable store and fail closed on live HTTP."""
    database_path = tmp_path / "evidence.sqlite3"
    container = AppContainer(offline_settings(database_path))
    monkeypatch.setattr(app.state, "container", container)
    monkeypatch.setattr("httpx.AsyncClient.send", no_live_call)
    return TestClient(app), container, database_path


def add_chunk(container: AppContainer, chunk_id: str = "chunk-original") -> Chunk:
    """Index a long exact chunk without the ingestion normalizer changing it."""
    chunk = Chunk(
        chunk_id=chunk_id,
        document_id=f"document-{chunk_id}",
        title="Synthetic <img src=x onerror=example> \u7814\u7a76",
        text=LONG_TEXT,
        source='../../untrusted/"source".md',
        metadata={"author": "Synthetic Author", "license": "fixture-only"},
    )
    container.document_store.add_documents(
        [Document(**chunk.model_dump(exclude={"chunk_id"}))], [chunk]
    )
    container.hybrid_retriever.add_chunks([chunk])
    return chunk


def run_query(client: TestClient, query: str = QUERY) -> dict[str, Any]:
    """Run the real API pipeline and require its existing success contract."""
    response = client.post("/query", json={"query": query})
    assert response.status_code == 200
    result: dict[str, Any] = response.json()["result"]
    assert result["state"] == "DONE", result
    return result


def export_json(client: TestClient, run_id: str) -> dict[str, Any]:
    """Require a successful versioned export instead of an empty fallback."""
    response = client.get(f"/runs/{quote(run_id, safe='')}/export?format=json")
    assert response.status_code == 200, response.text
    payload: dict[str, Any] = response.json()
    return payload


def test_api_exports_exact_evidence_and_readable_markdown(
    api: tuple[TestClient, AppContainer, Path],
) -> None:
    """The export contains full Unicode passages, not 240-character citations."""
    client, container, _ = api
    chunk = add_chunk(container)
    result = run_query(client)
    bundle = export_json(client, result["run_id"])
    snapshot = bundle["snapshot"]
    source = snapshot["sources"][0]

    assert bundle["schema_version"] == "1.0"
    assert bundle["status"] == "DONE"
    assert bundle["query"] == QUERY
    assert bundle["plan"] == result["plan"]
    assert bundle["answer"] == result["answer"]
    assert len(result["answer"]["citations"][0]["snippet"]) == 240
    assert source["chunk"] == chunk.model_dump()
    assert source["rank"] == 1
    assert source["retriever"] == "lexical_rerank"
    assert source["path"]
    assert isinstance(source["score"], float)
    assert source["text_sha256"] == hashlib.sha256(LONG_TEXT.encode()).hexdigest()
    assert snapshot["request"]["prompt"] == QUERY.strip()
    assert snapshot["request"]["context"] == f"[{chunk.chunk_id}] {chunk.title}: {LONG_TEXT}"
    assert (
        snapshot["context_sha256"]
        == hashlib.sha256(snapshot["request"]["context"].encode()).hexdigest()
    )
    assert bundle["generation"]["provider"] == "fake"
    assert bundle["generation"]["model_name"] is None
    assert bundle["claim_evidence"][0]["evidence_ranks"] == [1]
    assert bundle["claim_evidence"][0]["missing_chunk_ids"] == []
    assert bundle["citation_evidence"][0]["evidence_rank"] == 1
    assert [event["event_type"] for event in bundle["events"]] == [
        "state_transition",
        "decision_log",
        "state_transition",
        "state_transition",
        "evidence_snapshot",
        "generation_record",
        "state_transition",
        "state_transition",
    ]
    assert [event["id"] for event in bundle["events"]] == sorted(
        event["id"] for event in bundle["events"]
    )
    markdown = client.get(f"/runs/{result['run_id']}/export?format=markdown")
    assert markdown.status_code == 200
    assert markdown.headers["content-type"] == "text/markdown; charset=utf-8"
    assert markdown.headers["cache-control"] == "no-store"
    assert markdown.headers["x-content-type-options"] == "nosniff"
    assert markdown.text.startswith("# Research evidence bundle\n")
    assert LONG_TEXT in markdown.text
    assert "token overlap" in markdown.text
    assert "Review before sharing" in markdown.text
    assert "``````text\n" in markdown.text
    assert re.fullmatch(
        r'attachment; filename="evidence-[0-9a-f]{16}\.md"',
        markdown.headers["content-disposition"],
    )
    default_export = client.get(f"/runs/{result['run_id']}/export")
    assert default_export.json() == bundle
    assert re.fullmatch(
        r'attachment; filename="evidence-[0-9a-f]{16}\.json"',
        default_export.headers["content-disposition"],
    )


def test_snapshot_is_post_rerank_and_written_before_generation(
    api: tuple[TestClient, AppContainer, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The snapshot is the model input, not guessed from retrieval events."""
    client, container, _ = api
    original = add_chunk(container)
    excluded = add_chunk(container, "excluded")
    final_chunk = original.model_copy(update={"text": LONG_TEXT + "Final reranker addition."})
    seen: list[LLMRequest] = []

    async def rerank(query: str, results: list[SearchResult]) -> list[SearchResult]:
        assert query == QUERY.strip()
        assert {result.chunk.chunk_id for result in results} == {
            original.chunk_id,
            excluded.chunk_id,
        }
        return [
            SearchResult(
                chunk=final_chunk, score=0.625, retriever="fixture_reranker", path=["hybrid"]
            )
        ]

    async def generate(request: LLMRequest) -> LLMResponse:
        events = container.event_log.list_events()
        assert events[-1]["event_type"] == "evidence_snapshot"
        assert events[-1]["payload"]["request"] == request.model_dump(mode="json")
        seen.append(request.model_copy(deep=True))
        final_chunk.text = "Mutation while awaiting the provider must not change grounding."
        return LLMResponse(
            text="GraphRAG connects research entities.",
            parsed_claims=["GraphRAG connects research entities."],
            citation_chunk_ids=request.citation_chunk_ids,
            raw_provider="fixture",
            model_name="synthetic-model-v1",
        )

    monkeypatch.setattr(container.runner._executor._reranker, "rerank", rerank)
    monkeypatch.setattr(container.llm, "generate", generate)
    result = run_query(client)
    bundle = export_json(client, result["run_id"])
    assert bundle["snapshot"]["request"] == seen[0].model_dump(mode="json")
    assert len(bundle["snapshot"]["sources"]) == 1
    source = bundle["snapshot"]["sources"][0]
    assert source["chunk"]["text"] == LONG_TEXT + "Final reranker addition."
    assert source["score"] == 0.625
    assert source["retriever"] == "fixture_reranker"
    assert source["path"] == ["hybrid"]
    assert bundle["answer"]["citations"][0]["snippet"] == LONG_TEXT[:240]
    assert bundle["generation"]["model_name"] == "synthetic-model-v1"


def test_preserves_missing_references_and_unsupported_claim_warning(
    api: tuple[TestClient, AppContainer, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reference resolution is not silently confused with semantic support."""
    client, container, _ = api
    chunk = add_chunk(container)

    async def generate(request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            text="GraphRAG connects research entities. Unicorns teleport.",
            parsed_claims=["GraphRAG connects research entities.", "Unicorns teleport."],
            citation_chunk_ids=[*request.citation_chunk_ids, "not-retrieved"],
            raw_provider="fake",
        )

    monkeypatch.setattr(container.llm, "generate", generate)
    result = run_query(client)
    bundle = export_json(client, result["run_id"])
    assert bundle["answer"] == result["answer"]
    assert bundle["answer"]["ungrounded"] is True
    assert bundle["answer"]["answer"].startswith("[UNGROUNDED]")
    assert bundle["answer"]["warnings"] == ["One or more claims lacked retrieved chunk support."]
    assert bundle["claim_evidence"][0] == {
        "claim_number": 1,
        "proposed_chunk_ids": [chunk.chunk_id, "not-retrieved"],
        "grounded_chunk_ids": [chunk.chunk_id],
        "evidence_ranks": [1],
        "missing_chunk_ids": ["not-retrieved"],
    }
    assert bundle["claim_evidence"][1]["grounded_chunk_ids"] == []
    assert bundle["claim_evidence"][1]["missing_chunk_ids"] == ["not-retrieved"]
    assert any("not-retrieved" in warning for warning in bundle["warnings"])
    markdown = client.get(f"/runs/{result['run_id']}/export?format=markdown").text
    assert "One or more claims lacked retrieved chunk support." in markdown
    assert "not-retrieved" in markdown


def test_exports_survive_corpus_replacement_deletion_and_container_restart(
    api: tuple[TestClient, AppContainer, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Completed artifacts never consult the mutable corpus or rerun a model."""
    client, container, database_path = api
    chunk = add_chunk(container)
    result = run_query(client)
    path = f"/runs/{result['run_id']}/export"
    before = {fmt: client.get(path, params={"format": fmt}) for fmt in ("json", "markdown")}
    assert before["json"].status_code == 200
    changed = chunk.model_copy(update={"text": "The corpus was replaced.", "metadata": {}})
    container.document_store.add_documents([], [changed])
    assert client.get(path).content == before["json"].content
    with sqlite3.connect(database_path) as connection:
        connection.execute("DELETE FROM entity_edges")
        connection.execute("DELETE FROM entity_mentions")
        connection.execute("DELETE FROM graph_chunks")
        connection.execute("DELETE FROM chunks")
        connection.execute("DELETE FROM documents")
    restarted = AppContainer(offline_settings(database_path))
    assert restarted.document_store.list_chunks() == []
    monkeypatch.setattr(app.state, "container", restarted)
    monkeypatch.setattr(restarted.llm, "generate", no_live_call)
    monkeypatch.setattr(restarted.hybrid_retriever, "retrieve", no_live_call)
    monkeypatch.setattr(restarted.document_store, "list_chunks", no_live_call)
    monkeypatch.setattr(restarted.graph_store, "chunks_for_entities", no_live_call)
    for fmt in ("json", "markdown"):
        after = client.get(path, params={"format": fmt})
        assert after.status_code == 200
        assert after.content == before[fmt].content
        assert after.headers["content-disposition"] == before[fmt].headers["content-disposition"]


async def test_concurrent_runs_keep_their_own_context_and_provenance(
    api: tuple[TestClient, AppContainer, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Force overlapping generations on one executor, finishing in reverse order."""
    client, container, _ = api
    first_started = asyncio.Event()
    second_started = asyncio.Event()

    async def retrieve(plan: QueryPlan, max_results: int = 8) -> list[SearchResult]:
        del max_results
        label = plan.observation.original_query
        return [
            SearchResult(
                chunk=Chunk(
                    chunk_id=label,
                    document_id=label,
                    title=label,
                    text=f"{label} synthetic evidence",
                    source="fixture",
                    metadata={"run": label},
                ),
                score=1.0,
                retriever="fixture",
            )
        ]

    async def generate(request: LLMRequest) -> LLMResponse:
        if request.prompt == "alpha":
            first_started.set()
            await second_started.wait()
        else:
            await first_started.wait()
            second_started.set()
        return LLMResponse(
            text=request.prompt,
            parsed_claims=[request.prompt],
            citation_chunk_ids=request.citation_chunk_ids,
            raw_provider=f"fake-{request.prompt}",
        )

    monkeypatch.setattr(container.runner._executor, "retrieve", retrieve)
    monkeypatch.setattr(container.llm, "generate", generate)
    results = await asyncio.wait_for(
        asyncio.gather(container.runner.run("alpha"), container.runner.run("beta")), timeout=5
    )
    assert results[0].run_id != results[1].run_id
    for label, result in zip(("alpha", "beta"), results, strict=True):
        assert result.state == AgentState.DONE
        bundle = export_json(client, result.run_id)
        assert bundle["query"] == label
        assert bundle["generation"]["provider"] == f"fake-{label}"
        assert bundle["snapshot"]["sources"][0]["chunk"]["chunk_id"] == label
        assert bundle["snapshot"]["request"]["context"] == (
            f"[{label}] {label}: {label} synthetic evidence"
        )
        assert all(event["run_id"] == result.run_id for event in bundle["events"])


@pytest.mark.parametrize("fmt", ["json", "markdown"])
def test_unknown_run_is_404(api: tuple[TestClient, AppContainer, Path], fmt: str) -> None:
    client, _, _ = api
    response = client.get(f"/runs/unknown/export?format={fmt}")
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "run_not_found"


@pytest.mark.parametrize("fmt", ["html", "JSON", "", "../json"])
def test_invalid_export_format_is_422(api: tuple[TestClient, AppContainer, Path], fmt: str) -> None:
    client, _, _ = api
    response = client.get("/runs/unknown/export", params={"format": fmt})
    assert response.status_code == 422


@pytest.mark.parametrize(
    ("state", "code"),
    [
        (AgentState.REASONING, "run_incomplete"),
        (AgentState.ERROR, "run_failed"),
        (AgentState.DONE, "snapshot_unavailable"),
    ],
)
def test_nonexportable_runs_are_explicit_conflicts(
    api: tuple[TestClient, AppContainer, Path], state: AgentState, code: str
) -> None:
    client, container, _ = api
    container.event_log.append_transition(
        StateTransition(
            agent_id="legacy-agent",
            run_id="legacy",
            from_state=AgentState.IDLE,
            to_state=state,
            payload={},
        )
    )
    for fmt in ("json", "markdown"):
        response = client.get(f"/runs/legacy/export?format={fmt}")
        assert response.status_code == 409
        assert response.json()["detail"]["code"] == code


def test_empty_context_is_recorded_without_fabricating_sources(
    api: tuple[TestClient, AppContainer, Path],
) -> None:
    client, _, _ = api
    result = run_query(client)
    bundle = export_json(client, result["run_id"])
    assert bundle["snapshot"]["sources"] == []
    assert bundle["snapshot"]["request"]["context"] == ""
    assert bundle["claim_evidence"][0]["proposed_chunk_ids"] == []
    assert bundle["answer"]["ungrounded"] is True


def test_generation_failure_retains_snapshot_but_cannot_be_exported(
    api: tuple[TestClient, AppContainer, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, container, _ = api
    add_chunk(container)

    async def fail_generation(request: LLMRequest) -> LLMResponse:
        del request
        raise RuntimeError("synthetic provider failure")

    monkeypatch.setattr(container.llm, "generate", fail_generation)
    result = client.post("/query", json={"query": QUERY}).json()["result"]
    assert result["state"] == "ERROR"
    events = container.event_log.list_events(result["run_id"])
    assert [event["event_type"] for event in events][-2:] == [
        "evidence_snapshot",
        "state_transition",
    ]
    assert events[-1]["payload"]["to_state"] == "ERROR"
    response = client.get(f"/runs/{result['run_id']}/export")
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "run_failed"


@pytest.mark.parametrize(
    ("text", "metadata", "message"),
    [
        ("\U0001f52c" * 65536, {}, "context exceeds"),
        ("GraphRAG", {"large": "x" * 1048576}, "snapshot exceeds"),
    ],
    ids=["context-byte-limit", "snapshot-byte-limit"],
)
def test_oversized_context_or_metadata_fails_before_generation(
    api: tuple[TestClient, AppContainer, Path],
    monkeypatch: pytest.MonkeyPatch,
    text: str,
    metadata: dict[str, str],
    message: str,
) -> None:
    client, container, _ = api
    chunk = add_chunk(container)
    oversized = chunk.model_copy(update={"text": text, "metadata": metadata})

    async def retrieve(plan: QueryPlan, max_results: int = 8) -> list[SearchResult]:
        del plan, max_results
        return [SearchResult(chunk=oversized, score=1.0, retriever="fixture")]

    monkeypatch.setattr(container.runner._executor, "retrieve", retrieve)
    monkeypatch.setattr(container.llm, "generate", no_live_call)
    result = client.post("/query", json={"query": QUERY}).json()["result"]
    assert result["state"] == "ERROR"
    assert message in result["error"]
    response = client.get(f"/runs/{result['run_id']}/export")
    assert response.status_code == 409


def test_export_whitelists_operational_payloads_and_never_serializes_settings(
    api: tuple[TestClient, AppContainer, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, container, _ = api
    add_chunk(container)
    sentinel = "synthetic-not-a-real-key"
    original_generate = container.llm.generate

    async def generate(request: LLMRequest) -> LLMResponse:
        events = container.event_log.list_events()
        container.event_log.append_event(
            agent_id="local-agent",
            run_id=events[-1]["run_id"],
            event_type="private_diagnostics",
            payload={
                "api_key": sentinel,
                "headers": {"Authorization": sentinel},
                "environment": {"OPENAI_API_KEY": sentinel},
                "provider_thinking": "Do not export hidden thinking",
            },
        )
        return await original_generate(request)

    monkeypatch.setenv("OPENAI_API_KEY", sentinel)
    monkeypatch.setattr(container.llm, "generate", generate)
    result = run_query(client)
    bundle = export_json(client, result["run_id"])
    exported = json.dumps(bundle)
    assert sentinel not in exported
    assert "provider_thinking" not in exported
    assert "Authorization" not in exported
    omitted = [event for event in bundle["events"] if event["payload_omitted"]]
    assert len(omitted) == 1
    assert omitted[0]["event_type"] == "private_diagnostics"
    assert omitted[0]["payload"] is None
    assert any("omitted" in warning for warning in bundle["warnings"])


def test_download_filename_does_not_interpolate_untrusted_run_id(
    api: tuple[TestClient, AppContainer, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, container, _ = api
    add_chunk(container)
    hostile_id = 'run"\r\nX-Injected: example-\u7814\u7a76'
    monkeypatch.setattr("agent.runner.uuid4", lambda: hostile_id)
    result = run_query(client)
    response = client.get(f"/runs/{quote(result['run_id'], safe='')}/export")
    assert response.status_code == 200
    assert "x-injected" not in response.headers
    assert re.fullmatch(
        r'attachment; filename="evidence-[0-9a-f]{16}\.json"',
        response.headers["content-disposition"],
    )


def test_events_after_done_do_not_change_a_completed_artifact(
    api: tuple[TestClient, AppContainer, Path],
) -> None:
    client, container, _ = api
    add_chunk(container)
    result = run_query(client)
    path = f"/runs/{result['run_id']}/export"
    before = client.get(path)
    assert before.status_code == 200
    container.event_log.append_event(
        agent_id="local-agent",
        run_id=result["run_id"],
        event_type="later_audit_note",
        payload={"note": "The completed artifact is bounded by its first terminal event."},
    )
    assert client.get(path).content == before.content


def test_capture_enforces_exact_byte_and_source_thresholds() -> None:
    """Test the actual UTF-8 and serialized-payload boundaries, not a proxy."""
    result = SearchResult(
        chunk=Chunk(chunk_id="c", document_id="d", title="t", text="", source="fixture"),
        score=1.0,
        retriever="fixture",
    )
    limit = 262144
    result.chunk.text = "x" * (limit - len(b"[c] t: "))
    snapshot = EvidenceSnapshot.capture("query", [result])
    assert len(snapshot.request.context.encode("utf-8")) == limit
    result.chunk.text += "x"
    with pytest.raises(ValidationError, match="context exceeds"):
        EvidenceSnapshot.capture("query", [result])

    result.chunk.text = "small"
    result.chunk.metadata = {"padding": ""}
    snapshot = EvidenceSnapshot.capture("query", [result])
    size = len(json.dumps(snapshot.model_dump(mode="json"), sort_keys=True).encode("utf-8"))
    result.chunk.metadata["padding"] = "x" * (1048576 - size)
    snapshot = EvidenceSnapshot.capture("query", [result])
    assert (
        len(json.dumps(snapshot.model_dump(mode="json"), sort_keys=True).encode("utf-8")) == 1048576
    )
    result.chunk.metadata["padding"] += "x"
    with pytest.raises(ValidationError, match="snapshot exceeds"):
        EvidenceSnapshot.capture("query", [result])

    result.chunk.metadata = {}
    results = []
    for index in range(50):
        copy = result.model_copy(deep=True)
        copy.chunk.chunk_id = f"chunk-{index}"
        results.append(copy)
    assert len(EvidenceSnapshot.capture("query", results).sources) == 50
    with pytest.raises(ValueError, match="exceeds 50 sources"):
        EvidenceSnapshot.capture("query", [*results, result])


@pytest.mark.parametrize("damage", ["digest", "version", "plan", "generation", "sequence", "json"])
def test_corrupt_or_unsupported_saved_records_are_not_fabricated(
    api: tuple[TestClient, AppContainer, Path], damage: str
) -> None:
    client, container, database_path = api
    add_chunk(container)
    result = run_query(client)
    events = container.event_log.list_events(result["run_id"])
    target = next(event for event in events if event["event_type"] == "evidence_snapshot")
    payload = target["payload"]
    if damage == "digest":
        payload["sources"][0]["chunk"]["text"] = "Changed without updating the digest."
    elif damage == "version":
        payload["schema_version"] = "999"
    elif damage == "plan":
        target = next(event for event in events if event["event_type"] == "decision_log")
        payload = {**target["payload"], "run_id": "a-different-run"}
    elif damage == "sequence":
        target = events[-1]
        payload = {**target["payload"], "from_state": "IDLE"}
    with sqlite3.connect(database_path) as connection:
        if damage == "generation":
            connection.execute(
                "DELETE FROM agent_events WHERE run_id = ? AND event_type = 'generation_record'",
                (result["run_id"],),
            )
        else:
            connection.execute(
                "UPDATE agent_events SET payload = ? WHERE id = ?",
                ("{invalid JSON" if damage == "json" else json.dumps(payload), target["id"]),
            )
    response = client.get(f"/runs/{result['run_id']}/export")
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "invalid_run_record"


def test_snapshot_write_failure_prevents_generation(
    api: tuple[TestClient, AppContainer, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Do not generate an apparently exportable answer if durable capture fails."""
    client, container, _ = api
    add_chunk(container)
    append = container.event_log.append_event

    def fail_snapshot(agent_id: str, run_id: str, event_type: str, payload: dict[str, Any]) -> int:
        if event_type == "evidence_snapshot":
            raise sqlite3.OperationalError("synthetic snapshot write failure")
        return append(agent_id, run_id, event_type, payload)

    monkeypatch.setattr(container.event_log, "append_event", fail_snapshot)
    monkeypatch.setattr(container.llm, "generate", no_live_call)
    result = client.post("/query", json={"query": QUERY}).json()["result"]
    assert result["state"] == "ERROR"
    assert result["error"] == "synthetic snapshot write failure"
    assert client.get(f"/runs/{result['run_id']}/export").status_code == 409


def test_missing_final_citations_and_unknown_model_identity_are_explicit(
    api: tuple[TestClient, AppContainer, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, container, _ = api
    add_chunk(container)

    async def generate(request: LLMRequest) -> LLMResponse:
        return LLMResponse(text="GraphRAG.", citation_chunk_ids=request.citation_chunk_ids)

    def ground(answer_text: str, claims: list[Claim], retrieved_chunks: list[Chunk]) -> AgentAnswer:
        del claims, retrieved_chunks
        return AgentAnswer(
            answer=answer_text,
            claims=[Claim(text="GraphRAG.", chunk_ids=["missing-final"])],
            citations=[
                Citation(chunk_id="missing-final", document_id="missing", title="?", snippet="?")
            ],
            ungrounded=True,
            warnings=["Synthetic missing final reference."],
        )

    monkeypatch.setattr(container.llm, "generate", generate)
    monkeypatch.setattr(container.runner._executor._grounder, "ground", ground)
    result = run_query(client)
    bundle = export_json(client, result["run_id"])
    assert bundle["generation"]["provider"] == "unknown"
    assert bundle["generation"]["model_name"] is None
    assert set(bundle["generation"]) == {"provider", "model_name", "task_type", "claim_chunk_ids"}
    assert bundle["citation_evidence"][0]["evidence_rank"] is None
    assert bundle["claim_evidence"][0]["missing_chunk_ids"] == ["missing-final"]
    assert any(
        "Citation 1 references missing evidence" in warning for warning in bundle["warnings"]
    )


def test_openapi_publishes_the_versioned_json_schema(
    api: tuple[TestClient, AppContainer, Path],
) -> None:
    client, _, _ = api
    schema = client.get("/openapi.json").json()
    operation = schema["paths"]["/runs/{run_id}/export"]["get"]
    response = operation["responses"]["200"]
    assert response["content"]["application/json"]["schema"]["$ref"].endswith("/EvidenceBundle")
    assert "text/markdown" in response["content"]
    assert schema["components"]["schemas"]["EvidenceBundle"]["properties"]["schema_version"] == {
        "const": "1.0",
        "default": "1.0",
        "title": "Schema Version",
        "type": "string",
    }


def test_export_reads_only_saved_events_in_a_fresh_process(
    api: tuple[TestClient, AppContainer, Path],
) -> None:
    """No process-local snapshot cache, corpus lookup, or container is needed."""
    client, container, database_path = api
    add_chunk(container)
    result = run_query(client)
    expected = client.get(f"/runs/{result['run_id']}/export")
    assert expected.status_code == 200
    process = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-c",
            "import json, sys\n"
            "from storage.event_log import SQLiteEventLog\n"
            "from storage.evidence_export import EvidenceExporter\n"
            "bundle = EvidenceExporter(SQLiteEventLog(sys.argv[1])).export(sys.argv[2])\n"
            "print(json.dumps(bundle.model_dump(mode='json'), ensure_ascii=False, "
            "sort_keys=True, indent=2))\n",
            str(database_path),
            result["run_id"],
        ],
        check=False,
        capture_output=True,
        timeout=30,
    )
    assert process.returncode == 0, process.stderr.decode("utf-8")
    assert process.stdout == expected.content


def test_legacy_answer_override_still_completes_without_fabricated_evidence(
    api: tuple[TestClient, AppContainer, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, container, _ = api
    calls = 0

    async def legacy_answer(plan: QueryPlan, retrieved: list[SearchResult]) -> AgentAnswer:
        nonlocal calls
        del plan, retrieved
        calls += 1
        return AgentAnswer(answer="Legacy executor answer.", claims=[], citations=[])

    monkeypatch.setattr(container.runner._executor, "answer", legacy_answer)
    result = run_query(client)
    assert result["answer"]["answer"] == "Legacy executor answer."
    assert calls == 1
    response = client.get(f"/runs/{result['run_id']}/export")
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "snapshot_unavailable"


def test_type_error_inside_generation_is_not_retried_as_a_legacy_call(
    api: tuple[TestClient, AppContainer, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, container, _ = api
    calls = 0

    async def generate(request: LLMRequest) -> LLMResponse:
        nonlocal calls
        del request
        calls += 1
        raise TypeError("synthetic provider implementation error")

    monkeypatch.setattr(container.llm, "generate", generate)
    result = client.post("/query", json={"query": QUERY}).json()["result"]
    assert result["state"] == "ERROR"
    assert result["error"] == "synthetic provider implementation error"
    assert calls == 1


def test_effective_run_configuration_is_frozen_allowlisted_and_used(
    api: tuple[TestClient, AppContainer, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, container, _ = api
    add_chunk(container)
    limits = container.runner._safety_limits
    limits.max_source_docs = 2
    limits.max_hops = 2
    limits.retrieval_timeout_seconds = 17
    limits.reasoning_timeout_seconds = 23
    timeouts: list[tuple[str, float]] = []
    retrieve = container.runner._executor.retrieve

    async def change_settings_during_retrieval(
        plan: QueryPlan, max_results: int = 8
    ) -> list[SearchResult]:
        assert max_results == 2
        limits.max_source_docs = 7
        limits.max_hops = 5
        limits.retrieval_timeout_seconds = 8
        limits.reasoning_timeout_seconds = 9
        return await retrieve(plan, max_results)

    async def record_timeout(
        awaitable: Awaitable[object], timeout_seconds: float, label: str
    ) -> object:
        timeouts.append((label, timeout_seconds))
        return await with_timeout(awaitable, timeout_seconds, label)

    monkeypatch.setattr(container.runner._executor, "retrieve", change_settings_during_retrieval)
    monkeypatch.setattr("agent.runner.with_timeout", record_timeout)
    result = run_query(client)
    bundle = export_json(client, result["run_id"])
    assert bundle["configuration"] == {
        "max_source_docs": 2,
        "max_hops": 2,
        "retrieval_timeout_seconds": 17.0,
        "reasoning_timeout_seconds": 23.0,
    }
    assert bundle["events"][0]["payload"]["payload"]["configuration"] == bundle["configuration"]
    assert timeouts == [("retrieval", 17), ("reasoning", 23)]
    assert bundle["snapshot"]["limits"] == {
        "max_sources": 50,
        "max_context_bytes": 262144,
        "max_snapshot_bytes": 1048576,
    }
