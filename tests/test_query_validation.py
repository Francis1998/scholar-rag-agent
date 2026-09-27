"""Query boundaries reject invalid input before work without rewriting valid questions."""

from pathlib import Path
from typing import NoReturn
from unittest.mock import AsyncMock, Mock, call

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from scripts.demo_evidence_export import offline_settings

import agent.runner as runner_module
from agent.models import QueryObservation
from api.application import create_app
from api.dependencies import AppContainer
from api.retrieval import RetrievalRequest
from api.schemas import QueryRequest
from llm.providers import AnthropicAdapter, GeminiAdapter, KimiAdapter, OpenAIAdapter

BLANK_QUERIES = [
    pytest.param("", id="empty"),
    pytest.param("   ", id="spaces"),
    pytest.param("\t\r\n\v\f", id="ascii-whitespace"),
    pytest.param("\u00a0", id="nonbreaking-space"),
    pytest.param("\u2003", id="em-space"),
    pytest.param("\u2028\u2029", id="unicode-line-separators"),
    pytest.param("\u3000", id="ideographic-space"),
    pytest.param(" \t\u00a0\u2003\u3000\n", id="mixed-whitespace"),
]
NONSTRING_QUERIES = [
    pytest.param(None, id="null"),
    pytest.param(12, id="integer"),
    pytest.param(1.5, id="float"),
    pytest.param(False, id="boolean"),
    pytest.param([], id="list"),
    pytest.param({}, id="object"),
]
PADDED_QUERY = "\t \u00a0What  does GraphRAG connect? caf\u00e9 \u7814\u7a76\u2003\n"


def deny_network(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("Query validation tests must not use the network.")


@pytest.fixture
def query_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", deny_network)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", deny_network)
    return create_app(offline_settings(tmp_path / "query-validation.sqlite3"))


@pytest.fixture
def activity_spies(query_app: FastAPI, monkeypatch: pytest.MonkeyPatch) -> list[Mock]:
    container: AppContainer = query_app.state.container
    runner = container.runner
    spies: list[Mock] = []
    for owner, method in (
        (runner_module, "uuid4"),
        (runner_module, "normalize_document_ids"),
        (runner, "_configuration"),
        (runner._analyzer, "analyze"),
        (runner._planner, "plan"),
        (container.event_log, "append_transition"),
        (container.event_log, "append_event"),
    ):
        spy = Mock(name=method, wraps=getattr(owner, method))
        monkeypatch.setattr(owner, method, spy)
        spies.append(spy)
    for owner, method in (
        (runner._executor, "retrieve"),
        (runner._executor, "prepare_context"),
        (runner._executor, "answer"),
        (container.llm, "generate"),
        (container.llm._router._adapters["fake"], "generate"),
    ):
        spy = AsyncMock(name=method, wraps=getattr(owner, method))
        monkeypatch.setattr(owner, method, spy)
        spies.append(spy)
    for adapter in (OpenAIAdapter, AnthropicAdapter, GeminiAdapter, KimiAdapter):
        spy = AsyncMock(name=adapter.__name__, side_effect=deny_network)
        monkeypatch.setattr(adapter, "generate", spy)
        spies.append(spy)
    return spies


@pytest.mark.parametrize("endpoint", ["/query", "/retrieve"])
@pytest.mark.parametrize("query", [*BLANK_QUERIES, *NONSTRING_QUERIES])
def test_http_invalid_query_is_422_before_any_work(
    query_app: FastAPI,
    activity_spies: list[Mock],
    monkeypatch: pytest.MonkeyPatch,
    endpoint: str,
    query: object,
) -> None:
    container: AppContainer = query_app.state.container
    run = AsyncMock(wraps=container.runner.run)
    preview = AsyncMock(wraps=container.runner.preview)
    monkeypatch.setattr(container.runner, "run", run)
    monkeypatch.setattr(container.runner, "preview", preview)

    with TestClient(query_app) as client:
        response = client.post(endpoint, json={"query": query})
        assert response.status_code == 422, response.text
        assert response.json()["detail"][0]["loc"] == ["body", "query"]
        assert client.get("/runs").json() == {"runs": [], "next_cursor": None}

    for spy in [run, preview, *activity_spies]:
        spy.assert_not_called()
    assert container.event_log.list_events() == []


@pytest.mark.parametrize("entrypoint", ["run", "preview"])
@pytest.mark.parametrize(
    "query",
    [
        *BLANK_QUERIES,
        *NONSTRING_QUERIES,
        pytest.param(b"GraphRAG", id="bytes"),
        pytest.param(bytearray(b"GraphRAG"), id="bytearray"),
    ],
)
async def test_python_invalid_query_raises_before_any_work(
    query_app: FastAPI, activity_spies: list[Mock], entrypoint: str, query: object
) -> None:
    container: AppContainer = query_app.state.container
    with pytest.raises(ValueError, match="query must be a nonempty string"):
        await getattr(container.runner, entrypoint)(query)

    for spy in activity_spies:
        spy.assert_not_called()
    assert container.event_log.list_events() == []


@pytest.mark.parametrize("schema", [QueryRequest, RetrievalRequest])
def test_query_schema_keeps_nonblank_text_exactly(schema: type[QueryRequest]) -> None:
    assert schema(query=PADDED_QUERY).query == PADDED_QUERY


@pytest.mark.parametrize("via_http", [True, False], ids=["http", "python"])
@pytest.mark.parametrize("preserve_observation", [False, True], ids=["builtin", "custom-analyzer"])
async def test_padded_query_preserves_boundaries_and_existing_evidence(
    query_app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    via_http: bool,
    preserve_observation: bool,
) -> None:
    container: AppContainer = query_app.state.container
    runner = container.runner
    original_analyze = runner._analyzer.analyze

    def analyze(query: str) -> QueryObservation:
        observation = original_analyze(query)
        if preserve_observation:
            return observation.model_copy(update={"original_query": query})
        return observation

    observe = Mock(wraps=analyze)
    plan = Mock(wraps=runner._planner.plan)
    generate = AsyncMock(wraps=container.llm.generate)
    monkeypatch.setattr(runner._analyzer, "analyze", observe)
    monkeypatch.setattr(runner._planner, "plan", plan)
    monkeypatch.setattr(container.llm, "generate", generate)
    with TestClient(query_app) as client:
        ingested = client.post(
            "/ingest/text",
            json={
                "title": "Synthetic query validation note",
                "text": "GraphRAG connects entities in local synthetic evidence.",
            },
        )
        assert ingested.status_code == 200
        if via_http:
            preview_response = client.post("/retrieve", json={"query": PADDED_QUERY})
            assert preview_response.status_code == 200, preview_response.text
            preview = preview_response.json()
        else:
            preview = (await runner.preview(PADDED_QUERY)).model_dump(mode="json")
        generate.assert_not_called()
        assert container.event_log.list_events() == []

        if via_http:
            response = client.post("/query", json={"query": PADDED_QUERY})
            assert response.status_code == 200, response.text
            result = response.json()["result"]
        else:
            result = (await runner.run(PADDED_QUERY)).model_dump(mode="json")
        assert result["state"] == "DONE", result
        events = container.event_log.list_events(result["run_id"])
        snapshot = next(
            event["payload"] for event in events if event["event_type"] == "evidence_snapshot"
        )
        if not preserve_observation:
            exported = client.get(f"/runs/{result['run_id']}/export")
            assert exported.status_code == 200, exported.text
            assert exported.json()["query"] == PADDED_QUERY
            assert exported.json()["snapshot"] == snapshot

    # The built-in analyzer already normalizes outer whitespace; validation must not.
    expected_query = PADDED_QUERY if preserve_observation else PADDED_QUERY.strip()
    observe.assert_has_calls([call(PADDED_QUERY), call(PADDED_QUERY)])
    assert observe.call_count == 2
    assert plan.call_count == 2
    assert all(args.args[1].original_query == expected_query for args in plan.call_args_list)
    assert result["observation"]["original_query"] == expected_query
    assert result["plan"]["observation"]["original_query"] == expected_query
    assert preview["plan"]["observation"]["original_query"] == expected_query
    assert result["plan"]["tasks"][0]["query"] == expected_query
    assert preview["plan"]["tasks"][0]["query"] == expected_query
    assert snapshot["request"]["prompt"] == expected_query
    generate.assert_awaited_once()
    assert generate.await_args.args[0].prompt == expected_query
    assert events[0]["payload"]["payload"]["query"] == PADDED_QUERY
    assert preview["sources"]
    assert preview["sources"] == snapshot["sources"]
    assert preview["context"] == snapshot["request"]["context"]
    assert preview["context_sha256"] == snapshot["context_sha256"]
