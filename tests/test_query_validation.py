"""Query boundaries reject invalid input before work without rewriting valid questions."""

import json
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
# Raw JSON bodies: Python json.dumps emits NaN/Infinity by default, and JavaScript
# JSON.stringify escapes unpaired surrogates (for example after UTF-16 truncation).
UNENCODABLE_REQUESTS = [
    pytest.param('{"query": "PRIVATE_QUERY_\\ud800"}', "query", id="query-high-surrogate"),
    pytest.param('{"query": "PRIVATE_QUERY_\\udfff"}', "query", id="query-low-surrogate"),
    pytest.param('{"query": NaN}', "query", id="query-nan"),
    pytest.param('{"query": -Infinity}', "query", id="query-negative-infinity"),
    pytest.param(
        '{"query": "GraphRAG", "document_ids": ["PRIVATE_ID_\\ud83d"]}',
        "document_ids",
        id="document-id-surrogate",
    ),
    pytest.param(
        '{"query": "GraphRAG", "document_ids": [Infinity]}', "document_ids", id="document-id-inf"
    ),
    pytest.param(
        '{"query": "GraphRAG", "collection_id": "PRIVATE_COLLECTION_\\udc00"}',
        "collection_id",
        id="collection-id-surrogate",
    ),
    pytest.param(
        '{"query": "GraphRAG", "max_chunks_per_document": NaN}',
        "max_chunks_per_document",
        id="quota-nan",
    ),
    pytest.param(
        '{"query": "GraphRAG", "min_evidence_documents": Infinity}',
        "min_evidence_documents",
        id="minimum-infinity",
    ),
]


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


@pytest.mark.parametrize("endpoint", ["/query", "/retrieve"])
@pytest.mark.parametrize(("body", "field"), UNENCODABLE_REQUESTS)
def test_http_unencodable_input_is_sanitized_422_before_any_work(
    query_app: FastAPI,
    activity_spies: list[Mock],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    endpoint: str,
    body: str,
    field: str,
) -> None:
    container: AppContainer = query_app.state.container
    run = AsyncMock(wraps=container.runner.run)
    preview = AsyncMock(wraps=container.runner.preview)
    monkeypatch.setattr(container.runner, "run", run)
    monkeypatch.setattr(container.runner, "preview", preview)

    with TestClient(query_app, raise_server_exceptions=False) as client:
        response = client.post(
            endpoint,
            content=body.encode("ascii"),
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422, response.text
        assert client.get("/runs").json() == {"runs": [], "next_cursor": None}

    errors = response.json()["detail"]
    assert errors
    assert all(error["loc"][:2] == ["body", field] for error in errors)
    assert all(set(error) == {"loc", "type", "msg"} for error in errors)
    assert response.headers["Cache-Control"] == "no-store"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    json.dumps(response.json(), allow_nan=False).encode("utf-8")
    assert "PRIVATE" not in response.text
    assert "PRIVATE" not in caplog.text
    assert "unserializable_diagnostic" in caplog.text
    for spy in [run, preview, *activity_spies]:
        spy.assert_not_called()
    assert container.event_log.list_events() == []


@pytest.mark.parametrize("endpoint", ["/query", "/retrieve"])
def test_http_unencodable_query_keeps_other_validation_contracts(
    query_app: FastAPI, activity_spies: list[Mock], endpoint: str
) -> None:
    ordinary = {"min_evidence_documents": 0, "max_chunks_per_document": 0}
    with TestClient(query_app, raise_server_exceptions=False) as client:
        control = client.post(endpoint, json={"query": "GraphRAG", **ordinary})
        response = client.post(
            endpoint,
            content=json.dumps(
                {"query": "PRIVATE_QUERY_\ud800", **ordinary, "near_duplicate_threshold": "0.8"}
            ).encode("ascii"),
            headers={"Content-Type": "application/json"},
        )

    assert control.status_code == 422, control.text
    assert response.status_code == 422, response.text
    errors = response.json()["detail"]
    # Ordinary diagnostics keep FastAPI's normal input/ctx details.
    expected = control.json()["detail"]
    assert any("input" in error and "ctx" in error for error in expected)
    assert [error for error in errors if error["loc"][1] in ordinary] == expected
    assert [error for error in errors if error["loc"][1] == "query"] == [
        {
            "loc": ["body", "query"],
            "type": "string_unicode",
            "msg": "Input should be a valid string, unable to parse raw data as a unicode string",
        }
    ]
    threshold = [error for error in errors if error["loc"][1] == "near_duplicate_threshold"]
    assert threshold
    assert all(set(error) == {"loc", "type", "msg"} for error in threshold)
    assert "PRIVATE" not in response.text
    for spy in activity_spies:
        spy.assert_not_called()


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
