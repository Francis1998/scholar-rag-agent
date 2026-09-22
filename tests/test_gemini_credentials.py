"""Offline regressions for Gemini credentials in HTTP and durable run errors."""

import json
from collections.abc import Callable
from pathlib import Path
from unittest.mock import AsyncMock, call

import httpx
import pytest
from fastapi.testclient import TestClient

from agent.models import AgentRunResult, AgentState
from api.application import create_app
from api.dependencies import AppContainer
from config import Settings
from llm.providers import GeminiAdapter
from llm.schemas import LLMRequest, TaskType
from storage.event_log import SQLiteEventLog

DUMMY_KEY = "dummy-gemini-key-not-a-real-credential"
MODEL = "gemini-offline-test-model"
ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
REQUEST = LLMRequest(task_type=TaskType.DEFAULT, prompt="Synthetic question.", context="")


def mock_http(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[httpx.Request], httpx.Response],
) -> tuple[list[httpx.Request], AsyncMock]:
    """Use real HTTPX requests with a fresh offline client for every retry."""
    async_client = httpx.AsyncClient
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)

    def create_client(*, timeout: float) -> httpx.AsyncClient:
        return async_client(
            timeout=timeout, transport=httpx.MockTransport(respond), trust_env=False
        )

    sleep = AsyncMock()
    monkeypatch.setattr("llm.providers.httpx.AsyncClient", create_client)
    monkeypatch.setattr("llm.rate_limit.asyncio.sleep", sleep)
    return requests, sleep


def assert_header_transport(requests: list[httpx.Request]) -> None:
    """Verify the actual wire request, not just the adapter's properties."""
    assert requests
    for request in requests:
        assert request.method == "POST"
        assert str(request.url) == ENDPOINT
        assert not request.url.query
        assert DUMMY_KEY not in str(request.url)
        assert request.headers["x-goog-api-key"] == DUMMY_KEY
        assert request.headers["content-type"] == "application/json"
        assert DUMMY_KEY not in request.content.decode()


def test_gemini_endpoint_does_not_contain_credentials() -> None:
    adapter = GeminiAdapter(api_key=DUMMY_KEY, model=MODEL)
    assert adapter.endpoint == ENDPOINT
    assert adapter.headers == {"Content-Type": "application/json", "x-goog-api-key": DUMMY_KEY}


@pytest.mark.parametrize(
    ("status_code", "attempts"),
    [(400, 1), (401, 1), (403, 1), (404, 1), (429, 4), (500, 4), (502, 4), (503, 4), (504, 4)],
)
async def test_gemini_http_errors_do_not_expose_credentials(
    monkeypatch: pytest.MonkeyPatch, status_code: int, attempts: int
) -> None:
    requests, sleep = mock_http(
        monkeypatch, lambda request: httpx.Response(status_code, json={"error": "offline failure"})
    )
    adapter = GeminiAdapter(api_key=DUMMY_KEY, model=MODEL)
    with pytest.raises(httpx.HTTPStatusError) as caught:
        await adapter.generate(REQUEST)

    assert len(requests) == attempts
    assert sleep.await_args_list == [call(delay) for delay in (0.25, 0.5, 1.0)][: attempts - 1]
    assert caught.value.request is requests[-1]
    assert caught.value.response.status_code == status_code
    assert str(status_code) in str(caught.value)
    assert ENDPOINT in str(caught.value)
    assert DUMMY_KEY not in str(caught.value)
    assert DUMMY_KEY not in repr(caught.value)
    assert_header_transport(requests)


@pytest.mark.parametrize("error_type", [httpx.ConnectError, httpx.ReadTimeout])
async def test_gemini_transport_errors_keep_url_diagnostics_without_credentials(
    monkeypatch: pytest.MonkeyPatch, error_type: type[httpx.TransportError]
) -> None:
    failures: list[httpx.TransportError] = []

    def fail(request: httpx.Request) -> httpx.Response:
        error = error_type(f"Offline transport failure for {request.url}", request=request)
        failures.append(error)
        raise error

    requests, sleep = mock_http(monkeypatch, fail)
    with pytest.raises(error_type) as caught:
        await GeminiAdapter(api_key=DUMMY_KEY, model=MODEL).generate(REQUEST)

    assert len(requests) == 4
    assert sleep.await_args_list == [call(0.25), call(0.5), call(1.0)]
    assert caught.value is failures[-1]
    assert ENDPOINT in str(caught.value)
    assert DUMMY_KEY not in str(caught.value)
    assert_header_transport(requests)


@pytest.mark.parametrize("entrypoint", ["runner", "query"])
@pytest.mark.parametrize(("status_code", "attempts"), [(401, 1), (429, 4), (503, 4)])
async def test_gemini_run_errors_are_durable_without_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: str,
    status_code: int,
    attempts: int,
) -> None:
    database_path = tmp_path / "gemini-errors.sqlite3"
    app = create_app(
        Settings(
            _env_file=None,
            database_path=database_path,
            default_model="gemini",
            gemini_model=MODEL,
            OPENAI_API_KEY="",
            ANTHROPIC_API_KEY="",
            GEMINI_API_KEY=DUMMY_KEY,
            MOONSHOT_API_KEY="",
            SEMANTIC_SCHOLAR_API_KEY="",
        )
    )
    container: AppContainer = app.state.container
    requests, _ = mock_http(
        monkeypatch, lambda request: httpx.Response(status_code, json={"error": "offline failure"})
    )
    with TestClient(app) as client:
        if entrypoint == "runner":
            result = await container.runner.run("What does synthetic GraphRAG connect?")
        else:
            response = client.post(
                "/query", json={"query": "What does synthetic GraphRAG connect?"}
            )
            assert response.status_code == 200
            result = AgentRunResult.model_validate(response.json()["result"])

        assert len(requests) == attempts
        assert result.state == AgentState.ERROR
        assert result.answer is None
        assert result.error is not None
        assert str(status_code) in result.error
        assert ENDPOINT in result.error
        events = SQLiteEventLog(database_path).list_events(result.run_id)
        terminal = events[-1]["payload"]
        assert terminal == {
            "from_state": "REASONING",
            "to_state": "ERROR",
            "payload": {"error": result.error},
        }
        assert sum(event["payload"].get("to_state") == "ERROR" for event in events) == 1
        events_response = client.get(f"/runs/{result.run_id}/events")
        assert events_response.status_code == 200
        assert events_response.json() == events
        assert DUMMY_KEY not in json.dumps(terminal)
        assert DUMMY_KEY not in json.dumps(events)
        assert DUMMY_KEY not in result.model_dump_json()
        assert DUMMY_KEY not in events_response.text
        assert_header_transport(requests)
