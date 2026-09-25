"""Live providers must not turn unusable HTTP-success bodies into completed answers."""

import json
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient

from api.application import create_app
from config import Settings
from llm.providers import (
    AnthropicAdapter,
    GeminiAdapter,
    HTTPProviderAdapter,
    KimiAdapter,
    OpenAIAdapter,
)
from llm.schemas import LLMRequest, TaskType
from storage.event_log import SQLiteEventLog

PROVIDERS = [OpenAIAdapter, AnthropicAdapter, GeminiAdapter, KimiAdapter]
PRIVATE_MARKER = "synthetic-private-provider-body"
REQUEST = LLMRequest(
    task_type=TaskType.DEFAULT,
    prompt="What connects the synthetic notes?",
    context="[c1] Synthetic graph evidence.",
    citation_chunk_ids=["c1"],
)


def mock_http(
    monkeypatch: pytest.MonkeyPatch, response: httpx.Response
) -> tuple[list[httpx.Request], AsyncMock]:
    original_client = httpx.AsyncClient
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return response

    def create_client(*, timeout: float) -> httpx.AsyncClient:
        return original_client(
            timeout=timeout, transport=httpx.MockTransport(respond), trust_env=False
        )

    sleep = AsyncMock()
    monkeypatch.setattr("llm.providers.httpx.AsyncClient", create_client)
    monkeypatch.setattr("llm.rate_limit.asyncio.sleep", sleep)
    return requests, sleep


def answer_payload(text: object) -> dict[str, object]:
    return {
        "choices": [{"message": {"content": text}}],
        "content": [{"type": "text", "text": text}],
        "candidates": [{"content": {"parts": [{"text": text}]}}],
    }


INVALID_BODIES: list[object] = [
    {},
    {"choices": [], "content": [], "candidates": []},
    {"choices": [None], "content": None, "candidates": [None]},
    {"choices": [{}], "content": [{}], "candidates": [{}]},
    answer_payload(""),
    answer_payload(" \n\t\u2003"),
    answer_payload(None),
    answer_payload(42),
    {
        "choices": [{"message": {"content": None, "reasoning_content": PRIVATE_MARKER}}],
        "content": [{"type": "thinking", "thinking": PRIVATE_MARKER}],
        "candidates": [{"content": {"parts": [{"thought": True, "text": PRIVATE_MARKER}]}}],
    },
    {
        "choices": [{"message": {"tool_calls": [{"name": PRIVATE_MARKER}]}}],
        "content": [{"type": "tool_use", "name": PRIVATE_MARKER}],
        "candidates": [{"content": {"parts": [{"functionCall": {"name": PRIVATE_MARKER}}]}}],
    },
    {"error": {"message": PRIVATE_MARKER}},
    None,
    [],
    PRIVATE_MARKER,
    123,
]


@pytest.mark.parametrize("provider", PROVIDERS, ids=lambda provider: provider.provider_name)
@pytest.mark.parametrize("body", INVALID_BODIES)
async def test_unusable_success_response_is_an_explicit_nonretryable_error(
    provider: type[HTTPProviderAdapter], body: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests, sleep = mock_http(monkeypatch, httpx.Response(200, content=json.dumps(body)))
    with pytest.raises(ValueError, match=f"{provider.provider_name} returned an invalid response"):
        await provider(api_key="dummy-offline-key").generate(REQUEST)
    assert len(requests) == 1
    sleep.assert_not_awaited()


@pytest.mark.parametrize("provider", PROVIDERS, ids=lambda provider: provider.provider_name)
async def test_invalid_json_response_is_sanitized(
    provider: type[HTTPProviderAdapter], monkeypatch: pytest.MonkeyPatch
) -> None:
    requests, sleep = mock_http(
        monkeypatch, httpx.Response(200, text=f"<html>{PRIVATE_MARKER}</html>")
    )
    with pytest.raises(
        ValueError, match=f"{provider.provider_name} returned an invalid response"
    ) as caught:
        await provider(api_key="dummy-offline-key").generate(REQUEST)
    assert PRIVATE_MARKER not in str(caught.value)
    assert "dummy-offline-key" not in str(caught.value)
    assert len(requests) == 1
    sleep.assert_not_awaited()


@pytest.mark.parametrize("provider", PROVIDERS, ids=lambda provider: provider.provider_name)
async def test_valid_answer_keeps_text_citations_and_provenance(
    provider: type[HTTPProviderAdapter], monkeypatch: pytest.MonkeyPatch
) -> None:
    text = "  Synthetic graph evidence [c1].\n"
    requests, sleep = mock_http(monkeypatch, httpx.Response(200, json=answer_payload(text)))
    response = await provider(api_key="dummy-offline-key", model="configured-model").generate(
        REQUEST
    )
    assert response.text == text
    assert response.citation_chunk_ids == ["c1"]
    assert response.raw_provider == provider.provider_name
    assert response.model_name == "configured-model"
    assert len(requests) == 1
    sleep.assert_not_awaited()


@pytest.mark.parametrize("provider", PROVIDERS, ids=lambda provider: provider.provider_name)
def test_empty_response_fails_the_run_without_persisting_a_completed_answer(
    provider: type[HTTPProviderAdapter], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    database_path = tmp_path / "provider-errors.sqlite3"
    settings = Settings(
        _env_file=None,
        database_path=database_path,
        OPENAI_API_KEY="",
        ANTHROPIC_API_KEY="",
        GEMINI_API_KEY="",
        MOONSHOT_API_KEY="",
        SEMANTIC_SCHOLAR_API_KEY="",
    )
    app = create_app(settings)
    app.state.container.runner._executor._llm = provider(api_key="dummy-offline-key")
    requests, sleep = mock_http(
        monkeypatch, httpx.Response(200, json={"error": {"message": PRIVATE_MARKER}})
    )
    with TestClient(app) as client:
        response = client.post("/query", json={"query": REQUEST.prompt})
        assert response.status_code == 200
        result = response.json()["result"]
        assert result["state"] == "ERROR"
        assert result["answer"] is None
        assert f"{provider.provider_name} returned an invalid response" in result["error"]
        events = SQLiteEventLog(database_path).list_events(result["run_id"])
        assert events[-1]["payload"]["to_state"] == "ERROR"
        assert sum(event["payload"].get("to_state") == "ERROR" for event in events) == 1
        assert not any(event["event_type"] == "generation_record" for event in events)
        assert not any(event["payload"].get("to_state") == "DONE" for event in events)
        assert PRIVATE_MARKER not in json.dumps(events)
        assert PRIVATE_MARKER not in response.text
        assert "dummy-offline-key" not in json.dumps(events)
    with TestClient(create_app(settings)) as restarted:
        assert restarted.get(f"/runs/{result['run_id']}/events").json() == events
        exported = restarted.get(f"/runs/{result['run_id']}/export")
        assert exported.status_code == 409
    assert len(requests) == 1
    sleep.assert_not_awaited()
