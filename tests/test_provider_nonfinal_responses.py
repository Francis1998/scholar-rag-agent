"""Text-only adapters must not journal unfinished tool turns as final answers."""

import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient

from agent.evidence import EvidenceSnapshot
from api.application import create_app
from config import Settings
from llm.providers import (
    AnthropicAdapter,
    GeminiAdapter,
    HTTPProviderAdapter,
    KimiAdapter,
    OpenAIAdapter,
    ProviderResponseError,
)
from llm.router import ModelRouter, RoutingLLMAdapter
from llm.schemas import LLMRequest, TaskType
from retrieval.models import Chunk, Document
from storage.event_log import SQLiteEventLog

PREAMBLE = "Synthetic private preamble before an unfinished tool call."
TOOL_ARGUMENT = "synthetic-private-tool-argument"
THINKING = "synthetic-private-thinking"
DUMMY_KEY = "offline-dummy-key"
MODEL = "configured-provider-model"
REQUEST = LLMRequest(
    task_type=TaskType.DEFAULT,
    prompt="What does GraphRAG connect?",
    context="[c1] Synthetic evidence: GraphRAG connects research entities.",
    citation_chunk_ids=["c1"],
)


def chat_body(*, reason: object = None, **message: object) -> dict[str, object]:
    choice: dict[str, object] = {
        "message": {"content": PREAMBLE, "reasoning_content": THINKING, **message}
    }
    if reason is not None:
        choice["finish_reason"] = reason
    return {"model": "server-alias", "choices": [choice]}


def claude_body(*, reason: object = None, **block: object) -> dict[str, object]:
    body: dict[str, object] = {
        "content": [
            {"type": "thinking", "thinking": THINKING},
            {"type": "text", "text": PREAMBLE},
            block,
        ]
    }
    if reason is not None:
        body["stop_reason"] = reason
    return body


def gemini_body(*, reason: object = None, **part: object) -> dict[str, object]:
    candidate: dict[str, object] = {
        "content": {"parts": [{"thought": True, "text": THINKING}, {"text": PREAMBLE}, part]}
    }
    if reason is not None:
        candidate["finishReason"] = reason
    return {"modelVersion": "server-alias", "candidates": [candidate]}


NONFINAL_CASES = [
    *[
        pytest.param(
            provider,
            chat_body(reason=reason),
            id=f"{provider.provider_name}-{reason}",
        )
        for provider in (OpenAIAdapter, KimiAdapter)
        for reason in ("tool_calls", "function_call")
    ],
    *[
        pytest.param(
            provider,
            chat_body(
                reason=reason,
                tool_calls=[
                    {
                        "id": "synthetic-call",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": json.dumps({"query": TOOL_ARGUMENT}),
                        },
                    }
                ],
            ),
            id=f"{provider.provider_name}-tool_calls-{reason}",
        )
        for provider in (OpenAIAdapter, KimiAdapter)
        for reason in (None, "stop")
    ],
    *[
        pytest.param(
            provider,
            chat_body(
                reason=reason,
                function_call={
                    "name": "lookup",
                    "arguments": json.dumps({"query": TOOL_ARGUMENT}),
                },
            ),
            id=f"{provider.provider_name}-function_call-{reason}",
        )
        for provider in (OpenAIAdapter, KimiAdapter)
        for reason in (None, "stop")
    ],
    pytest.param(
        OpenAIAdapter,
        chat_body(
            tool_calls=[
                {
                    "id": "synthetic-call",
                    "type": "custom",
                    "custom": {"name": "lookup", "input": TOOL_ARGUMENT},
                }
            ]
        ),
        id="openai-custom-tool",
    ),
    *[
        pytest.param(AnthropicAdapter, claude_body(reason=reason), id=f"anthropic-{reason}")
        for reason in ("tool_use", "pause_turn")
    ],
    *[
        pytest.param(
            AnthropicAdapter,
            claude_body(
                reason=reason,
                type="tool_use",
                id="synthetic-call",
                name="lookup",
                input={"query": TOOL_ARGUMENT},
            ),
            id=f"anthropic-tool-block-{reason}",
        )
        for reason in (None, "end_turn")
    ],
    *[
        pytest.param(GeminiAdapter, gemini_body(reason=reason), id=f"gemini-{reason}")
        for reason in (
            "MALFORMED_FUNCTION_CALL",
            "UNEXPECTED_TOOL_CALL",
            "TOO_MANY_TOOL_CALLS",
        )
    ],
    *[
        pytest.param(
            GeminiAdapter,
            gemini_body(reason=reason, **{field: call}),
            id=f"gemini-{field}-{reason}",
        )
        for field, call in (
            ("functionCall", {"name": "lookup", "args": {"query": TOOL_ARGUMENT}}),
            ("toolCall", {"toolType": "GOOGLE_SEARCH_WEB", "args": {"query": TOOL_ARGUMENT}}),
        )
        for reason in (None, "STOP")
    ],
    pytest.param(
        GeminiAdapter,
        gemini_body(
            thought=True,
            functionCall={"name": "lookup", "args": {"query": TOOL_ARGUMENT}},
        ),
        id="gemini-call-not-hidden-by-thought-filter",
    ),
]


@pytest.fixture
def provider_http(
    body: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> tuple[list[httpx.Request], AsyncMock]:
    original_client = httpx.AsyncClient
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=body)

    def create_client(*, timeout: float) -> httpx.AsyncClient:
        return original_client(
            timeout=timeout, transport=httpx.MockTransport(respond), trust_env=False
        )

    sleep = AsyncMock()
    monkeypatch.setattr("llm.providers.httpx.AsyncClient", create_client)
    monkeypatch.setattr("llm.rate_limit.asyncio.sleep", sleep)
    return requests, sleep


def nonfinal_error(provider: type[HTTPProviderAdapter]) -> str:
    return (
        f"{provider.provider_name} returned an invalid response: "
        "nonfinal tool or continuation output."
    )


@pytest.mark.parametrize(("provider", "body"), NONFINAL_CASES)
async def test_nonfinal_text_is_sanitized_and_not_retried_or_replaced(
    provider: type[HTTPProviderAdapter],
    provider_http: tuple[list[httpx.Request], AsyncMock],
) -> None:
    requests, sleep = provider_http
    adapter = provider(api_key=DUMMY_KEY, model=MODEL)
    router = RoutingLLMAdapter(
        ModelRouter(
            adapters={provider.provider_name: adapter}, default_provider=provider.provider_name
        )
    )

    with pytest.raises(ProviderResponseError) as caught:
        await router.generate(REQUEST)

    assert str(caught.value) == nonfinal_error(provider)
    for private in (PREAMBLE, TOOL_ARGUMENT, THINKING, DUMMY_KEY):
        assert private not in str(caught.value)
    assert len(requests) == 1
    assert requests[0].url == adapter.endpoint
    assert json.loads(requests[0].content) == adapter.payload(REQUEST)
    assert "tools" not in json.loads(requests[0].content)
    assert len(adapter._limiter._timestamps) == 1
    sleep.assert_not_awaited()


@pytest.mark.parametrize(
    ("provider", "body"),
    [case for case in NONFINAL_CASES if case.values[0] is not AnthropicAdapter],
)
@pytest.mark.parametrize("nonfinal_first", [False, True], ids=["later", "first"])
def test_only_first_candidate_controls_nonfinal_rejection(
    provider: type[HTTPProviderAdapter], body: dict[str, object], nonfinal_first: bool
) -> None:
    key = "candidates" if provider is GeminiAdapter else "choices"
    good = gemini_body() if provider is GeminiAdapter else chat_body()
    bad_candidates = body[key]
    good_candidates = good[key]
    assert isinstance(bad_candidates, list)
    assert isinstance(good_candidates, list)
    selected = deepcopy(body)
    selected[key] = (
        bad_candidates + good_candidates if nonfinal_first else good_candidates + bad_candidates
    )
    adapter = provider(api_key=DUMMY_KEY, model=MODEL)
    if nonfinal_first:
        with pytest.raises(ProviderResponseError, match="nonfinal tool or continuation output"):
            adapter.parse_response(selected, REQUEST)
    else:
        response = adapter.parse_response(selected, REQUEST)
        assert response.text == PREAMBLE
        assert response.model_name == MODEL
        assert response.citation_chunk_ids == ["c1"]
        assert response.raw_provider == provider.provider_name


@pytest.mark.parametrize(("provider", "body"), NONFINAL_CASES)
def test_nonfinal_run_preserves_frozen_input_but_never_a_completed_answer(
    provider: type[HTTPProviderAdapter],
    provider_http: tuple[list[httpx.Request], AsyncMock],
    tmp_path: Path,
) -> None:
    credentials = {
        "OPENAI_API_KEY": "",
        "ANTHROPIC_API_KEY": "",
        "GEMINI_API_KEY": "",
        "MOONSHOT_API_KEY": "",
        "SEMANTIC_SCHOLAR_API_KEY": "",
    }
    key_prefix = "MOONSHOT" if provider is KimiAdapter else provider.provider_name.upper()
    credentials[f"{key_prefix}_API_KEY"] = DUMMY_KEY
    settings = Settings(
        _env_file=None,
        database_path=tmp_path / "nonfinal.sqlite3",
        default_model=provider.provider_name,
        **credentials,
        **{f"{provider.provider_name}_model": MODEL},
    )
    app = create_app(settings)
    container = app.state.container
    chunk = Chunk(
        chunk_id="c1",
        document_id="synthetic-document",
        title="Synthetic evidence",
        text="GraphRAG connects research entities.",
        source="offline-fixture",
    )
    container.document_store.add_documents(
        [Document(**chunk.model_dump(exclude={"chunk_id"}))], [chunk]
    )
    container.hybrid_retriever.add_chunks([chunk])
    requests, sleep = provider_http

    with TestClient(app) as client:
        response = client.post("/query", json={"query": REQUEST.prompt})
        assert response.status_code == 200
        result = response.json()["result"]
        events = SQLiteEventLog(settings.database_path).list_events(result["run_id"])
        [snapshot_payload] = [
            event["payload"] for event in events if event["event_type"] == "evidence_snapshot"
        ]
        snapshot = EvidenceSnapshot.model_validate(snapshot_payload)
        assert snapshot.request.prompt == REQUEST.prompt
        assert snapshot.request.context == REQUEST.context
        assert snapshot.request.citation_chunk_ids == ["c1"]
        assert snapshot.request.task_type == TaskType.REASONING
        assert [source.chunk for source in snapshot.sources] == [chunk]
        assert len(requests) == 1
        adapter = provider(api_key=DUMMY_KEY, model=MODEL)
        assert requests[0].url == adapter.endpoint
        assert json.loads(requests[0].content) == adapter.payload(snapshot.request)
        assert result["state"] == "ERROR"
        assert result["answer"] is None
        assert result["error"] == nonfinal_error(provider)
        assert [event["event_type"] for event in events] == [
            "state_transition",
            "decision_log",
            "state_transition",
            "state_transition",
            "evidence_snapshot",
            "state_transition",
        ]
        assert events[-1]["payload"] == {
            "from_state": "REASONING",
            "to_state": "ERROR",
            "payload": {"error": nonfinal_error(provider)},
        }
        assert not any(event["payload"].get("to_state") == "DONE" for event in events)
        for private in (PREAMBLE, TOOL_ARGUMENT, THINKING, DUMMY_KEY, "server-alias"):
            assert private not in json.dumps(events)
            assert private not in response.text

    chunk.text = "Changed after the failed run."
    container.document_store.add_documents(
        [Document(**chunk.model_dump(exclude={"chunk_id"}))], [chunk]
    )
    with TestClient(create_app(settings)) as restarted:
        assert restarted.get(f"/runs/{result['run_id']}/events").json() == events
        assert restarted.get(f"/runs/{result['run_id']}/export").status_code == 409
    assert len(requests) == 1
    sleep.assert_not_awaited()


@pytest.mark.parametrize("reason", [None, "custom-reason", [], {}])
@pytest.mark.parametrize("provider", [OpenAIAdapter, KimiAdapter, AnthropicAdapter, GeminiAdapter])
def test_non_tool_text_keeps_legacy_multipart_and_provenance_contract(
    provider: type[HTTPProviderAdapter], reason: object
) -> None:
    text = "  Grounded answer [c1].\n"
    if provider is AnthropicAdapter:
        body = claude_body(reason=reason)
        body["content"] = [
            {"type": "thinking", "text": THINKING},
            {"type": "text", "text": "  Grounded "},
            {"type": "redacted_thinking", "data": THINKING},
            {"type": "text", "text": "answer [c1].\n"},
        ]
    elif provider is GeminiAdapter:
        body = {
            "modelVersion": "server-alias",
            "candidates": [
                {
                    "finishReason": reason,
                    "content": {
                        "parts": [
                            {"thought": True, "text": THINKING},
                            {"text": "  Grounded ", "functionCall": None, "toolCall": None},
                            {"inlineData": {"mimeType": "image/png", "data": ""}},
                            {"text": "answer [c1].\n"},
                        ]
                    },
                }
            ],
        }
    else:
        body = chat_body(
            reason=reason,
            content=[
                {"type": "reasoning", "text": THINKING},
                {"type": "text", "text": "  Grounded "},
                {"text": "answer [c1].\n"},
            ],
            tool_calls=[],
            function_call=None,
        )
    response = provider(api_key=DUMMY_KEY, model=MODEL).parse_response(body, REQUEST)
    assert response.text == text
    assert response.citation_chunk_ids == ["c1"]
    assert response.raw_provider == provider.provider_name
    assert response.model_name == MODEL
