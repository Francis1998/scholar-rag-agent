"""Offline regression coverage for configurable provider model contracts."""

import json
from pathlib import Path
from typing import NamedTuple
from unittest.mock import patch

import httpx
import pytest
from pydantic import ValidationError

from agent.models import AgentState
from api.dependencies import AppContainer
from config import Settings
from llm.providers import (
    AnthropicAdapter,
    GeminiAdapter,
    HTTPProviderAdapter,
    KimiAdapter,
    OpenAIAdapter,
)
from llm.router import RoutingLLMAdapter, build_model_router
from llm.schemas import LLMRequest, LLMResponse, TaskType
from retrieval.models import Chunk, Document
from storage.event_log import SQLiteEventLog


class ProviderCase(NamedTuple):
    """A provider's current ID and an explicit, provider-specific override."""

    name: str
    adapter: type[HTTPProviderAdapter]
    model: str
    override: str
    key_env: str


PROVIDERS = [
    ProviderCase("openai", OpenAIAdapter, "gpt-6-astra", "gpt-5.6-terra", "OPENAI_API_KEY"),
    ProviderCase(
        "anthropic",
        AnthropicAdapter,
        "claude-sonnet-5-5",
        "claude-haiku-4-5",
        "ANTHROPIC_API_KEY",
    ),
    ProviderCase("gemini", GeminiAdapter, "gemini-3.8-flash", "gemini-3.7-flash", "GEMINI_API_KEY"),
    ProviderCase("kimi", KimiAdapter, "kimi-k3", "kimi-k2.6", "MOONSHOT_API_KEY"),
]


@pytest.fixture(autouse=True)
def clean_provider_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep credentials and model selections independent of the caller's shell."""
    monkeypatch.delenv("SCHOLAR_RAG_DEFAULT_MODEL", raising=False)
    for provider in PROVIDERS:
        monkeypatch.delenv(provider.key_env, raising=False)
        monkeypatch.delenv(f"SCHOLAR_RAG_{provider.name.upper()}_MODEL", raising=False)


@pytest.fixture(params=PROVIDERS, ids=[provider.name for provider in PROVIDERS])
def provider_case(request: pytest.FixtureRequest) -> ProviderCase:
    """Exercise the same configuration contract for every live provider."""
    assert isinstance(request.param, ProviderCase)
    return request.param


@pytest.fixture
def llm_request() -> LLMRequest:
    """A stateless text request with one citation."""
    return LLMRequest(
        task_type=TaskType.DEFAULT,
        prompt="Summarize the findings.",
        context="[c1] Evidence.",
        citation_chunk_ids=["c1"],
    )


def assert_requested_model(adapter: HTTPProviderAdapter, request: LLMRequest, model: str) -> None:
    """Check the actual wire selection, including Gemini's URL-based model ID."""
    if isinstance(adapter, GeminiAdapter):
        assert httpx.URL(adapter.endpoint).path == f"/v1beta/models/{model}:generateContent"
    else:
        assert adapter.payload(request)["model"] == model


def assert_anthropic_request(request: httpx.Request, llm_request: LLMRequest, model: str) -> None:
    """Assert the bounded text contract independently of the production payload builder."""
    assert request.method == "POST"
    assert str(request.url) == "https://api.anthropic.com/v1/messages"
    assert request.headers["x-api-key"] == "test-provider-key"
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert request.headers["content-type"] == "application/json"
    assert "anthropic-beta" not in request.headers
    assert "authorization" not in request.headers
    expected: dict[str, object] = {
        "model": model,
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": f"Context:\n{llm_request.context}\n\nQuestion:\n{llm_request.prompt}",
            }
        ],
    }
    if model == "claude-sonnet-5-5":
        expected["thinking"] = {"type": "between_tools"}
        expected["output_config"] = {"effort": "medium"}
    elif model == "claude-sonnet-5":
        expected["thinking"] = {"type": "disabled"}
    assert json.loads(request.content) == expected


def test_settings_expose_current_model_defaults(provider_case: ProviderCase) -> None:
    """Settings use current defaults without changing the default provider family."""
    settings = Settings(_env_file=None)
    assert getattr(settings, f"{provider_case.name}_model") == provider_case.model
    assert settings.default_model == "openai"


@pytest.mark.parametrize("value", ["", " \t\n", None, 123, b"byte-model"])
def test_model_settings_reject_invalid_identifiers(
    provider_case: ProviderCase, value: str | bytes | int | None
) -> None:
    """An explicit blank, null, or non-string ID must not silently use a default."""
    field = f"{provider_case.name}_model"
    with pytest.raises(ValidationError) as error:
        Settings(_env_file=None, **{field: value})
    assert error.value.errors()[0]["loc"] == (field,)


@pytest.mark.parametrize("value", ["", " \t"])
def test_model_environment_rejects_blank_identifiers(
    provider_case: ProviderCase, value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Blank environment values are configuration errors even without credentials."""
    monkeypatch.setenv(f"SCHOLAR_RAG_{provider_case.name.upper()}_MODEL", value)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_model_selection_precedence_and_whitespace(
    provider_case: ProviderCase, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Explicit settings beat environment, then dotenv, then the default."""
    field = f"{provider_case.name}_model"
    env_name = f"SCHOLAR_RAG_{field.upper()}"
    dotenv = tmp_path / ".env"
    dotenv.write_text(f'{env_name}="  file-model  "\n', encoding="utf-8")
    assert getattr(Settings(_env_file=dotenv), field) == "file-model"

    monkeypatch.setenv(env_name, f"  {provider_case.override}  ")
    assert getattr(Settings(_env_file=dotenv), field) == provider_case.override
    assert (
        getattr(Settings(_env_file=dotenv, **{field: "  explicit-model  "}), field)
        == "explicit-model"
    )


def test_explicit_adapter_model_is_preserved(
    provider_case: ProviderCase, llm_request: LLMRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Direct constructor overrides remain independent of application settings."""
    monkeypatch.setenv(f"SCHOLAR_RAG_{provider_case.name.upper()}_MODEL", "different-model")
    adapter = provider_case.adapter(api_key="test-key", model=provider_case.override)
    assert_requested_model(adapter, llm_request, provider_case.override)


def test_environment_model_reaches_routed_adapter(
    provider_case: ProviderCase, llm_request: LLMRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An environment override reaches the selected provider's payload or URL."""
    monkeypatch.setenv(provider_case.key_env, "test-key")
    monkeypatch.setenv(
        f"SCHOLAR_RAG_{provider_case.name.upper()}_MODEL", f"  {provider_case.override}  "
    )
    router = build_model_router(Settings(_env_file=None, default_model=provider_case.name))
    adapter = router.route(TaskType.DEFAULT)
    assert adapter.provider_name == provider_case.name
    assert isinstance(adapter, HTTPProviderAdapter)
    assert_requested_model(adapter, llm_request, provider_case.override)


@pytest.mark.parametrize("use_override", [False, True], ids=["current", "override"])
@pytest.mark.parametrize("selection", ["constructor", "router"])
async def test_current_model_http_contract(
    provider_case: ProviderCase,
    llm_request: LLMRequest,
    use_override: bool,
    selection: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HTTPX exercises model-compatible JSON and normalized, non-secret provenance."""
    model = provider_case.override if use_override else provider_case.model
    if selection == "router":
        monkeypatch.setenv(provider_case.key_env, "test-provider-key")
        if use_override:
            monkeypatch.setenv(f"SCHOLAR_RAG_{provider_case.name.upper()}_MODEL", f"  {model}  ")
        router = build_model_router(Settings(_env_file=None, default_model=provider_case.name))
        adapter = router.route(llm_request.task_type)
        assert isinstance(adapter, HTTPProviderAdapter)
    elif use_override:
        adapter = provider_case.adapter(api_key="test-provider-key", model=model)
    else:
        adapter = provider_case.adapter(api_key="test-provider-key")
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.method == "POST"
        assert str(request.url) == adapter.endpoint
        payload = json.loads(request.content)
        rejected_parameters = {"temperature", "top_p", "top_k", "top_logprobs", "logprobs"}
        if rejected_parameters.intersection(payload):
            return httpx.Response(400, json={"error": "Unsupported sampling parameter"})
        if provider_case.name == "gemini":
            assert not request.url.query
            assert "test-provider-key" not in str(request.url)
            assert request.headers["x-goog-api-key"] == "test-provider-key"
            assert request.headers["content-type"] == "application/json"
            assert payload == {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": (
                                    f"Context:\n{llm_request.context}"
                                    f"\n\nQuestion:\n{llm_request.prompt}"
                                )
                            }
                        ]
                    }
                ]
            }
            data = {
                "modelVersion": "server-reported-version",
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"thought": True, "text": "Not answer text."},
                                {"text": "Grounded "},
                                {"text": "answer [c1]."},
                            ]
                        }
                    }
                ],
            }
        elif provider_case.name == "anthropic":
            assert_anthropic_request(request, llm_request, model)
            data = {
                "model": "server-reported-version",
                "content": [
                    {"type": "thinking", "thinking": "Not answer text."},
                    {"type": "text", "text": "Grounded "},
                    {"type": "text", "text": "answer [c1]."},
                ],
            }
        else:
            assert payload["model"] == model
            assert set(payload) == {"model", "messages"}
            data = {
                "model": "server-reported-version",
                "choices": [
                    {
                        "message": {
                            "reasoning_content": "Not answer text.",
                            "content": "Grounded answer [c1].",
                        }
                    }
                ],
            }
        return httpx.Response(200, json=data)

    client = httpx.AsyncClient(
        timeout=60.0, transport=httpx.MockTransport(respond), trust_env=False
    )
    with patch("llm.providers.httpx.AsyncClient", return_value=client) as create_client:
        response = await adapter.generate(llm_request)

    create_client.assert_called_once_with(timeout=60.0)
    assert len(requests) == 1
    assert response.text == "Grounded answer [c1]."
    assert response.citation_chunk_ids == ["c1"]
    assert response.raw_provider == provider_case.name
    assert response.model_name == model
    assert "test-provider-key" not in response.model_dump_json()


@pytest.mark.parametrize(
    "model",
    [
        "claude-sonnet-5-5",
        "claude-sonnet-5",
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
        "claude-opus-5-5",
        "claude-fable-5-1",
        "custom-anthropic-model",
        "claude-sonnet-5-custom",
        "claude-sonnet-5-5-custom",
    ],
)
async def test_anthropic_overrides_use_only_exact_model_controls(
    model: str, llm_request: LLMRequest
) -> None:
    """New/legacy Sonnet contracts must not leak to other IDs, even lookalikes."""
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert_anthropic_request(request, llm_request, model)
        return httpx.Response(
            200,
            json={
                "model": "server-reported-version",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "Grounded answer [c1]."}],
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond), trust_env=False)
    with patch("llm.providers.httpx.AsyncClient", return_value=client):
        response = await AnthropicAdapter(api_key="test-provider-key", model=model).generate(
            llm_request
        )
    assert len(requests) == 1
    assert response.text == "Grounded answer [c1]."
    assert response.model_name == model


@pytest.mark.parametrize(
    ("override", "model"),
    [
        (None, "claude-sonnet-5-5"),
        ("  claude-sonnet-5  ", "claude-sonnet-5"),
        ("custom-anthropic-model", "custom-anthropic-model"),
    ],
    ids=["default", "legacy-rollback", "custom"],
)
async def test_anthropic_runner_saves_the_actually_requested_model(
    override: str | None, model: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Real routing and SQLite persistence retain the wire ID, not a server alias."""
    if override is not None:
        monkeypatch.setenv("SCHOLAR_RAG_ANTHROPIC_MODEL", override)
    settings = Settings(
        _env_file=None,
        database_path=tmp_path / "anthropic-provenance.sqlite3",
        ANTHROPIC_API_KEY="test-provider-key",
        SEMANTIC_SCHOLAR_API_KEY="",
    )
    container = AppContainer(settings)
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
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "model": "server-reported-version",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "GraphRAG connects research entities [c1]."}],
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond), trust_env=False)
    with patch("llm.providers.httpx.AsyncClient", return_value=client):
        result = await container.runner.run("What does GraphRAG connect?")

    assert result.state == AgentState.DONE, result.error
    assert result.answer is not None
    assert result.answer.answer == "GraphRAG connects research entities [c1]."
    assert len(requests) == 1
    events = SQLiteEventLog(settings.database_path).list_events(result.run_id)
    [snapshot] = [
        event["payload"] for event in events if event["event_type"] == "evidence_snapshot"
    ]
    [generation] = [
        event["payload"] for event in events if event["event_type"] == "generation_record"
    ]
    llm_request = LLMRequest.model_validate(snapshot["request"])
    assert llm_request.task_type == TaskType.REASONING
    assert_anthropic_request(requests[0], llm_request, model)
    assert generation == {
        "provider": "anthropic",
        "model_name": json.loads(requests[0].content)["model"],
        "task_type": "reasoning",
        "claim_chunk_ids": [["c1"]],
    }
    assert "test-provider-key" not in json.dumps(events)
    assert "server-reported-version" not in json.dumps(events)


@pytest.mark.parametrize(
    ("configured", "default", "task", "expected"),
    [
        (("openai", "anthropic", "gemini", "kimi"), "openai", TaskType.REASONING, "anthropic"),
        (("openai", "anthropic", "gemini", "kimi"), "openai", TaskType.SPEED, "gemini"),
        (("openai", "anthropic", "gemini", "kimi"), "openai", TaskType.COST, "kimi"),
        (("openai", "anthropic", "gemini", "kimi"), "kimi", TaskType.DEFAULT, "kimi"),
        (("openai", "kimi"), "kimi", TaskType.REASONING, "kimi"),
        (("openai",), "anthropic", TaskType.SPEED, "openai"),
        (("gemini",), "anthropic", TaskType.REASONING, "fake"),
        (("openai", "anthropic"), "fake", TaskType.REASONING, "anthropic"),
        (("openai", "anthropic"), "fake", TaskType.DEFAULT, "fake"),
        ((), "openai", TaskType.COST, "fake"),
    ],
)
def test_routing_preferences_and_missing_provider_fallbacks_are_unchanged(
    configured: tuple[str, ...], default: str, task: TaskType, expected: str
) -> None:
    """Model settings must not alter task preferences or broaden fallback selection."""
    credentials = {
        provider.key_env: "test-key" for provider in PROVIDERS if provider.name in configured
    }
    router = build_model_router(Settings(_env_file=None, default_model=default, **credentials))
    assert router.route(task).provider_name == expected


@pytest.mark.parametrize("task", list(TaskType))
async def test_model_settings_without_keys_keep_deterministic_offline_fallback(
    task: TaskType, llm_request: LLMRequest
) -> None:
    """Model IDs alone never enable live calls, including reasoning requests."""
    models = {f"{provider.name}_model": provider.override for provider in PROVIDERS}
    adapter = RoutingLLMAdapter(build_model_router(Settings(_env_file=None, **models)))
    request = llm_request.model_copy(update={"task_type": task})
    with patch("llm.providers.httpx.AsyncClient", side_effect=AssertionError("Unexpected HTTP")):
        response = await adapter.generate(request)
        assert response == await adapter.generate(request)
    assert response.raw_provider == "fake"
    assert response.model_name is None
    assert response.citation_chunk_ids == ["c1"]


def test_response_model_name_is_optional_for_existing_callers() -> None:
    """Adding provenance does not require changes to existing response producers."""
    response = LLMResponse(text="Existing adapter response.", raw_provider="custom")
    assert response.model_name is None
