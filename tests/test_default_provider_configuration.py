"""Default-provider validation and unchanged routing contracts."""

import json
import os
from collections.abc import Callable, Iterator
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from pydantic import ValidationError

from api import dependencies
from api.application import create_app
from config import Settings, load_settings
from llm.defaults import (
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_KIMI_MODEL,
    DEFAULT_OPENAI_MODEL,
)
from llm.fake import FakeLLMAdapter
from llm.router import ModelRouter, build_model_router
from llm.schemas import TaskType

PROVIDER_NAMES = ("openai", "anthropic", "gemini", "kimi", "fake")
PROVIDER_KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "MOONSHOT_API_KEY")
DEFAULT_PROVIDER_ENV = "SCHOLAR_RAG_DEFAULT_MODEL"
INVALID_PROVIDERS = [
    pytest.param("", id="empty"),
    pytest.param(" \t", id="whitespace-only"),
    pytest.param("unknown-provider", id="unknown"),
    pytest.param("moonshot", id="not-a-provider-family"),
    pytest.param(DEFAULT_OPENAI_MODEL, id="openai-model-id"),
    pytest.param(DEFAULT_ANTHROPIC_MODEL, id="anthropic-model-id"),
    pytest.param(DEFAULT_GEMINI_MODEL, id="gemini-model-id"),
    pytest.param(DEFAULT_KIMI_MODEL, id="kimi-model-id"),
    *[pytest.param(name.upper(), id=f"{name}-uppercase") for name in PROVIDER_NAMES],
    *[pytest.param(f" {name}", id=f"{name}-leading-space") for name in PROVIDER_NAMES],
    *[pytest.param(f"{name} ", id=f"{name}-trailing-space") for name in PROVIDER_NAMES],
]


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    monkeypatch.chdir(tmp_path)
    for name in list(os.environ):
        if name.upper().startswith("SCHOLAR_RAG_") or name.upper() in (
            *PROVIDER_KEYS,
            "SEMANTIC_SCHOLAR_API_KEY",
        ):
            monkeypatch.delenv(name)
    send = AsyncMock(side_effect=AssertionError("Provider configuration must not make HTTP calls."))
    sync_send = Mock(side_effect=AssertionError("Provider configuration must not make HTTP calls."))
    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    monkeypatch.setattr(httpx.Client, "send", sync_send)
    yield
    send.assert_not_called()
    sync_send.assert_not_called()


def set_provider_source(
    value: str, source: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    if source == "environment":
        monkeypatch.setenv(DEFAULT_PROVIDER_ENV, value)
    else:
        assert source == "dotenv"
        (tmp_path / ".env").write_text(
            f"{DEFAULT_PROVIDER_ENV}={json.dumps(value)}\n", encoding="utf-8"
        )


@pytest.fixture(params=["constructor", "environment", "dotenv"])
def make_settings(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Callable[[str], Settings]:
    def make(value: str) -> Settings:
        if request.param == "constructor":
            return Settings(default_model=value)
        set_provider_source(value, request.param, monkeypatch, tmp_path)
        return load_settings()

    return make


def assert_provider_validation_error(error: ValidationError) -> None:
    [detail] = error.errors()
    assert detail["loc"] == ("default_model",)
    assert detail["type"] == "literal_error"
    for name in PROVIDER_NAMES:
        assert f"'{name}'" in detail["msg"]


@pytest.mark.parametrize("value", INVALID_PROVIDERS)
def test_settings_reject_invalid_default_provider(
    make_settings: Callable[[str], Settings], value: str
) -> None:
    with pytest.raises(ValidationError) as caught:
        make_settings(value)
    assert_provider_validation_error(caught.value)


@pytest.mark.parametrize("value", [None, False, 1, b"openai", ["openai"]])
def test_settings_reject_nonstring_default_provider(value: object) -> None:
    with pytest.raises(ValidationError) as caught:
        Settings(default_model=value)
    assert caught.value.errors()[0]["loc"] == ("default_model",)


@pytest.mark.parametrize("provider", PROVIDER_NAMES)
def test_valid_families_reach_default_route_without_overriding_task_preferences(
    make_settings: Callable[[str], Settings], provider: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key in PROVIDER_KEYS:
        monkeypatch.setenv(key, "test-provider-key")
    settings = make_settings(provider)
    assert settings.default_model == provider
    assert type(settings.default_model) is str
    router = build_model_router(settings)
    assert router.route(TaskType.DEFAULT).provider_name == provider
    assert router.route(TaskType.REASONING).provider_name == "anthropic"
    assert router.route(TaskType.SPEED).provider_name == "gemini"
    assert router.route(TaskType.COST).provider_name == "kimi"


def test_unset_default_provider_remains_openai() -> None:
    assert load_settings().default_model == "openai"


def test_provider_configuration_keeps_settings_precedence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    set_provider_source("kimi", "dotenv", monkeypatch, tmp_path)
    assert load_settings().default_model == "kimi"
    monkeypatch.setenv(DEFAULT_PROVIDER_ENV, "anthropic")
    assert load_settings().default_model == "anthropic"
    assert Settings(default_model="gemini").default_model == "gemini"

    set_provider_source("invalid-file-provider", "dotenv", monkeypatch, tmp_path)
    assert load_settings().default_model == "anthropic"
    monkeypatch.setenv(DEFAULT_PROVIDER_ENV, "invalid-environment-provider")
    assert Settings(default_model="fake").default_model == "fake"
    with pytest.raises(ValidationError) as caught:
        load_settings()
    assert_provider_validation_error(caught.value)


@pytest.mark.parametrize("entrypoint", ["container", "application"])
@pytest.mark.parametrize("source", ["environment", "dotenv"])
@pytest.mark.parametrize("value", ["", "unknown-provider", DEFAULT_OPENAI_MODEL, " OpenAI "])
def test_invalid_provider_fails_before_container_storage_or_router_initialization(
    entrypoint: str,
    source: str,
    value: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "startup.sqlite3"
    monkeypatch.setenv("SCHOLAR_RAG_DATABASE_PATH", str(database_path))
    for key in PROVIDER_KEYS:
        monkeypatch.setenv(key, "test-provider-key")
    set_provider_source(value, source, monkeypatch, tmp_path)
    container = Mock(wraps=dependencies.AppContainer)
    storage = Mock(wraps=dependencies.SQLiteEventLog)
    router = Mock(wraps=dependencies.build_model_router)
    monkeypatch.setattr(dependencies, "AppContainer", container)
    monkeypatch.setattr(dependencies, "SQLiteEventLog", storage)
    monkeypatch.setattr(dependencies, "build_model_router", router)

    with pytest.raises(ValidationError) as caught:
        if entrypoint == "container":
            dependencies.create_container()
        else:
            create_app()
    assert_provider_validation_error(caught.value)
    container.assert_not_called()
    storage.assert_not_called()
    router.assert_not_called()
    assert not database_path.exists()


@pytest.mark.parametrize("provider", PROVIDER_NAMES)
@pytest.mark.parametrize("task", list(TaskType))
def test_valid_defaults_without_keys_keep_fake_fallback(provider: str, task: TaskType) -> None:
    router = build_model_router(Settings(default_model=provider))
    assert router.route(task).provider_name == "fake"


@pytest.mark.parametrize("task", list(TaskType))
def test_missing_preferred_and_default_providers_keep_openai_fallback(task: TaskType) -> None:
    router = build_model_router(
        Settings(default_model="anthropic", OPENAI_API_KEY="test-provider-key")
    )
    assert router.route(task).provider_name == "openai"


@pytest.mark.parametrize("name", ["custom-adapter", " Custom Adapter "])
@pytest.mark.parametrize("task", list(TaskType))
def test_direct_router_keeps_custom_default_adapter_names(name: str, task: TaskType) -> None:
    custom = FakeLLMAdapter()
    router = ModelRouter(adapters={name: custom, "openai": FakeLLMAdapter()}, default_provider=name)
    assert router.route(task) is custom


def test_direct_router_keeps_unknown_default_fallbacks() -> None:
    fallback = FakeLLMAdapter()
    router = ModelRouter(adapters={"openai": fallback}, default_provider="external-provider")
    assert router.route(TaskType.DEFAULT) is fallback
    assert (
        ModelRouter(default_provider="external-provider").route(TaskType.DEFAULT).provider_name
        == "fake"
    )
