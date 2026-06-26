"""Tests for application dependency wiring."""

from pathlib import Path

from api.dependencies import AppContainer
from config import Settings
from llm.fake import FakeLLMAdapter
from llm.router import RoutingLLMAdapter, build_model_router
from llm.schemas import TaskType


def test_app_container_uses_configured_llm_router(tmp_path: Path) -> None:
    """AppContainer should not hardcode the fake LLM when provider keys exist."""

    container = AppContainer(
        Settings(
            database_path=tmp_path / "container.sqlite3",
            OPENAI_API_KEY="test-openai-key",
        )
    )

    assert not isinstance(container.llm, FakeLLMAdapter)
    assert isinstance(container.llm, RoutingLLMAdapter)


def test_build_model_router_falls_back_to_fake_without_provider_keys() -> None:
    """Model router should preserve offline fake fallback without live credentials."""

    router = build_model_router(Settings())

    assert router.route(TaskType.REASONING).provider_name == "fake"


def test_build_model_router_honors_configured_default_provider() -> None:
    """Model router should use the configured default provider for default tasks."""

    router = build_model_router(
        Settings(
            default_model="anthropic",
            ANTHROPIC_API_KEY="test-anthropic-key",
        )
    )

    assert router.route(TaskType.DEFAULT).provider_name == "anthropic"
