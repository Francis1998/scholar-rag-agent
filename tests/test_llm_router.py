"""Tests for LLM routing and fake generation."""

from llm.fake import FakeLLMAdapter
from llm.router import ModelRouter
from llm.schemas import LLMRequest, TaskType


async def test_fake_llm_returns_validated_response() -> None:
    """Fake LLM produces a validated response with chunk citations."""
    response = await FakeLLMAdapter().generate(
        LLMRequest(
            task_type=TaskType.REASONING,
            prompt="What is RAG?",
            context="[c1] RAG uses retrieval.",
            citation_chunk_ids=["c1"],
        )
    )
    assert response.citation_chunk_ids == ["c1"]


def test_router_defaults_to_fake_without_provider_keys() -> None:
    """Router falls back to the fake adapter when no live providers are configured."""
    adapter = ModelRouter().route(TaskType.REASONING)
    assert adapter.provider_name == "fake"
