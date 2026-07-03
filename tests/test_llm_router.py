"""Tests for LLM routing and fake generation."""

from llm.fake import FakeLLMAdapter
from llm.providers import AnthropicAdapter, GeminiAdapter, KimiAdapter, OpenAIAdapter
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


def test_live_provider_defaults_use_current_model_stack() -> None:
    """Live provider defaults should stay aligned with the current agentic AI stack."""
    request = LLMRequest(
        task_type=TaskType.REASONING,
        prompt="What is GraphRAG?",
        context="[c1] GraphRAG links entities across papers.",
        citation_chunk_ids=["c1"],
    )

    assert OpenAIAdapter(api_key="test-key").payload(request)["model"] == "gpt-5.5"
    assert AnthropicAdapter(api_key="test-key").payload(request)["model"] == "claude-sonnet-4-6"
    assert "gemini-3.1-pro-preview" in GeminiAdapter(api_key="test-key").endpoint
    assert KimiAdapter(api_key="test-key").payload(request)["model"] == "kimi-k2"


def test_gemini_parse_response_concatenates_all_text_parts() -> None:
    """Gemini parsing must join every text part, not only the first.

    A Gemini candidate's ``content.parts`` is a list that can hold multiple
    text segments interleaved with non-text parts (for example a
    ``functionCall``). Reading only ``parts[0]`` silently truncated multi-part
    answers, dropping cited evidence from the grounded response.
    """
    adapter = GeminiAdapter(api_key="test-key")
    request = LLMRequest(
        task_type=TaskType.REASONING,
        prompt="Summarize the findings.",
        context="[c1] evidence.",
        citation_chunk_ids=["c1"],
    )
    data = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "The study "},
                        {"functionCall": {"name": "noop"}},
                        {"text": "supports the hypothesis [c1]."},
                    ]
                }
            }
        ]
    }

    response = adapter.parse_response(data, request)

    assert response.text == "The study supports the hypothesis [c1]."
