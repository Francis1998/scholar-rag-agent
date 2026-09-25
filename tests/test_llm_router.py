"""Tests for LLM routing and fake generation."""

import pytest

from llm.fake import FakeLLMAdapter
from llm.providers import (
    AnthropicAdapter,
    GeminiAdapter,
    HTTPProviderAdapter,
    KimiAdapter,
    OpenAIAdapter,
)
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


@pytest.mark.parametrize(
    ("adapter", "model"),
    [
        (OpenAIAdapter(api_key="test-key"), "gpt-6-astra"),
        (AnthropicAdapter(api_key="test-key"), "claude-sonnet-5"),
        (GeminiAdapter(api_key="test-key"), "gemini-3.8-flash"),
        (KimiAdapter(api_key="test-key"), "kimi-k3"),
    ],
    ids=["openai", "anthropic", "gemini", "kimi"],
)
def test_live_provider_defaults_use_current_model_stack(
    adapter: HTTPProviderAdapter, model: str
) -> None:
    """Default model IDs match the provider catalogs checked on 2026-09-17."""
    request = LLMRequest(
        task_type=TaskType.REASONING,
        prompt="What is GraphRAG?",
        context="[c1] GraphRAG links entities across papers.",
        citation_chunk_ids=["c1"],
    )

    if isinstance(adapter, GeminiAdapter):
        assert adapter.endpoint == (
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        )
    else:
        assert adapter.payload(request)["model"] == model


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
                        {"thought": True, "text": "Internal reasoning is not an answer."},
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


def test_gemini_parse_response_rejects_non_dict_candidate() -> None:
    """An unusable candidate must raise a response error, not fabricate an answer."""
    adapter = GeminiAdapter(api_key="test-key")
    request = LLMRequest(
        task_type=TaskType.REASONING,
        prompt="Summarize the findings.",
        context="[c1] evidence.",
        citation_chunk_ids=["c1"],
    )

    with pytest.raises(ValueError, match="gemini returned an invalid response"):
        adapter.parse_response({"candidates": [None]}, request)


@pytest.mark.parametrize(
    "adapter",
    [OpenAIAdapter(api_key="test-key"), KimiAdapter(api_key="test-key")],
    ids=["openai", "kimi"],
)
def test_openai_parse_response_joins_structured_content_parts(
    adapter: HTTPProviderAdapter,
) -> None:
    """OpenAI parsing must join a structured content-part list into plain text.

    The base Chat Completions contract returns ``message.content`` as a string,
    but OpenAI-compatible gateways (LiteLLM, vLLM, OpenRouter) may return it as a
    list of ``{"type": "text", "text": ...}`` parts. Coercing that list with
    ``str(...)`` produced a Python repr (``"[{'type': 'text', ...}]"``) as the
    answer instead of the text. Each part's ``text`` must be extracted and
    joined. The KimiAdapter subclass shares this parser and behavior.
    """
    request = LLMRequest(
        task_type=TaskType.REASONING,
        prompt="Summarize the findings.",
        context="[c1] evidence.",
        citation_chunk_ids=["c1"],
    )
    data = {
        "choices": [
            {
                "message": {
                    "reasoning_content": "Internal reasoning is not an answer.",
                    "content": [
                        {"type": "reasoning", "text": "Non-answer content."},
                        {"type": "text", "text": "The study "},
                        {"type": "text", "text": "supports the hypothesis [c1]."},
                    ],
                }
            }
        ]
    }

    response = adapter.parse_response(data, request)

    assert response.text == "The study supports the hypothesis [c1]."


def test_openai_parse_response_reads_plain_string_content() -> None:
    """A plain-string ``message.content`` is returned unchanged."""
    adapter = OpenAIAdapter(api_key="test-key")
    request = LLMRequest(
        task_type=TaskType.REASONING,
        prompt="Summarize.",
        context="[c1] evidence.",
        citation_chunk_ids=["c1"],
    )
    data = {"choices": [{"message": {"content": "Grounded answer [c1]."}}]}

    response = adapter.parse_response(data, request)

    assert response.text == "Grounded answer [c1]."


def test_anthropic_parse_response_joins_all_text_blocks() -> None:
    """Anthropic parsing must join text blocks and skip non-text blocks.

    Anthropic's ``content`` is an ordered list of typed blocks. A leading
    non-text block (for example ``thinking`` or ``tool_use``) has no ``text``
    key, so reading ``content[0]['text']`` raised ``KeyError`` and crashed the
    request; when the first block was text but more followed, the answer was
    truncated. All text blocks must be concatenated and non-text blocks skipped.
    """
    adapter = AnthropicAdapter(api_key="test-key")
    request = LLMRequest(
        task_type=TaskType.REASONING,
        prompt="Summarize the findings.",
        context="[c1] evidence.",
        citation_chunk_ids=["c1"],
    )
    data = {
        "content": [
            {"type": "thinking", "thinking": "internal reasoning"},
            {"type": "thinking", "text": "Non-answer text must also be ignored."},
            {"type": "text", "text": "The study "},
            {"type": "tool_use", "text": "Not an answer."},
            {"type": "text", "text": "supports the hypothesis [c1]."},
        ]
    }

    response = adapter.parse_response(data, request)

    assert response.text == "The study supports the hypothesis [c1]."
