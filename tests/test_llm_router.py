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


def test_gemini_parse_response_tolerates_non_dict_candidate() -> None:
    """A malformed or blocked candidate must degrade to empty text, not crash.

    Gemini can return a ``candidates`` list whose first element is not an object
    (for example ``null`` for a blocked candidate, or a malformed gateway
    payload). The parser guarded the ``candidates`` list but not ``candidates[0]``
    and called ``.get`` on it directly, raising ``AttributeError`` and failing the
    whole request. The other adapters guard their first element, so parsing must
    return an empty completion here rather than raise.
    """
    adapter = GeminiAdapter(api_key="test-key")
    request = LLMRequest(
        task_type=TaskType.REASONING,
        prompt="Summarize the findings.",
        context="[c1] evidence.",
        citation_chunk_ids=["c1"],
    )

    response = adapter.parse_response({"candidates": [None]}, request)

    assert response.text == ""
    assert response.citation_chunk_ids == ["c1"]


def test_openai_parse_response_joins_structured_content_parts() -> None:
    """OpenAI parsing must join a structured content-part list into plain text.

    The base Chat Completions contract returns ``message.content`` as a string,
    but OpenAI-compatible gateways (LiteLLM, vLLM, OpenRouter) may return it as a
    list of ``{"type": "text", "text": ...}`` parts. Coercing that list with
    ``str(...)`` produced a Python repr (``"[{'type': 'text', ...}]"``) as the
    answer instead of the text. Each part's ``text`` must be extracted and
    joined. The KimiAdapter subclass shares this parser and behavior.
    """
    adapter = OpenAIAdapter(api_key="test-key")
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
                    "content": [
                        {"type": "text", "text": "The study "},
                        {"type": "text", "text": "supports the hypothesis [c1]."},
                    ]
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
            {"type": "text", "text": "The study "},
            {"type": "text", "text": "supports the hypothesis [c1]."},
        ]
    }

    response = adapter.parse_response(data, request)

    assert response.text == "The study supports the hypothesis [c1]."
