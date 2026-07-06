"""HTTP-based adapters for OpenAI, Anthropic, Gemini, and Moonshot Kimi."""

from collections.abc import Mapping

import httpx

from llm.base import BaseLLMAdapter
from llm.rate_limit import AsyncRateLimiter, with_backoff
from llm.schemas import LLMRequest, LLMResponse

TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}


def _is_transient_http_error(exc: Exception) -> bool:
    """Return whether a provider error is worth retrying.

    Args:
        exc: Exception raised during a provider call.

    Returns:
        True for transport-level failures and transient HTTP status codes;
        False for permanent client errors (for example 400 or 401) so they are
        surfaced immediately instead of being retried.
    """
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in TRANSIENT_STATUS_CODES
    return False


class HTTPProviderAdapter(BaseLLMAdapter):
    """Base class for optional live HTTP LLM providers."""

    def __init__(self, api_key: str, model: str, requests_per_minute: int = 60) -> None:
        """Create a provider adapter with API key and model name."""
        self._api_key = api_key
        self._model = model
        self._limiter = AsyncRateLimiter(requests_per_minute=requests_per_minute)

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a validated response through provider HTTP APIs."""
        await self._limiter.acquire()
        return await with_backoff(
            lambda: self._generate_once(request),
            is_retryable=_is_transient_http_error,
        )

    async def _generate_once(self, request: LLMRequest) -> LLMResponse:
        """Execute one provider call without retry handling."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                self.endpoint,
                headers=self.headers,
                json=self.payload(request),
            )
            response.raise_for_status()
            return self.parse_response(response.json(), request)

    @property
    def endpoint(self) -> str:
        """Return provider endpoint URL."""
        raise NotImplementedError

    @property
    def headers(self) -> Mapping[str, str]:
        """Return provider headers."""
        raise NotImplementedError

    def payload(self, request: LLMRequest) -> dict[str, object]:
        """Return provider request payload."""
        raise NotImplementedError

    def parse_response(self, data: Mapping[str, object], request: LLMRequest) -> LLMResponse:
        """Parse provider response JSON into the validated schema."""
        raise NotImplementedError


class OpenAIAdapter(HTTPProviderAdapter):
    """OpenAI GPT adapter using the chat completions API."""

    provider_name = "openai"

    def __init__(self, api_key: str, model: str = "gpt-5.5") -> None:
        """Create an OpenAI adapter."""
        super().__init__(api_key=api_key, model=model)

    @property
    def endpoint(self) -> str:
        """Return OpenAI endpoint URL."""
        return "https://api.openai.com/v1/chat/completions"

    @property
    def headers(self) -> Mapping[str, str]:
        """Return OpenAI request headers."""
        return {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}

    def payload(self, request: LLMRequest) -> dict[str, object]:
        """Return OpenAI chat-completions payload."""
        return {
            "model": self._model,
            "messages": [
                {"role": "system", "content": "Answer with citation chunk IDs from the context."},
                {
                    "role": "user",
                    "content": f"Context:\n{request.context}\n\nQuestion:\n{request.prompt}",
                },
            ],
            "temperature": 0.1,
        }

    def parse_response(self, data: Mapping[str, object], request: LLMRequest) -> LLMResponse:
        """Parse OpenAI response JSON.

        The base contract returns ``message.content`` as a string, but
        OpenAI-compatible gateways (LiteLLM, vLLM, OpenRouter) may return it as a
        list of ``{"type": "text", "text": ...}`` parts. Both shapes are handled;
        coercing the list with ``str(...)`` would otherwise emit a Python repr as
        the answer instead of the text.
        """
        choices = data.get("choices")
        text = ""
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            message = choices[0].get("message")
            if isinstance(message, dict):
                text = self._message_text(message.get("content"))
        return LLMResponse(
            text=text,
            citation_chunk_ids=request.citation_chunk_ids,
            raw_provider=self.provider_name,
        )

    @staticmethod
    def _message_text(content: object) -> str:
        """Extract assistant text from an OpenAI-compatible message content.

        Args:
            content: The ``message.content`` value, a string or a list of
                structured content parts.

        Returns:
            The string content, or the concatenated ``text`` of each part; an
            empty string for unrecognized shapes.
        """
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                str(part["text"])
                for part in content
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            )
        return ""


class AnthropicAdapter(HTTPProviderAdapter):
    """Anthropic Claude adapter."""

    provider_name = "anthropic"

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6") -> None:
        """Create an Anthropic adapter."""
        super().__init__(api_key=api_key, model=model)

    @property
    def endpoint(self) -> str:
        """Return Anthropic endpoint URL."""
        return "https://api.anthropic.com/v1/messages"

    @property
    def headers(self) -> Mapping[str, str]:
        """Return Anthropic request headers."""
        return {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

    def payload(self, request: LLMRequest) -> dict[str, object]:
        """Return Anthropic messages payload."""
        return {
            "model": self._model,
            "max_tokens": 1024,
            "temperature": 0.1,
            "messages": [
                {
                    "role": "user",
                    "content": f"Context:\n{request.context}\n\nQuestion:\n{request.prompt}",
                }
            ],
        }

    def parse_response(self, data: Mapping[str, object], request: LLMRequest) -> LLMResponse:
        """Parse Anthropic response JSON.

        Anthropic returns ``content`` as an ordered list of typed blocks. All
        text blocks are concatenated and non-text blocks (for example
        ``thinking`` or ``tool_use``) are skipped, so a leading non-text block
        neither raises nor truncates the answer.
        """
        content = data.get("content")
        text = ""
        if isinstance(content, list):
            text = "".join(
                str(block["text"])
                for block in content
                if isinstance(block, dict) and isinstance(block.get("text"), str)
            )
        return LLMResponse(
            text=text,
            citation_chunk_ids=request.citation_chunk_ids,
            raw_provider=self.provider_name,
        )


class GeminiAdapter(HTTPProviderAdapter):
    """Google Gemini adapter."""

    provider_name = "gemini"

    def __init__(self, api_key: str, model: str = "gemini-3.1-pro-preview") -> None:
        """Create a Gemini adapter."""
        super().__init__(api_key=api_key, model=model)

    @property
    def endpoint(self) -> str:
        """Return Gemini endpoint URL."""
        return f"https://generativelanguage.googleapis.com/v1beta/models/{self._model}:generateContent?key={self._api_key}"

    @property
    def headers(self) -> Mapping[str, str]:
        """Return Gemini request headers."""
        return {"Content-Type": "application/json"}

    def payload(self, request: LLMRequest) -> dict[str, object]:
        """Return Gemini generateContent payload."""
        return {
            "contents": [
                {"parts": [{"text": f"Context:\n{request.context}\n\nQuestion:\n{request.prompt}"}]}
            ]
        }

    def parse_response(self, data: Mapping[str, object], request: LLMRequest) -> LLMResponse:
        """Parse Gemini response JSON."""
        candidates = data.get("candidates")
        text = ""
        if isinstance(candidates, list) and candidates:
            content = candidates[0].get("content", {})
            parts = content.get("parts", []) if isinstance(content, dict) else []
            if isinstance(parts, list):
                text = "".join(
                    str(part["text"])
                    for part in parts
                    if isinstance(part, dict) and isinstance(part.get("text"), str)
                )
        return LLMResponse(
            text=text,
            citation_chunk_ids=request.citation_chunk_ids,
            raw_provider=self.provider_name,
        )


class KimiAdapter(OpenAIAdapter):
    """Moonshot Kimi adapter using its OpenAI-compatible endpoint."""

    provider_name = "kimi"

    def __init__(self, api_key: str, model: str = "kimi-k2") -> None:
        """Create a Kimi adapter."""
        super().__init__(api_key=api_key, model=model)

    @property
    def endpoint(self) -> str:
        """Return Moonshot endpoint URL."""
        return "https://api.moonshot.ai/v1/chat/completions"
