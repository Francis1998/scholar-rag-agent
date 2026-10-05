"""HTTP-based adapters for OpenAI, Anthropic, Gemini, and Moonshot Kimi."""

import math
import re
import time
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from email.utils import parsedate
from json import JSONDecodeError

import httpx

from llm.base import BaseLLMAdapter
from llm.defaults import (
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_KIMI_MODEL,
    DEFAULT_OPENAI_MODEL,
)
from llm.rate_limit import AsyncRateLimiter, with_backoff
from llm.schemas import LLMRequest, LLMResponse

TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}
_HTTP_DATE = re.compile(
    r"(?:"
    r"(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun), [0-9]{2} [A-Z][a-z]{2} (?P<imf_year>[0-9]{4}) "
    r"[0-9]{2}:[0-9]{2}:[0-9]{2} GMT"
    r"|(?P<rfc850>(?:Mon|Tues|Wednes|Thurs|Fri|Satur|Sun)day, "
    r"[0-9]{2}-[A-Z][a-z]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2} GMT)"
    r"|(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun) [A-Z][a-z]{2} (?:[0-9]{2}| [0-9]) "
    r"[0-9]{2}:[0-9]{2}:[0-9]{2} (?P<asctime_year>[0-9]{4})"
    r")"
)


class ProviderResponseError(ValueError):
    """A successful HTTP response did not contain a usable final answer."""


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


def _retry_after_seconds(exc: Exception) -> float | None:
    """Read an HTTP Retry-After minimum without exposing malformed header values."""
    if not isinstance(exc, httpx.HTTPStatusError):
        return None
    value = exc.response.headers.get("Retry-After")
    if value is None:
        return None
    value = value.strip(" \t")
    if value.isascii() and value.isdigit():
        digits = value.lstrip("0") or "0"
        # A finite float has at most 309 decimal digits; never parse an unbounded int.
        if len(digits) > 309:
            return None
        integer = int(digits)
        try:
            seconds = float(integer)
        except OverflowError:
            return None
        if seconds < integer:
            seconds = math.nextafter(seconds, math.inf)
        return seconds if math.isfinite(seconds) else None

    # The mail parser also accepts non-HTTP forms, zones, and trailing data.
    match = _HTTP_DATE.fullmatch(value)
    if match is None:
        return None
    parts = parsedate(value)
    if parts is None or not 0 <= parts[5] <= 60:
        return None
    now = time.time()
    year, month, day, hour, minute, second = parts[:6]
    explicit_year = match.group("imf_year") or match.group("asctime_year")
    if explicit_year is not None:
        year = int(explicit_year)
    if match.group("rfc850") is not None:
        current = datetime.fromtimestamp(now, UTC)
        year = current.year + (year - current.year) % 100
        if (year, month, day, hour, minute, second) > (
            current.year + 50,
            current.month,
            current.day,
            current.hour,
            current.minute,
            current.second,
        ):
            year -= 100
    try:
        target = datetime(year, month, day, hour, minute, tzinfo=UTC) + timedelta(seconds=second)
    except (ValueError, OverflowError):
        return None
    return max(0.0, target.timestamp() - now)


class HTTPProviderAdapter(BaseLLMAdapter):
    """Base class for optional live HTTP LLM providers."""

    def __init__(self, api_key: str, model: str, requests_per_minute: int = 60) -> None:
        """Create a provider adapter with API key and model name."""
        self._api_key = api_key
        self._model = model
        self._limiter = AsyncRateLimiter(requests_per_minute=requests_per_minute)

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a validated response, admitting each provider attempt."""

        async def admitted_attempt() -> LLMResponse:
            await self._limiter.acquire()
            return await self._generate_once(request)

        return await with_backoff(
            admitted_attempt,
            is_retryable=_is_transient_http_error,
            minimum_delay_seconds=_retry_after_seconds,
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
            try:
                data: object = response.json()
            except (JSONDecodeError, UnicodeDecodeError):
                raise ProviderResponseError(
                    f"{self.provider_name} returned an invalid response: invalid JSON."
                ) from None
            if not isinstance(data, dict):
                raise ProviderResponseError(
                    f"{self.provider_name} returned an invalid response: expected a JSON object."
                )
            return self.parse_response(data, request)

    def _text_response(self, text: str, request: LLMRequest) -> LLMResponse:
        if not text.strip():
            raise ProviderResponseError(
                f"{self.provider_name} returned an invalid response: no nonblank final answer text."
            )
        return LLMResponse(
            text=text,
            citation_chunk_ids=request.citation_chunk_ids,
            raw_provider=self.provider_name,
            model_name=self._model,
        )

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

    def __init__(self, api_key: str, model: str = DEFAULT_OPENAI_MODEL) -> None:
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
            if choices[0].get("finish_reason") == "length":
                raise ProviderResponseError(
                    f"{self.provider_name} returned an invalid response: answer was truncated."
                )
            if choices[0].get("finish_reason") in ("tool_calls", "function_call"):
                raise ProviderResponseError(
                    f"{self.provider_name} returned an invalid response: "
                    "nonfinal tool or continuation output."
                )
            message = choices[0].get("message")
            if isinstance(message, dict):
                tool_calls = message.get("tool_calls")
                if (isinstance(tool_calls, list) and tool_calls) or isinstance(
                    message.get("function_call"), dict
                ):
                    raise ProviderResponseError(
                        f"{self.provider_name} returned an invalid response: "
                        "nonfinal tool or continuation output."
                    )
                text = self._message_text(message.get("content"))
        return self._text_response(text, request)

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
                if isinstance(part, dict)
                and part.get("type", "text") == "text"
                and isinstance(part.get("text"), str)
            )
        return ""


class AnthropicAdapter(HTTPProviderAdapter):
    """Anthropic Claude adapter."""

    provider_name = "anthropic"

    def __init__(self, api_key: str, model: str = DEFAULT_ANTHROPIC_MODEL) -> None:
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
        payload: dict[str, object] = {
            "model": self._model,
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "user",
                    "content": f"Context:\n{request.context}\n\nQuestion:\n{request.prompt}",
                }
            ],
        }
        # Without tools, between_tools yields text only; Sonnet 5.5 rejects disabled.
        if self._model == "claude-sonnet-5-5":
            payload["thinking"] = {"type": "between_tools"}
            payload["output_config"] = {"effort": "medium"}
        elif self._model == "claude-sonnet-5":
            payload["thinking"] = {"type": "disabled"}
        return payload

    def parse_response(self, data: Mapping[str, object], request: LLMRequest) -> LLMResponse:
        """Parse Anthropic response JSON.

        Anthropic returns ``content`` as an ordered list of typed blocks. All
        text blocks are concatenated and thinking blocks are skipped. Tool
        requests and paused turns cannot be completed by this text-only adapter.
        """
        if data.get("stop_reason") in ("max_tokens", "model_context_window_exceeded"):
            raise ProviderResponseError(
                f"{self.provider_name} returned an invalid response: answer was truncated."
            )
        if data.get("stop_reason") in ("tool_use", "pause_turn"):
            raise ProviderResponseError(
                f"{self.provider_name} returned an invalid response: "
                "nonfinal tool or continuation output."
            )
        content = data.get("content")
        text = ""
        if isinstance(content, list):
            if any(
                isinstance(block, dict) and block.get("type") == "tool_use" for block in content
            ):
                raise ProviderResponseError(
                    f"{self.provider_name} returned an invalid response: "
                    "nonfinal tool or continuation output."
                )
            text = "".join(
                str(block["text"])
                for block in content
                if isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            )
        return self._text_response(text, request)


class GeminiAdapter(HTTPProviderAdapter):
    """Google Gemini adapter."""

    provider_name = "gemini"

    def __init__(self, api_key: str, model: str = DEFAULT_GEMINI_MODEL) -> None:
        """Create a Gemini adapter."""
        super().__init__(api_key=api_key, model=model)

    @property
    def endpoint(self) -> str:
        """Return Gemini endpoint URL."""
        return (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self._model}:generateContent"
        )

    @property
    def headers(self) -> Mapping[str, str]:
        """Return Gemini request headers."""
        return {"Content-Type": "application/json", "x-goog-api-key": self._api_key}

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
        if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict):
            if candidates[0].get("finishReason") == "MAX_TOKENS":
                raise ProviderResponseError(
                    f"{self.provider_name} returned an invalid response: answer was truncated."
                )
            if candidates[0].get("finishReason") in (
                "MALFORMED_FUNCTION_CALL",
                "UNEXPECTED_TOOL_CALL",
                "TOO_MANY_TOOL_CALLS",
            ):
                raise ProviderResponseError(
                    f"{self.provider_name} returned an invalid response: "
                    "nonfinal tool or continuation output."
                )
            content = candidates[0].get("content", {})
            parts = content.get("parts", []) if isinstance(content, dict) else []
            if isinstance(parts, list):
                if any(
                    isinstance(part, dict)
                    and (
                        isinstance(part.get("functionCall"), dict)
                        or isinstance(part.get("toolCall"), dict)
                    )
                    for part in parts
                ):
                    raise ProviderResponseError(
                        f"{self.provider_name} returned an invalid response: "
                        "nonfinal tool or continuation output."
                    )
                text = "".join(
                    str(part["text"])
                    for part in parts
                    if isinstance(part, dict)
                    and not part.get("thought", False)
                    and isinstance(part.get("text"), str)
                )
        return self._text_response(text, request)


class KimiAdapter(OpenAIAdapter):
    """Moonshot Kimi adapter using its OpenAI-compatible endpoint."""

    provider_name = "kimi"

    def __init__(self, api_key: str, model: str = DEFAULT_KIMI_MODEL) -> None:
        """Create a Kimi adapter."""
        super().__init__(api_key=api_key, model=model)

    @property
    def endpoint(self) -> str:
        """Return Moonshot endpoint URL."""
        return "https://api.moonshot.ai/v1/chat/completions"
