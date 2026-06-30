"""Tests for provider rate limiting and backoff helpers."""

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from llm.providers import OpenAIAdapter
from llm.rate_limit import AsyncRateLimiter, with_backoff
from llm.schemas import LLMRequest, TaskType


@pytest.mark.asyncio
async def test_rate_limiter_records_request_timestamps() -> None:
    """AsyncRateLimiter records timestamps for admitted requests."""
    limiter = AsyncRateLimiter(requests_per_minute=60)
    with patch("llm.rate_limit.monotonic", return_value=1.0):
        await limiter.acquire()
    assert limiter._timestamps == [1.0]


@pytest.mark.asyncio
async def test_rate_limiter_waits_when_window_is_saturated() -> None:
    """AsyncRateLimiter sleeps when the request window is saturated."""
    limiter = AsyncRateLimiter(requests_per_minute=1)
    limiter._timestamps = [0.0]
    sleep_mock = AsyncMock()
    with (
        patch("llm.rate_limit.monotonic", return_value=0.0),
        patch("llm.rate_limit.asyncio.sleep", sleep_mock),
    ):
        await limiter.acquire()
    sleep_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_with_backoff_retries_transient_failures() -> None:
    """with_backoff retries until the operation succeeds."""
    operation = AsyncMock(side_effect=[RuntimeError("transient"), "ok"])
    result = await with_backoff(operation, retries=2, initial_delay_seconds=0.0)
    assert result == "ok"
    assert operation.call_count == 2


@pytest.mark.asyncio
async def test_with_backoff_raises_after_exhausting_retries() -> None:
    """with_backoff re-raises the last error when retries are exhausted."""
    operation = AsyncMock(side_effect=RuntimeError("permanent"))

    with pytest.raises(RuntimeError, match="permanent"):
        await with_backoff(operation, retries=1, initial_delay_seconds=0.0)

    assert operation.call_count == 2


@pytest.mark.asyncio
async def test_with_backoff_does_not_retry_non_retryable_errors() -> None:
    """A non-retryable error should be raised on the first attempt."""
    operation = AsyncMock(side_effect=ValueError("permanent client error"))

    with pytest.raises(ValueError, match="permanent client error"):
        await with_backoff(
            operation,
            retries=3,
            initial_delay_seconds=0.0,
            is_retryable=lambda exc: not isinstance(exc, ValueError),
        )

    assert operation.call_count == 1


@pytest.mark.asyncio
async def test_provider_does_not_retry_permanent_http_errors() -> None:
    """A permanent 4xx response is surfaced after a single provider call."""
    response = httpx.Response(
        400,
        json={"error": "bad request"},
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
    )
    mock_client = AsyncMock()
    mock_client.post.return_value = response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    adapter = OpenAIAdapter(api_key="test-key")
    request = LLMRequest(task_type=TaskType.DEFAULT, prompt="hi", context="ctx")

    with (
        patch("llm.providers.httpx.AsyncClient", return_value=mock_client),
        pytest.raises(httpx.HTTPStatusError),
    ):
        await adapter.generate(request)

    assert mock_client.post.await_count == 1


@pytest.mark.asyncio
async def test_provider_retries_transient_http_errors() -> None:
    """A transient 503 is retried before a successful response succeeds."""
    error_response = httpx.Response(
        503,
        json={"error": "unavailable"},
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
    )
    ok_response = httpx.Response(
        200,
        json={"choices": [{"message": {"content": "hello"}}]},
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
    )
    mock_client = AsyncMock()
    mock_client.post.side_effect = [error_response, ok_response]
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    adapter = OpenAIAdapter(api_key="test-key")
    request = LLMRequest(task_type=TaskType.DEFAULT, prompt="hi", context="ctx")

    with (
        patch("llm.providers.httpx.AsyncClient", return_value=mock_client),
        patch("llm.rate_limit.asyncio.sleep", AsyncMock()),
    ):
        result = await adapter.generate(request)

    assert result.text == "hello"
    assert mock_client.post.await_count == 2
