"""Tests for provider rate limiting and backoff helpers."""

from unittest.mock import AsyncMock, patch

import pytest

from llm.rate_limit import AsyncRateLimiter, with_backoff


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
