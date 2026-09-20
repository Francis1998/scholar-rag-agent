"""Tests for provider rate limiting and backoff helpers."""

import asyncio
from collections.abc import Iterator
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from llm.providers import OpenAIAdapter
from llm.rate_limit import AsyncRateLimiter, with_backoff
from llm.schemas import LLMRequest, LLMResponse, TaskType


class ManualClock:
    """Wake sleepers explicitly without advancing real time."""

    def __init__(self) -> None:
        self.now = 0.0
        self.delays: list[float] = []
        self.sleepers: list[asyncio.Future[None]] = []

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        assert delay > 0.0, "the limiter must not busy-spin with zero-duration sleeps"
        self.delays.append(delay)
        sleeper: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self.sleepers.append(sleeper)
        try:
            await sleeper
        finally:
            self.sleepers.remove(sleeper)

    def wake_at(self, now: float) -> None:
        self.now = now
        for sleeper in self.sleepers:
            if not sleeper.done():
                sleeper.set_result(None)


@pytest.fixture
def clock() -> Iterator[ManualClock]:
    manual_clock = ManualClock()
    with (
        patch("llm.rate_limit.monotonic", manual_clock.monotonic),
        patch("llm.rate_limit.asyncio.sleep", manual_clock.sleep),
    ):
        yield manual_clock


async def checkpoint() -> None:
    """Let already-ready tasks reach their next await, without a timer."""
    ready: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    asyncio.get_running_loop().call_soon(ready.set_result, None)
    await ready


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
        patch("llm.rate_limit.monotonic", side_effect=[0.0, 60.0]),
        patch("llm.rate_limit.asyncio.sleep", sleep_mock),
    ):
        await limiter.acquire()
    sleep_mock.assert_awaited_once_with(60.0)
    assert limiter._timestamps == [60.0]


@pytest.mark.parametrize("requests_per_minute", [0, -1])
def test_rate_limiter_rejects_nonpositive_capacity(requests_per_minute: int) -> None:
    with pytest.raises(ValueError, match="requests_per_minute must be positive"):
        AsyncRateLimiter(requests_per_minute=requests_per_minute)


@pytest.mark.asyncio
async def test_rate_limiter_expires_exact_window_boundary() -> None:
    limiter = AsyncRateLimiter(requests_per_minute=2)
    limiter._timestamps = [0.0, 30.0]
    sleep_mock = AsyncMock()
    with (
        patch("llm.rate_limit.monotonic", return_value=60.0),
        patch("llm.rate_limit.asyncio.sleep", sleep_mock),
    ):
        await limiter.acquire()

    sleep_mock.assert_not_awaited()
    assert limiter._timestamps == [30.0, 60.0]


@pytest.mark.asyncio
@pytest.mark.parametrize("requests_per_minute", [1, 2, 3])
async def test_rate_limiter_bounds_concurrent_admissions(
    clock: ManualClock, requests_per_minute: int
) -> None:
    limiter = AsyncRateLimiter(requests_per_minute=requests_per_minute)
    admissions: list[float] = []

    async def acquire() -> None:
        await limiter.acquire()
        admissions.append(clock.now)

    tasks = [asyncio.create_task(acquire()) for _ in range(requests_per_minute * 3)]
    try:
        await checkpoint()
        assert admissions == [0.0] * requests_per_minute

        for batch, now in enumerate((60.0, 120.0), start=1):
            clock.wake_at(now)
            await asyncio.gather(
                *tasks[batch * requests_per_minute : (batch + 1) * requests_per_minute]
            )
            assert admissions == [
                time for time in (0.0, 60.0, 120.0)[: batch + 1] for _ in range(requests_per_minute)
            ]
            assert limiter._timestamps == [now] * requests_per_minute
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("early_wake", [0.0, 20.0, 59.5])
async def test_rate_limiter_rechecks_after_early_wake(
    clock: ManualClock, early_wake: float
) -> None:
    limiter = AsyncRateLimiter(requests_per_minute=1)
    await limiter.acquire()
    waiting = asyncio.create_task(limiter.acquire())
    try:
        await checkpoint()
        assert clock.delays == [60.0]

        clock.wake_at(early_wake)
        await checkpoint()
        assert not waiting.done()
        assert limiter._timestamps == [0.0]
        assert clock.delays == [60.0, 60.0 - early_wake]

        clock.wake_at(60.0)
        await checkpoint()
        assert waiting.done()
        await waiting
        assert limiter._timestamps == [60.0]
        assert clock.delays == [60.0, 60.0 - early_wake]
    finally:
        waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled_index", [0, 1], ids=["sleeping", "queued"])
async def test_rate_limiter_cancellation_preserves_capacity(
    clock: ManualClock, cancelled_index: int
) -> None:
    limiter = AsyncRateLimiter(requests_per_minute=1)
    await limiter.acquire()
    tasks = [asyncio.create_task(limiter.acquire()) for _ in range(2)]
    try:
        await checkpoint()
        assert clock.delays == [60.0]
        tasks[cancelled_index].cancel()
        with pytest.raises(asyncio.CancelledError):
            await tasks[cancelled_index]
        await checkpoint()
        assert limiter._timestamps == [0.0]

        survivor = tasks[1 - cancelled_index]
        assert not survivor.done()
        clock.wake_at(60.0)
        await checkpoint()
        assert survivor.done(), "cancellation must not leave admission locked"
        await survivor
        assert limiter._timestamps == [60.0]
        assert not clock.sleepers

        clock.wake_at(120.0)
        await limiter.acquire()
        assert limiter._timestamps == [120.0]
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_rate_limiter_instances_do_not_share_capacity(clock: ManualClock) -> None:
    first = AsyncRateLimiter(requests_per_minute=1)
    second = AsyncRateLimiter(requests_per_minute=1)
    await first.acquire()
    waiting = asyncio.create_task(first.acquire())
    try:
        await checkpoint()
        await second.acquire()
        assert second._timestamps == [0.0]
        assert not waiting.done()
    finally:
        waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)


@pytest.mark.asyncio
async def test_rate_limiter_prunes_expired_timestamps_after_waiting() -> None:
    """Timestamps that age out during the wait must not linger in the window.

    A saturated limiter sleeps until the oldest slot frees up. Previously the
    freed slot was left in the list and only the new timestamp was appended, so
    the stale entry kept counting against the cap and throttled throughput below
    the configured rate. After the wait the window must contain only the
    just-admitted request.
    """
    limiter = AsyncRateLimiter(requests_per_minute=1)
    limiter._timestamps = [0.0]
    sleep_mock = AsyncMock()
    with (
        patch("llm.rate_limit.monotonic", side_effect=[0.0, 61.0]),
        patch("llm.rate_limit.asyncio.sleep", sleep_mock),
    ):
        await limiter.acquire()
    sleep_mock.assert_awaited_once()
    assert limiter._timestamps == [61.0]


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
    assert len(adapter._limiter._timestamps) == 1


@pytest.mark.asyncio
async def test_provider_io_does_not_hold_admission_or_refund_cancellation(
    clock: ManualClock,
) -> None:
    adapter = OpenAIAdapter(api_key="test-key")
    request = LLMRequest(task_type=TaskType.DEFAULT, prompt="hi", context="ctx")
    release = asyncio.Event()

    async def generate_once(request: LLMRequest) -> LLMResponse:
        await release.wait()
        return LLMResponse(text="hello", raw_provider="openai")

    with patch.object(adapter, "_generate_once", side_effect=generate_once) as generate_mock:
        tasks = [asyncio.create_task(adapter.generate(request)) for _ in range(2)]
        try:
            await checkpoint()
            assert generate_mock.await_count == 2
            assert adapter._limiter._timestamps == [0.0, 0.0]
            assert not clock.delays

            tasks[0].cancel()
            with pytest.raises(asyncio.CancelledError):
                await tasks[0]
            assert adapter._limiter._timestamps == [0.0, 0.0]
            assert generate_mock.await_count == 2

            release.set()
            await checkpoint()
            assert tasks[1].done()
            assert (await tasks[1]).text == "hello"
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


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
        patch.object(adapter, "_limiter") as limiter,
    ):
        limiter.acquire = AsyncMock()
        result = await adapter.generate(request)

    assert result.text == "hello"
    assert mock_client.post.await_count == 2
    limiter.acquire.assert_awaited_once()
