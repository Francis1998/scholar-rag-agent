"""Tests for provider rate limiting and backoff helpers."""

import asyncio
import json
from collections.abc import Iterator
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from llm.providers import (
    AnthropicAdapter,
    GeminiAdapter,
    HTTPProviderAdapter,
    KimiAdapter,
    OpenAIAdapter,
    ProviderResponseError,
)
from llm.rate_limit import AsyncRateLimiter, with_backoff
from llm.schemas import LLMRequest, LLMResponse, TaskType

REQUEST = LLMRequest(
    task_type=TaskType.DEFAULT,
    prompt="Summarize the evidence.",
    context="[c1] Synthetic evidence.",
    citation_chunk_ids=["c1"],
)


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


@pytest.fixture(
    params=[OpenAIAdapter, AnthropicAdapter, GeminiAdapter, KimiAdapter],
    ids=lambda provider: provider.provider_name,
)
def adapter(request: pytest.FixtureRequest) -> HTTPProviderAdapter:
    provider: type[HTTPProviderAdapter] = request.param
    instance = provider(api_key="test-key", model="configured-model")
    instance._limiter = AsyncRateLimiter(requests_per_minute=1)
    return instance


@pytest.fixture(
    params=[429, 500, 502, 503, 504, httpx.ConnectError, httpx.ReadTimeout],
    ids=["429", "500", "502", "503", "504", "connect-error", "read-timeout"],
)
def transient_failures(
    request: pytest.FixtureRequest,
) -> list[httpx.Response | httpx.TransportError]:
    failure: int | type[httpx.TransportError] = request.param
    if isinstance(failure, int):
        return [httpx.Response(failure, json={"error": f"failure {i}"}) for i in range(4)]
    return [failure(f"synthetic transport failure {i}") for i in range(4)]


def successful_response() -> httpx.Response:
    text = "  Synthetic answer [c1].\n"
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": text}}],
            "content": [{"type": "text", "text": text}],
            "candidates": [{"content": {"parts": [{"text": text}]}}],
        },
    )


def mock_provider_http(
    monkeypatch: pytest.MonkeyPatch,
    adapter: HTTPProviderAdapter,
    clock: ManualClock,
    outcomes: list[httpx.Response | httpx.TransportError],
) -> list[float]:
    original_client = httpx.AsyncClient
    request_times: list[float] = []

    def respond(request: httpx.Request) -> httpx.Response:
        request_times.append(clock.now)
        assert request.method == "POST"
        assert request.url == adapter.endpoint
        assert json.loads(request.content) == adapter.payload(REQUEST)
        for name, value in adapter.headers.items():
            assert request.headers[name] == value
        assert len(request_times) <= len(outcomes), "unexpected extra HTTP attempt"
        outcome = outcomes[len(request_times) - 1]
        if isinstance(outcome, httpx.TransportError):
            raise outcome
        return outcome

    def create_client(*, timeout: float) -> httpx.AsyncClient:
        return original_client(
            timeout=timeout, transport=httpx.MockTransport(respond), trust_env=False
        )

    monkeypatch.setattr("llm.providers.httpx.AsyncClient", create_client)
    return request_times


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
    assert limiter.acquire.await_count == 2


async def test_provider_retries_wait_for_each_http_admission(
    adapter: HTTPProviderAdapter,
    clock: ManualClock,
    monkeypatch: pytest.MonkeyPatch,
    transient_failures: list[httpx.Response | httpx.TransportError],
) -> None:
    request_times = mock_provider_http(
        monkeypatch, adapter, clock, [*transient_failures[:3], successful_response()]
    )
    task = asyncio.create_task(adapter.generate(REQUEST))
    try:
        await checkpoint()
        for attempt, delay in enumerate((0.25, 0.5, 1.0), start=1):
            previous_time = (attempt - 1) * 60.0
            expected_times = [i * 60.0 for i in range(attempt)]
            assert request_times == expected_times
            assert adapter._limiter._timestamps == [previous_time]
            assert clock.delays[-1] == delay

            clock.wake_at(previous_time + delay)
            await checkpoint()
            assert request_times == expected_times
            assert not task.done()
            assert clock.delays[-1] == 60.0 - delay

            clock.wake_at(previous_time + 30.0)
            await checkpoint()
            assert request_times == expected_times
            assert adapter._limiter._timestamps == [previous_time]
            assert clock.delays[-1] == 30.0

            clock.wake_at(attempt * 60.0)
            await checkpoint()

        assert task.done()
        result = await task
        assert request_times == [0.0, 60.0, 120.0, 180.0]
        assert adapter._limiter._timestamps == [180.0]
        assert result.text == "  Synthetic answer [c1].\n"
        assert result.citation_chunk_ids == REQUEST.citation_chunk_ids
        assert result.raw_provider == adapter.provider_name
        assert result.model_name == "configured-model"
        assert not clock.sleepers
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_provider_exhaustion_counts_all_attempts_and_preserves_last_error(
    adapter: HTTPProviderAdapter,
    clock: ManualClock,
    monkeypatch: pytest.MonkeyPatch,
    transient_failures: list[httpx.Response | httpx.TransportError],
) -> None:
    adapter._limiter = AsyncRateLimiter(requests_per_minute=4)
    request_times = mock_provider_http(monkeypatch, adapter, clock, transient_failures)
    task = asyncio.create_task(adapter.generate(REQUEST))
    try:
        await checkpoint()
        assert request_times == [0.0]
        for now in (0.25, 0.75, 1.75):
            clock.wake_at(now)
            await checkpoint()
        assert task.done()
        last_failure = transient_failures[-1]
        if isinstance(last_failure, httpx.TransportError):
            with pytest.raises(type(last_failure)) as caught_transport:
                await task
            assert caught_transport.value is last_failure
        else:
            with pytest.raises(httpx.HTTPStatusError) as caught_status:
                await task
            assert caught_status.value.response is last_failure
            assert caught_status.value.request is last_failure.request
        assert request_times == [0.0, 0.25, 0.75, 1.75]
        assert adapter._limiter._timestamps == request_times
        assert clock.delays == [0.25, 0.5, 1.0]
        assert not clock.sleepers
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("status_code", [400, 401, 200], ids=["400", "401", "invalid-success"])
async def test_nonretryable_provider_errors_consume_one_http_admission(
    adapter: HTTPProviderAdapter,
    clock: ManualClock,
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
) -> None:
    response = httpx.Response(status_code, json={})
    request_times = mock_provider_http(monkeypatch, adapter, clock, [response])
    task = asyncio.create_task(adapter.generate(REQUEST))
    try:
        await checkpoint()
        assert task.done()
        if status_code == 200:
            with pytest.raises(ProviderResponseError, match="no nonblank final answer text"):
                await task
        else:
            with pytest.raises(httpx.HTTPStatusError) as caught:
                await task
            assert caught.value.response is response
        assert request_times == [0.0]
        assert adapter._limiter._timestamps == [0.0]
        assert not clock.delays
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("phase", ["initial-admission", "retry-admission", "backoff"])
async def test_provider_cancellation_sends_no_extra_request_or_claims_future_capacity(
    adapter: HTTPProviderAdapter,
    clock: ManualClock,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    first_response = successful_response() if phase == "initial-admission" else httpx.Response(503)
    request_times = mock_provider_http(
        monkeypatch, adapter, clock, [first_response, successful_response()]
    )
    if phase == "initial-admission":
        await adapter.generate(REQUEST)
    tasks = [asyncio.create_task(adapter.generate(REQUEST))]
    try:
        await checkpoint()
        if phase == "retry-admission":
            clock.wake_at(0.25)
            await checkpoint()
        assert request_times == [0.0]
        assert not tasks[0].done()
        assert adapter._limiter._timestamps == [0.0]

        assert tasks[0].cancel("cancel waiting generation")
        with pytest.raises(asyncio.CancelledError) as caught:
            await tasks[0]
        assert caught.value.args == ("cancel waiting generation",)
        assert adapter._limiter._timestamps == [0.0]
        assert request_times == [0.0]
        assert not clock.sleepers

        tasks.append(asyncio.create_task(adapter.generate(REQUEST)))
        await checkpoint()
        assert request_times == [0.0]
        assert not tasks[1].done()
        clock.wake_at(60.0)
        await checkpoint()
        assert tasks[1].done()
        assert (await tasks[1]).text == "  Synthetic answer [c1].\n"
        assert request_times == [0.0, 60.0]
        assert adapter._limiter._timestamps == [60.0]
        assert not clock.sleepers
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def test_concurrent_generation_and_retry_share_http_capacity(
    adapter: HTTPProviderAdapter, clock: ManualClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    request_times = mock_provider_http(
        monkeypatch,
        adapter,
        clock,
        [httpx.Response(503), successful_response(), successful_response()],
    )
    tasks = [asyncio.create_task(adapter.generate(REQUEST)) for _ in range(2)]
    try:
        await checkpoint()
        assert request_times == [0.0]
        clock.wake_at(0.25)
        await checkpoint()
        assert request_times == [0.0]

        clock.wake_at(60.0)
        await checkpoint()
        assert request_times == [0.0, 60.0]
        assert tasks[1].done()
        assert not tasks[0].done()
        assert adapter._limiter._timestamps == [60.0]

        clock.wake_at(120.0)
        await checkpoint()
        assert all(task.done() for task in tasks)
        results = await asyncio.gather(*tasks)
        assert all(result.raw_provider == adapter.provider_name for result in results)
        assert request_times == [0.0, 60.0, 120.0]
        assert adapter._limiter._timestamps == [120.0]
        assert not clock.sleepers
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def test_generate_once_override_is_admitted_before_each_attempt(clock: ManualClock) -> None:
    attempt_times: list[float] = []

    class CustomAdapter(OpenAIAdapter):
        async def _generate_once(self, request: LLMRequest) -> LLMResponse:
            attempt_times.append(clock.now)
            if len(attempt_times) == 1:
                raise httpx.ConnectError("synthetic custom transport failure")
            return self._text_response("custom answer", request)

    adapter = CustomAdapter(api_key="test-key")
    adapter._limiter = AsyncRateLimiter(requests_per_minute=1)
    task = asyncio.create_task(adapter.generate(REQUEST))
    try:
        await checkpoint()
        assert attempt_times == [0.0]
        clock.wake_at(0.25)
        await checkpoint()
        assert attempt_times == [0.0]
        assert not task.done()
        clock.wake_at(60.0)
        await checkpoint()
        assert task.done()
        assert (await task).text == "custom answer"
        assert attempt_times == [0.0, 60.0]
        assert adapter._limiter._timestamps == [60.0]
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
