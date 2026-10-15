"""Live invocations must own their input across admission, HTTP, and retries."""

import asyncio
import json
from collections.abc import Awaitable, Callable
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from llm.base import BaseLLMAdapter
from llm.providers import (
    AnthropicAdapter,
    GeminiAdapter,
    HTTPProviderAdapter,
    KimiAdapter,
    OpenAIAdapter,
)
from llm.rate_limit import AsyncRateLimiter
from llm.router import ModelRouter, RoutingLLMAdapter
from llm.schemas import LLMRequest, LLMResponse, TaskType

PROVIDERS = [OpenAIAdapter, AnthropicAdapter, GeminiAdapter, KimiAdapter]
TASK_TYPES = {
    "openai": TaskType.DEFAULT,
    "anthropic": TaskType.REASONING,
    "gemini": TaskType.SPEED,
    "kimi": TaskType.COST,
}


class AsyncGate:
    """Expose a suspension point without relying on elapsed time."""

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def wait(self) -> None:
        self.entered.set()
        await self.release.wait()


@pytest.fixture(params=PROVIDERS, ids=lambda provider: provider.provider_name)
def adapter(request: pytest.FixtureRequest) -> HTTPProviderAdapter:
    provider: type[HTTPProviderAdapter] = request.param
    return provider(api_key="test-key", model="configured-model")


@pytest.fixture
def llm_request(adapter: HTTPProviderAdapter) -> LLMRequest:
    return LLMRequest(
        task_type=TASK_TYPES[adapter.provider_name],
        prompt="Summarize the original evidence.",
        context="[c1] Original evidence.",
        citation_chunk_ids=["c1"],
    )


def successful_response(chunk_id: str = "c1") -> httpx.Response:
    text = f"  Synthetic answer [{chunk_id}].\n"
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": text}}],
            "content": [{"type": "text", "text": text}],
            "candidates": [{"content": {"parts": [{"text": text}]}}],
        },
    )


def mock_http(
    monkeypatch: pytest.MonkeyPatch,
    respond: Callable[[httpx.Request], Awaitable[httpx.Response]],
) -> list[httpx.Request]:
    original_client = httpx.AsyncClient
    requests: list[httpx.Request] = []

    async def record(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return await respond(request)

    def create_client(*, timeout: float) -> httpx.AsyncClient:
        return original_client(
            timeout=timeout, transport=httpx.MockTransport(record), trust_env=False
        )

    monkeypatch.setattr("llm.providers.httpx.AsyncClient", create_client)
    return requests


@pytest.mark.parametrize("phase", ["admission", "http", "backoff", "retry-admission"])
@pytest.mark.parametrize("mutation", ["replace-list", "list-alias"])
@pytest.mark.parametrize("routed", [False, True], ids=["direct", "routed"])
async def test_generation_owns_request_before_first_suspension(
    adapter: HTTPProviderAdapter,
    llm_request: LLMRequest,
    phase: str,
    mutation: str,
    routed: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = llm_request.model_dump()
    expected_payload = adapter.payload(llm_request)
    citation_alias = llm_request.citation_chunk_ids
    gate = AsyncGate()
    needs_retry = phase in ("backoff", "retry-admission")
    admissions = 0
    original_acquire = AsyncRateLimiter.acquire

    async def acquire(limiter: AsyncRateLimiter) -> None:
        nonlocal admissions
        admissions += 1
        if (phase == "admission" and admissions == 1) or (
            phase == "retry-admission" and admissions == 2
        ):
            await gate.wait()
        await original_acquire(limiter)

    async def respond(request: httpx.Request) -> httpx.Response:
        if needs_retry and len(requests) == 1:
            return httpx.Response(503, headers={"Retry-After": "2"})
        if phase == "http":
            await gate.wait()
        return successful_response()

    async def sleep(delay: float) -> None:
        if phase == "backoff":
            await gate.wait()

    requests = mock_http(monkeypatch, respond)
    backoff_sleep = AsyncMock(side_effect=sleep)
    monkeypatch.setattr(AsyncRateLimiter, "acquire", acquire)
    monkeypatch.setattr("llm.rate_limit.asyncio.sleep", backoff_sleep)
    generator: BaseLLMAdapter = adapter
    if routed:
        adapters: dict[str, BaseLLMAdapter] = {
            provider.provider_name: provider(api_key="test-key") for provider in PROVIDERS
        }
        adapters[adapter.provider_name] = adapter
        generator = RoutingLLMAdapter(ModelRouter(adapters))

    with patch.object(adapter, "_generate_once", wraps=adapter._generate_once) as generate_once:
        task = asyncio.create_task(generator.generate(llm_request))
        try:
            await asyncio.wait_for(gate.entered.wait(), timeout=5)
            assert not task.done()
            assert len(requests) == (0 if phase == "admission" else 1)
            llm_request.task_type = (
                TaskType.DEFAULT if llm_request.task_type == TaskType.COST else TaskType.COST
            )
            llm_request.prompt = "Use the changed question."
            llm_request.context = "[c2] Changed evidence."
            if mutation == "replace-list":
                llm_request.citation_chunk_ids = ["c2"]
            else:
                citation_alias[:] = ["c2"]
            changed = llm_request.model_dump()
            gate.release.set()
            response = await asyncio.wait_for(task, timeout=5)

            expected_attempts = 2 if needs_retry else 1
            assert [json.loads(request.content) for request in requests] == [
                expected_payload
            ] * expected_attempts
            assert all(str(request.url) == adapter.endpoint for request in requests)
            assert response.model_dump() == {
                "text": "  Synthetic answer [c1].\n",
                "citation_chunk_ids": ["c1"],
                "parsed_claims": [],
                "raw_provider": adapter.provider_name,
                "model_name": "configured-model",
            }
            assert llm_request.model_dump() == changed
            assert admissions == expected_attempts
            assert len(adapter._limiter._timestamps) == expected_attempts
            if needs_retry:
                backoff_sleep.assert_awaited_once_with(2.0)
            else:
                backoff_sleep.assert_not_awaited()
            snapshots = [call.args[0] for call in generate_once.await_args_list]
            assert len(snapshots) == expected_attempts
            assert snapshots[0] is not llm_request
            assert snapshots[0].citation_chunk_ids is not citation_alias
            assert snapshots[0].model_dump() == original
            assert all(snapshot is snapshots[0] for snapshot in snapshots)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def test_overlapping_generations_snapshot_reused_request_independently(
    adapter: HTTPProviderAdapter,
    llm_request: LLMRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gates = [AsyncGate(), AsyncGate()]
    expected_payloads = [adapter.payload(llm_request)]
    citation_alias = llm_request.citation_chunk_ids

    async def respond(request: httpx.Request) -> httpx.Response:
        index = len(requests) - 1
        await gates[index].wait()
        return successful_response(f"c{index + 1}")

    requests = mock_http(monkeypatch, respond)
    tasks: list[asyncio.Task[LLMResponse]] = []
    try:
        tasks.append(asyncio.create_task(adapter.generate(llm_request)))
        await asyncio.wait_for(gates[0].entered.wait(), timeout=5)
        llm_request.prompt = "Summarize the second evidence."
        llm_request.context = "[c2] Second evidence."
        citation_alias[:] = ["c2"]
        expected_payloads.append(adapter.payload(llm_request))
        tasks.append(asyncio.create_task(adapter.generate(llm_request)))
        await asyncio.wait_for(gates[1].entered.wait(), timeout=5)
        llm_request.prompt = "A later question."
        llm_request.context = "[c3] Later evidence."
        citation_alias[:] = ["c3"]

        gates[1].release.set()
        second = await asyncio.wait_for(tasks[1], timeout=5)
        assert not tasks[0].done()
        gates[0].release.set()
        first = await asyncio.wait_for(tasks[0], timeout=5)
        assert [json.loads(request.content) for request in requests] == expected_payloads
        assert first.citation_chunk_ids == ["c1"]
        assert second.citation_chunk_ids == ["c2"]
        assert first.text == "  Synthetic answer [c1].\n"
        assert second.text == "  Synthetic answer [c2].\n"
        second.citation_chunk_ids.append("response-only")
        assert first.citation_chunk_ids == ["c1"]
        assert llm_request.citation_chunk_ids == ["c3"]
        assert llm_request.prompt == "A later question."
        assert llm_request.context == "[c3] Later evidence."
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def test_cancelled_generation_does_not_affect_next_request_snapshot(
    adapter: HTTPProviderAdapter,
    llm_request: LLMRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = AsyncGate()
    expected_payloads = [adapter.payload(llm_request)]

    async def respond(request: httpx.Request) -> httpx.Response:
        if len(requests) == 1:
            await gate.wait()
        return successful_response("c2")

    requests = mock_http(monkeypatch, respond)
    backoff_sleep = AsyncMock()
    monkeypatch.setattr("llm.rate_limit.asyncio.sleep", backoff_sleep)
    task = asyncio.create_task(adapter.generate(llm_request))
    try:
        await asyncio.wait_for(gate.entered.wait(), timeout=5)
        llm_request.prompt = "A new question."
        llm_request.context = "[c2] New evidence."
        llm_request.citation_chunk_ids[:] = ["c2"]
        expected_payloads.append(adapter.payload(llm_request))
        assert task.cancel("cancel in-flight generation")
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
        assert caught.value.args == ("cancel in-flight generation",)
        assert len(requests) == 1
        assert len(adapter._limiter._timestamps) == 1

        response = await asyncio.wait_for(adapter.generate(llm_request), timeout=5)
        assert [json.loads(request.content) for request in requests] == expected_payloads
        assert response.citation_chunk_ids == ["c2"]
        assert llm_request.citation_chunk_ids == ["c2"]
        assert len(adapter._limiter._timestamps) == 2
        backoff_sleep.assert_not_awaited()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
