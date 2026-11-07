"""Operation failures must remain distinct from the surrounding phase deadline."""

import asyncio
from collections.abc import Generator

import pytest

from agent.safety import with_timeout


class ProviderTimeoutError(TimeoutError):
    """A provider-specific timeout with its own diagnostic."""


@pytest.mark.parametrize("error_type", [TimeoutError, ProviderTimeoutError])
@pytest.mark.parametrize("kind", ["coroutine", "task", "future", "awaitable"])
@pytest.mark.parametrize("immediate", [True, False])
async def test_with_timeout_preserves_operation_timeout(
    error_type: type[TimeoutError], kind: str, immediate: bool
) -> None:
    failure = error_type("provider read timed out after 0.01s")
    cause = asyncio.CancelledError("provider's independent deadline")
    failure.__cause__ = cause
    failure.add_note("request ID: timeout-regression")

    async def fail() -> str:
        if not immediate:
            await asyncio.sleep(0)
        raise failure

    class OperationAwaitable:
        def __await__(self) -> Generator[None, None, str]:
            return fail().__await__()

    if kind == "future":
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        if immediate:
            future.set_exception(failure)
        else:
            asyncio.get_running_loop().call_soon(future.set_exception, failure)
        operation = future
    elif kind == "task":
        operation = asyncio.create_task(fail())
    elif kind == "awaitable":
        operation = OperationAwaitable()
    else:
        operation = fail()

    with pytest.raises(TimeoutError) as caught:
        await with_timeout(operation, 30, "retrieval")

    assert caught.value is failure
    assert str(caught.value) == "provider read timed out after 0.01s"
    assert caught.value.__cause__ is cause
    assert caught.value.__notes__ == ["request ID: timeout-regression"]


@pytest.mark.parametrize("kind", ["coroutine", "task", "future", "awaitable"])
async def test_with_timeout_preserves_successful_awaitables(kind: str) -> None:
    async def succeed() -> str:
        await asyncio.sleep(0)
        return "completed"

    class OperationAwaitable:
        def __await__(self) -> Generator[None, None, str]:
            return succeed().__await__()

    if kind == "future":
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        future.set_result("completed")
        operation = future
    elif kind == "task":
        operation = asyncio.create_task(succeed())
    elif kind == "awaitable":
        operation = OperationAwaitable()
    else:
        operation = succeed()

    assert await with_timeout(operation, 30, "retrieval") == "completed"


async def test_with_timeout_waits_for_deadline_cancellation_cleanup() -> None:
    cleaned_up = asyncio.Event()

    async def block() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            cleaned_up.set()

    operation = asyncio.create_task(block())
    with pytest.raises(TimeoutError, match=r"^retrieval timed out after 0\.1s$"):
        await with_timeout(operation, 0.1, "retrieval")

    assert operation.cancelled()
    assert cleaned_up.is_set()


async def test_with_timeout_preserves_timeout_raised_during_deadline_cleanup() -> None:
    failure = ProviderTimeoutError("provider cleanup timed out")

    async def block() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError as exc:
            raise failure from exc

    operation = asyncio.create_task(block())
    with pytest.raises(TimeoutError) as caught:
        await with_timeout(operation, 0.1, "retrieval")

    assert caught.value is failure
    assert operation.done()
    assert not operation.cancelled()


async def test_with_timeout_preserves_cancellation_suppression() -> None:
    async def block() -> str:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return "cleanup result"

    operation = asyncio.create_task(block())
    assert await with_timeout(operation, 0.1, "retrieval") == "cleanup result"
    assert operation.done()
    assert not operation.cancelled()


@pytest.mark.parametrize("timeout_seconds", [0, -1])
async def test_with_timeout_closes_coroutine_before_nonpositive_deadline(
    timeout_seconds: float,
) -> None:
    started = False

    async def operation() -> None:
        nonlocal started
        started = True

    coroutine = operation()
    with pytest.raises(TimeoutError, match="retrieval timed out after"):
        await with_timeout(coroutine, timeout_seconds, "retrieval")

    assert not started
    assert coroutine.cr_frame is None


async def test_with_timeout_preserves_completed_future_timeout_at_zero_deadline() -> None:
    failure = ProviderTimeoutError("already-failed provider")
    future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
    future.set_exception(failure)

    with pytest.raises(TimeoutError) as caught:
        await with_timeout(future, 0, "retrieval")

    assert caught.value is failure
