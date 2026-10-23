"""Provider rate limiting and exponential backoff helpers."""

import asyncio
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from time import monotonic
from typing import TypeVar

T = TypeVar("T")


@dataclass(slots=True)
class AsyncRateLimiter:
    """Per-instance sliding-window admission for callers on one event loop."""

    requests_per_minute: int
    _timestamps: list[float] = field(default_factory=list)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Require a positive integer request capacity without coercion."""
        if (
            not isinstance(self.requests_per_minute, int)
            or isinstance(self.requests_per_minute, bool)
            or self.requests_per_minute <= 0
        ):
            raise ValueError("requests_per_minute must be a positive integer")

    async def acquire(self) -> None:
        """Wait for capacity; cancellation before admission consumes no slot."""
        # One sleeper owns admission; cancellation releases the lock to the next caller.
        async with self._lock:
            while True:
                now = monotonic()
                self._timestamps = [
                    timestamp for timestamp in self._timestamps if now - timestamp < 60.0
                ]
                if len(self._timestamps) < self.requests_per_minute:
                    self._timestamps.append(now)
                    return
                await asyncio.sleep(60.0 - (now - self._timestamps[0]))


async def with_backoff(
    operation: Callable[[], Awaitable[T]],
    retries: int = 3,
    initial_delay_seconds: float = 0.25,
    is_retryable: Callable[[Exception], bool] | None = None,
    minimum_delay_seconds: Callable[[Exception], float | None] | None = None,
) -> T:
    """Run an async operation with exponential backoff for transient failures.

    Args:
        operation: Zero-argument coroutine factory to execute.
        retries: Nonnegative integer retry budget after the initial attempt, excluding bool.
        initial_delay_seconds: Finite nonnegative int/float delay, excluding bool;
            doubled each retry. Must be representable as a finite float.
        is_retryable: Predicate deciding whether a raised exception should be
            retried. When ``None`` every exception is treated as retryable,
            preserving the previous behavior. A non-retryable exception is
            re-raised immediately without consuming further attempts.
        minimum_delay_seconds: Optional per-error minimum wait, returning finite,
            nonnegative seconds or ``None``. Evaluated only before a retry;
            it can extend, but never shorten, the exponential delay.

    Returns:
        The successful operation result.

    Raises:
        ValueError: Invalid retry budget or initial delay, before any operation,
            predicate, delay callback, or sleep, even when retries is zero.
        Exception: The last error raised by ``operation`` once retries are
            exhausted or a non-retryable error is encountered.
    """
    if not isinstance(retries, int) or isinstance(retries, bool) or retries < 0:
        raise ValueError("retries must be a nonnegative integer")
    try:
        valid_delay = (
            isinstance(initial_delay_seconds, (int, float))
            and not isinstance(initial_delay_seconds, bool)
            and math.isfinite(initial_delay_seconds)
            and initial_delay_seconds >= 0
        )
    except OverflowError:
        valid_delay = False
    if not valid_delay:
        raise ValueError("initial_delay_seconds must be a finite nonnegative number")

    retry_predicate = is_retryable if is_retryable is not None else (lambda _exc: True)
    delay_seconds = initial_delay_seconds
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return await operation()
        except Exception as exc:
            last_error = exc
            if attempt == retries or not retry_predicate(exc):
                break
            minimum = minimum_delay_seconds(exc) if minimum_delay_seconds is not None else None
            await asyncio.sleep(
                max(delay_seconds, minimum) if minimum is not None else delay_seconds
            )
            delay_seconds *= 2
    if last_error is None:
        raise RuntimeError("backoff operation failed without an exception")
    raise last_error
