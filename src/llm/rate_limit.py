"""Provider rate limiting and exponential backoff helpers."""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from time import monotonic
from typing import TypeVar

T = TypeVar("T")


@dataclass(slots=True)
class AsyncRateLimiter:
    """Simple per-provider async rate limiter."""

    requests_per_minute: int
    _timestamps: list[float] = field(default_factory=list)

    async def acquire(self) -> None:
        """Wait until a request slot is available."""
        now = monotonic()
        window_start = now - 60.0
        self._timestamps = [
            timestamp for timestamp in self._timestamps if timestamp >= window_start
        ]
        if len(self._timestamps) >= self.requests_per_minute:
            sleep_seconds = 60.0 - (now - self._timestamps[0])
            await asyncio.sleep(max(sleep_seconds, 0.0))
        self._timestamps.append(monotonic())


async def with_backoff(
    operation: Callable[[], Awaitable[T]],
    retries: int = 3,
    initial_delay_seconds: float = 0.25,
) -> T:
    """Run an async operation with exponential backoff for transient failures."""
    delay_seconds = initial_delay_seconds
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return await operation()
        except Exception as exc:
            last_error = exc
            if attempt == retries:
                break
            await asyncio.sleep(delay_seconds)
            delay_seconds *= 2
    if last_error is None:
        raise RuntimeError("backoff operation failed without an exception")
    raise last_error
