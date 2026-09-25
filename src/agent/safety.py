"""Safety limits, cancellation, and timeout helpers."""

import asyncio
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Annotated, TypeVar

from pydantic import Field, TypeAdapter, ValidationError

T = TypeVar("T")
_TIMEOUT_SECONDS: TypeAdapter[float] = TypeAdapter(
    Annotated[float, Field(gt=0, allow_inf_nan=False)]
)


class CancelledRunError(RuntimeError):
    """Raised when cooperative cancellation is requested."""


@dataclass(slots=True)
class SafetyLimits:
    """Bounded execution limits applied to every request."""

    retrieval_timeout_seconds: float = 30.0
    reasoning_timeout_seconds: float = 60.0
    max_source_docs: int = 50
    max_hops: int = 5

    def __post_init__(self) -> None:
        """Validate phase timeouts on construction and dataclass replacement."""
        for field in ("retrieval_timeout_seconds", "reasoning_timeout_seconds"):
            try:
                seconds = _TIMEOUT_SECONDS.validate_python(getattr(self, field))
            except ValidationError as exc:
                raise ValueError(f"{field} must be a finite, positive number of seconds.") from exc
            setattr(self, field, seconds)

    def clamp_hops(self, requested_hops: int) -> int:
        """Clamp requested graph hops to the configured safe range."""
        return max(0, min(requested_hops, self.max_hops))

    def clamp_sources(self, requested_sources: int) -> int:
        """Clamp requested sources to the configured source-document bound."""
        return max(1, min(requested_sources, self.max_source_docs))


class CancellationToken:
    """Cooperative cancellation token checked between agent phases."""

    def __init__(self) -> None:
        """Create an uncancelled token."""
        self._cancelled = False

    def cancel(self) -> None:
        """Request cancellation for the run."""
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        """Return whether cancellation has been requested."""
        return self._cancelled

    def raise_if_cancelled(self) -> None:
        """Raise if cancellation has been requested."""
        if self._cancelled:
            raise CancelledRunError("agent run was cancelled")


async def with_timeout(awaitable: Awaitable[T], timeout_seconds: float, label: str) -> T:
    """Run an awaitable with a labelled timeout error."""
    try:
        return await asyncio.wait_for(awaitable, timeout=timeout_seconds)
    except TimeoutError as exc:
        raise TimeoutError(f"{label} timed out after {timeout_seconds:.1f}s") from exc
