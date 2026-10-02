"""Observe-Decide-Act runtime for Scholar RAG Agent."""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["AgentRunner"]

if TYPE_CHECKING:
    from agent.runner import AgentRunner
else:

    def __getattr__(name: str) -> type[AgentRunner]:
        """Load the runner on demand without making model imports initialize the runtime."""
        if name != "AgentRunner":
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

        from agent.runner import AgentRunner

        globals()[name] = AgentRunner
        return AgentRunner
