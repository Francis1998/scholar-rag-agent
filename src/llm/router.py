"""Model routing policy for multi-LLM execution."""

from llm.base import BaseLLMAdapter
from llm.fake import FakeLLMAdapter
from llm.schemas import TaskType


class ModelRouter:
    """Route tasks to configured model adapters by task type."""

    def __init__(self, adapters: dict[str, BaseLLMAdapter] | None = None) -> None:
        """Create a router with optional provider adapters."""
        self._adapters = adapters or {"fake": FakeLLMAdapter()}

    def route(self, task_type: TaskType) -> BaseLLMAdapter:
        """Return the adapter selected for a task type."""
        preferred = {
            TaskType.REASONING: "anthropic",
            TaskType.SPEED: "gemini",
            TaskType.COST: "kimi",
            TaskType.DEFAULT: "openai",
        }[task_type]
        return (
            self._adapters.get(preferred) or self._adapters.get("openai") or self._adapters["fake"]
        )
