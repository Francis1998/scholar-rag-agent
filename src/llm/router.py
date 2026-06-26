"""Model routing policy for multi-LLM execution."""

from config import Settings
from llm.base import BaseLLMAdapter
from llm.fake import FakeLLMAdapter
from llm.providers import AnthropicAdapter, GeminiAdapter, KimiAdapter, OpenAIAdapter
from llm.schemas import LLMRequest, LLMResponse, TaskType


class ModelRouter:
    """Route tasks to configured model adapters by task type."""

    def __init__(
        self,
        adapters: dict[str, BaseLLMAdapter] | None = None,
        default_provider: str = "openai",
    ) -> None:
        """Create a router with optional provider adapters."""
        configured_adapters = dict(adapters or {})
        configured_adapters.setdefault("fake", FakeLLMAdapter())
        self._adapters = configured_adapters
        self._default_provider = default_provider

    def route(self, task_type: TaskType) -> BaseLLMAdapter:
        """Return the adapter selected for a task type."""
        preferred = {
            TaskType.REASONING: "anthropic",
            TaskType.SPEED: "gemini",
            TaskType.COST: "kimi",
            TaskType.DEFAULT: self._default_provider,
        }[task_type]
        return (
            self._adapters.get(preferred)
            or self._adapters.get(self._default_provider)
            or self._adapters.get("openai")
            or self._adapters["fake"]
        )


class RoutingLLMAdapter(BaseLLMAdapter):
    """LLM adapter that delegates generation to a task-aware model router."""

    provider_name = "router"

    def __init__(self, router: ModelRouter) -> None:
        """Create an adapter around a model router."""

        self._router = router

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate with the adapter selected for the request task type."""

        return await self._router.route(request.task_type).generate(request)


def build_model_router(settings: Settings) -> ModelRouter:
    """Build a model router from configured provider credentials."""

    adapters: dict[str, BaseLLMAdapter] = {"fake": FakeLLMAdapter()}
    if settings.openai_api_key:
        adapters["openai"] = OpenAIAdapter(api_key=settings.openai_api_key)
    if settings.anthropic_api_key:
        adapters["anthropic"] = AnthropicAdapter(api_key=settings.anthropic_api_key)
    if settings.gemini_api_key:
        adapters["gemini"] = GeminiAdapter(api_key=settings.gemini_api_key)
    if settings.moonshot_api_key:
        adapters["kimi"] = KimiAdapter(api_key=settings.moonshot_api_key)
    return ModelRouter(adapters=adapters, default_provider=settings.default_model)
