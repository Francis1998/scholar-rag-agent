"""Base interface for provider-independent LLM adapters."""

from abc import ABC, abstractmethod

from llm.schemas import LLMRequest, LLMResponse


class BaseLLMAdapter(ABC):
    """Unified asynchronous interface for all LLM providers."""

    provider_name: str

    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a validated response for a provider-independent request."""
