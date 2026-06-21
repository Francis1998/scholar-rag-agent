"""HyDE query expansion for retrieval."""

from llm.base import BaseLLMAdapter
from llm.schemas import LLMRequest, TaskType


class HyDEExpander:
    """Generate hypothetical-document expansion text for retrieval."""

    def __init__(self, llm: BaseLLMAdapter | None = None) -> None:
        """Create a HyDE expander with an optional LLM adapter."""
        self._llm = llm

    async def expand(self, query: str) -> str:
        """Return query plus hypothetical scientific answer text."""
        if self._llm is None:
            hypothetical_abstract = (
                f"{query} is discussed with methods, evidence, limitations, and findings."
            )
            return f"{query}\nHypothetical abstract: {hypothetical_abstract}"
        response = await self._llm.generate(
            LLMRequest(
                task_type=TaskType.SPEED,
                prompt=(
                    "Write a concise hypothetical scientific abstract that would answer: "
                    f"{query}"
                ),
                context="",
                citation_chunk_ids=[],
            )
        )
        return f"{query}\n{response.text}"
