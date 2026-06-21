"""Deterministic fake LLM adapter for tests and local demos."""

from llm.base import BaseLLMAdapter
from llm.schemas import LLMRequest, LLMResponse


class FakeLLMAdapter(BaseLLMAdapter):
    """Return deterministic citation-aware responses without network access."""

    provider_name = "fake"

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a deterministic answer grounded in supplied chunk ids."""
        citations = request.citation_chunk_ids[:3]
        citation_text = ", ".join(f"[{chunk_id}]" for chunk_id in citations) or "[no-citation]"
        text = f"The retrieved literature indicates: {request.prompt} {citation_text}"
        return LLMResponse(
            text=text,
            citation_chunk_ids=citations,
            parsed_claims=[f"Retrieved evidence addresses {request.prompt}"],
            raw_provider=self.provider_name,
        )
