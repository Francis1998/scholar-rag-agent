"""LLM request and response schemas."""

from enum import StrEnum

from pydantic import BaseModel, Field


class TaskType(StrEnum):
    """Supported model-routing task types."""

    REASONING = "reasoning"
    SPEED = "speed"
    COST = "cost"
    DEFAULT = "default"


class LLMRequest(BaseModel):
    """Provider-independent LLM request."""

    task_type: TaskType
    prompt: str
    context: str
    citation_chunk_ids: list[str] = Field(default_factory=list)


class LLMResponse(BaseModel):
    """Validated provider-independent LLM response."""

    text: str
    citation_chunk_ids: list[str] = Field(default_factory=list)
    parsed_claims: list[str] = Field(default_factory=list)
    raw_provider: str = "unknown"
