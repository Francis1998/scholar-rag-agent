"""FastAPI request and response schemas."""

from pydantic import BaseModel, Field

from agent.models import AgentRunResult


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"


class IngestTextRequest(BaseModel):
    """Request body for ingesting text content."""

    title: str
    text: str
    source: str = "api"


class IngestResponse(BaseModel):
    """Response body for ingestion requests."""

    document_id: str
    chunk_ids: list[str]


class QueryRequest(BaseModel):
    """Request body for agent query execution."""

    query: str = Field(min_length=1)


class QueryResponse(BaseModel):
    """Response body for agent query execution."""

    result: AgentRunResult
