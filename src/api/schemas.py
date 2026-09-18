"""FastAPI request and response schemas."""

from pydantic import BaseModel, Field, field_validator
from pydantic.json_schema import SkipJsonSchema

from agent.models import AgentRunResult
from retrieval.scope import DocumentIds


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
    document_ids: DocumentIds | SkipJsonSchema[None] = Field(
        default=None,
        frozen=True,
        description=(
            "Optional document selection: 1-100 supplied IDs, each 1-128 characters after "
            "outer whitespace trimming. Duplicates keep first-seen order. Omit for the full "
            "corpus; explicit null or empty scope is invalid. Unknown IDs match no chunks."
        ),
    )

    @field_validator("document_ids", mode="before")
    @classmethod
    def reject_explicit_null(cls, value: object) -> object:
        """Omission alone selects the full corpus in the HTTP API."""
        if value is None:
            raise ValueError("document_ids cannot be null; omit it to search the full corpus.")
        return value


class QueryResponse(BaseModel):
    """Response body for agent query execution."""

    result: AgentRunResult
