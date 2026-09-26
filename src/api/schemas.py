"""FastAPI request and response schemas."""

from typing import Self

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic.json_schema import SkipJsonSchema

from agent.models import AgentRunResult
from retrieval.scope import DocumentIds
from storage.paper_collections import CollectionId


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

    query: str = Field(
        min_length=1,
        description="Research question with non-whitespace text; validation does not trim it.",
    )
    document_ids: DocumentIds | SkipJsonSchema[None] = Field(
        default=None,
        frozen=True,
        description=(
            "Optional document selection: 1-100 supplied IDs, each 1-128 characters after "
            "outer whitespace trimming. Duplicates keep first-seen order. Omit for the full "
            "corpus; explicit null or empty scope is invalid. Unknown IDs match no chunks."
        ),
    )

    collection_id: CollectionId | SkipJsonSchema[None] = Field(
        default=None,
        frozen=True,
        description=(
            "Saved paper collection ID. Resolved to document_ids before query/preview work. "
            "Mutually exclusive with document_ids; omit both for the full corpus. "
            "Null, malformed, unknown, or broken collections never widen scope."
        ),
    )

    @field_validator("query")
    @classmethod
    def reject_blank_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank.")
        return value

    @field_validator("document_ids", "collection_id", mode="before")
    @classmethod
    def reject_explicit_null(cls, value: object) -> object:
        """Omission alone selects the full corpus in the HTTP API."""
        if value is None:
            raise ValueError("Scope cannot be null; omit it to search the full corpus.")
        return value

    @model_validator(mode="after")
    def exclusive_scope(self) -> Self:
        if self.document_ids is not None and self.collection_id is not None:
            raise ValueError("Supply either collection_id or document_ids, not both.")
        return self


class QueryResponse(BaseModel):
    """Response body for agent query execution."""

    result: AgentRunResult
