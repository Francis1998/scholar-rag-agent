"""Shared document and retrieval models."""

from pydantic import BaseModel, Field


class SourceRef(BaseModel):
    """Reference to an input paper source."""

    source_type: str
    value: str


class Document(BaseModel):
    """Normalized scientific document."""

    document_id: str
    title: str
    text: str
    source: str
    metadata: dict[str, str] = Field(default_factory=dict)


class Chunk(BaseModel):
    """Searchable text chunk with document provenance."""

    chunk_id: str
    document_id: str
    title: str
    text: str
    source: str
    metadata: dict[str, str] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """Scored retrieval result."""

    chunk: Chunk
    score: float
    retriever: str
    path: list[str] = Field(default_factory=list)


class Entity(BaseModel):
    """Entity extracted from scientific text."""

    name: str
    label: str = "ENTITY"


class EntityEdge(BaseModel):
    """Undirected relationship between entities observed in one chunk."""

    source: str
    target: str
    chunk_id: str
    weight: float = 1.0
