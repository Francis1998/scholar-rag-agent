"""Core API routes, independent of application and container initialization."""

from fastapi import APIRouter, Request

from api.dependencies import AppContainer
from api.schemas import (
    HealthResponse,
    IngestResponse,
    IngestTextRequest,
    QueryRequest,
    QueryResponse,
)
from ingestion.chunking import stable_id
from retrieval.models import Document
from retrieval.scope import scope_arguments

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return service health."""
    return HealthResponse()


@router.post("/ingest/text", response_model=IngestResponse)
async def ingest_text(request: Request, payload: IngestTextRequest) -> IngestResponse:
    """Ingest raw text as a scientific document."""
    container: AppContainer = request.app.state.container
    document = Document(
        document_id=stable_id(f"{payload.source}:{payload.title}:{payload.text}", "doc"),
        title=payload.title,
        text=payload.text,
        source=payload.source,
        metadata={"source_type": "api"},
    )
    chunks = container.ingestion_pipeline.ingest_documents([document])
    return IngestResponse(
        document_id=document.document_id, chunk_ids=[chunk.chunk_id for chunk in chunks]
    )


@router.post("/query", response_model=QueryResponse)
async def query(request: Request, payload: QueryRequest) -> QueryResponse:
    """Execute an Observe-Decide-Act RAG query."""
    container: AppContainer = request.app.state.container
    result = await container.runner.run(payload.query, **scope_arguments(payload.document_ids))
    return QueryResponse(result=result)


@router.get("/runs/{run_id}/events")
async def run_events(request: Request, run_id: str) -> list[dict[str, object]]:
    """Return durable events for a run."""
    container: AppContainer = request.app.state.container
    return container.event_log.list_events(run_id)
