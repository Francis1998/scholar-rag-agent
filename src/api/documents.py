"""Discover selectable papers and inspect stored evidence without running an agent."""

import logging
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request, Response

from api.dependencies import AppContainer
from storage.document_catalog import (
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    DocumentCatalogError,
    DocumentCatalogPage,
    Identity,
    SourceFilter,
    TitleFilter,
)
from storage.document_chunks import (
    ChunkCursor,
    ChunkCursorError,
    DocumentChunksError,
    DocumentChunksPage,
    DocumentNotFoundError,
)
from storage.source_context import (
    DEFAULT_CONTEXT_NEIGHBORS,
    MAX_CONTEXT_NEIGHBORS,
    ContextChunkIdentity,
    ContextDocumentIdentity,
    SourceContext,
    SourceContextError,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get(
    "/documents",
    response_model=DocumentCatalogPage,
    responses={409: {"description": "Invalid or unselectable saved document summary"}},
)
def list_documents(
    request: Request,
    response: Response,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = DEFAULT_PAGE_SIZE,
    cursor: Annotated[Identity | None, Query()] = None,
    source: Annotated[SourceFilter | None, Query()] = None,
    title: Annotated[TitleFilter | None, Query()] = None,
) -> DocumentCatalogPage:
    """List document IDs, bounded labels and stored chunk counts in ID order."""
    container: AppContainer = request.app.state.container
    headers = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}
    response.headers.update(headers)
    try:
        return container.document_catalog.list_documents(
            limit=limit, cursor=cursor, source=source, title=title
        )
    except DocumentCatalogError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": exc.code, "message": str(exc)},
            headers=headers,
        ) from exc


@router.get(
    "/documents/context",
    response_model=SourceContext,
    responses={
        404: {"description": "Document or exact chunk is not in the current corpus"},
        409: {"description": "Invalid/ambiguous source order, evidence, or stored-data bound"},
        422: {"description": "Invalid exact identifiers or neighbor counts"},
        503: {"description": "Existing corpus storage unavailable"},
    },
)
def read_source_context(
    request: Request,
    response: Response,
    document_id: Annotated[ContextDocumentIdentity, Query()],
    chunk_id: Annotated[ContextChunkIdentity, Query()],
    before: Annotated[int, Query(ge=0, le=MAX_CONTEXT_NEIGHBORS)] = DEFAULT_CONTEXT_NEIGHBORS,
    after: Annotated[int, Query(ge=0, le=MAX_CONTEXT_NEIGHBORS)] = DEFAULT_CONTEXT_NEIGHBORS,
) -> SourceContext:
    """Read a source-order window around an exact current chunk; never expand retrieval."""
    container: AppContainer = request.app.state.container
    headers = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}
    response.headers.update(headers)
    try:
        return container.source_context.read(document_id, chunk_id, before=before, after=after)
    except (DocumentNotFoundError, DocumentChunksError, SourceContextError) as exc:
        logger.warning("Source context read failed: %s", exc.code)
        status = 404 if isinstance(exc, DocumentNotFoundError) else 409
        if isinstance(exc, SourceContextError):
            status = exc.status_code
        raise HTTPException(
            status_code=status,
            detail={"code": exc.code, "message": str(exc)},
            headers=headers,
        ) from exc


@router.get(
    "/documents/{document_id:path}/chunks",
    response_model=DocumentChunksPage,
    responses={
        404: {"description": "Document is not in the stored corpus"},
        409: {"description": "Invalid or unsupported saved chunk evidence"},
        422: {"description": "Invalid document ID, limit, or document-scoped cursor"},
    },
)
def list_document_chunks(
    request: Request,
    response: Response,
    document_id: Identity,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = DEFAULT_PAGE_SIZE,
    cursor: Annotated[ChunkCursor | None, Query()] = None,
) -> DocumentChunksPage:
    """Inspect bounded stored chunk text in exact document scope and ascending ID order."""
    container: AppContainer = request.app.state.container
    headers = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}
    response.headers.update(headers)
    try:
        return container.document_chunks.list_chunks(document_id, limit=limit, cursor=cursor)
    except (DocumentNotFoundError, ChunkCursorError, DocumentChunksError) as exc:
        status = 404 if isinstance(exc, DocumentNotFoundError) else 422
        if isinstance(exc, DocumentChunksError):
            status = 409
        raise HTTPException(
            status_code=status,
            detail={"code": exc.code, "message": str(exc)},
            headers=headers,
        ) from exc
