"""Discover selectable papers without reading their bodies or running an agent."""

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
