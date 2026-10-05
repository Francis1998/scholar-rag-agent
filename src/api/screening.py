"""Explicit human screening metadata, independent of retrieval and generation."""

import logging
from collections.abc import Callable, Coroutine
from typing import Annotated, Any

from fastapi import APIRouter, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute

from api.collections import COLLECTION_HEADERS, collection_errors
from api.dependencies import AppContainer
from storage.document_catalog import Identity
from storage.paper_collections import MAX_REVISION, CollectionId
from storage.paper_screening import (
    ScreeningExport,
    ScreeningExportFormat,
    ScreeningQueue,
    ScreeningReview,
    ScreeningStatus,
    ScreeningSubmission,
)

logger = logging.getLogger(__name__)


def _invalid_request() -> JSONResponse:
    logger.warning("Screening operation failed: invalid_screening_request")
    return JSONResponse(
        status_code=422,
        content={
            "detail": {
                "code": "invalid_screening_request",
                "message": "Invalid screening request; check fields and bounds.",
            }
        },
        headers=COLLECTION_HEADERS,
    )


class _ScreeningRoute(APIRoute):
    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        original = super().get_route_handler()

        async def handler(request: Request) -> Response:
            try:
                return await original(request)
            except RequestValidationError:
                return _invalid_request()

        return handler


router = APIRouter(
    prefix="/collections/{collection_id}/screening",
    tags=["paper screening"],
    route_class=_ScreeningRoute,
)


@router.get("", response_model=ScreeningQueue)
def list_screening(
    request: Request,
    response: Response,
    collection_id: CollectionId,
    collection_revision: Annotated[int, Query(ge=1, le=MAX_REVISION)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    cursor: Annotated[Identity | None, Query()] = None,
    status: Annotated[ScreeningStatus | None, Query()] = None,
) -> ScreeningQueue:
    container: AppContainer = request.app.state.container
    with collection_errors():
        result = container.paper_screening.list_queue(
            collection_id,
            collection_revision=collection_revision,
            limit=limit,
            cursor=cursor,
            status=status,
        )
    response.headers.update(COLLECTION_HEADERS)
    return result


@router.get(
    "/export",
    response_model=ScreeningExport,
    responses={
        200: {"content": {"text/csv": {"schema": {"type": "string"}}}},
        404: {"description": "Unknown or deleted collection"},
        409: {"description": "Stale revision, missing document, or invalid saved metadata"},
        413: {"description": "Complete export exceeds its UTF-8 byte limit; no partial export"},
        422: {
            "description": "Invalid revision, format, duplicate, or unsupported query parameters"
        },
        503: {"description": "Collection or screening storage unavailable"},
    },
)
def export_screening(
    request: Request,
    collection_id: CollectionId,
    collection_revision: Annotated[int, Query(ge=1, le=MAX_REVISION)],
    format: ScreeningExportFormat = "json",
) -> Response:
    """Download all members; never filter, page, retrieve, generate, or write."""
    revision_text = request.query_params["collection_revision"]
    if (
        not revision_text.isascii()
        or not revision_text.isdecimal()
        or len(revision_text) > 19
        or any(
            key not in {"collection_revision", "format"}
            or len(request.query_params.getlist(key)) != 1
            for key in request.query_params
        )
    ):
        return _invalid_request()
    container: AppContainer = request.app.state.container
    with collection_errors():
        download = container.paper_screening.export_results(
            collection_id, collection_revision=collection_revision, format=format
        )
    return Response(
        content=download.content,
        media_type=download.media_type,
        headers={
            **COLLECTION_HEADERS,
            "Content-Disposition": f'attachment; filename="{download.filename}"',
        },
    )


@router.put("/{document_id:path}", response_model=ScreeningReview)
def submit_screening(
    request: Request,
    response: Response,
    collection_id: CollectionId,
    document_id: Identity,
    payload: ScreeningSubmission,
) -> ScreeningReview:
    container: AppContainer = request.app.state.container
    with collection_errors():
        result = container.paper_screening.submit(collection_id, document_id, payload)
    response.headers.update(COLLECTION_HEADERS)
    return result
