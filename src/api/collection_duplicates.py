"""Read-only, all-or-nothing duplicate-candidate downloads for an explicit collection."""

import logging
from collections.abc import Callable, Coroutine
from typing import Annotated, Any

from fastapi import APIRouter, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute

from api.collections import COLLECTION_HEADERS, collection_errors
from api.dependencies import AppContainer
from storage.collection_duplicates import CollectionDuplicateReport, DuplicateFormat
from storage.paper_collections import MAX_REVISION, CollectionId

logger = logging.getLogger(__name__)


def _invalid_request() -> JSONResponse:
    logger.warning("Duplicate inspection failed: invalid_duplicate_request")
    return JSONResponse(
        status_code=422,
        content={
            "detail": {
                "code": "invalid_duplicate_request",
                "message": "Provide a collection ID, required integer collection_revision, "
                "finite title_threshold in [0, 1], and format json or markdown. "
                "Repeated or unsupported parameters are not accepted.",
            }
        },
        headers=COLLECTION_HEADERS,
    )


class _DuplicateRoute(APIRoute):
    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        original = super().get_route_handler()

        async def handler(request: Request) -> Response:
            try:
                return await original(request)
            except RequestValidationError:
                return _invalid_request()

        return handler


router = APIRouter(
    prefix="/collections/{collection_id}/duplicates",
    tags=["collection duplicates"],
    route_class=_DuplicateRoute,
)


@router.get(
    "",
    response_model=CollectionDuplicateReport,
    responses={
        200: {"content": {"text/markdown": {"schema": {"type": "string"}}}},
        404: {"description": "Unknown collection"},
        409: {"description": "Stale revision, missing member, or invalid stored metadata"},
        413: {"description": "Input or complete JSON/Markdown report exceeds a fixed bound"},
        422: {"description": "Invalid, missing, duplicate, or unsupported request parameters"},
        503: {"description": "Existing collection or document storage unavailable"},
    },
)
def inspect_duplicates(
    request: Request,
    collection_id: CollectionId,
    collection_revision: Annotated[int, Query(ge=1, le=MAX_REVISION)],
    title_threshold: Annotated[float, Query(ge=0.0, le=1.0, allow_inf_nan=False)] = 0.92,
    format: Annotated[DuplicateFormat, Query()] = "json",
) -> Response:
    revision_text = request.query_params["collection_revision"]
    if (
        not revision_text.isascii()
        or not revision_text.isdecimal()
        or len(revision_text) > 19
        or any(
            key not in {"collection_revision", "title_threshold", "format"}
            or len(request.query_params.getlist(key)) != 1
            for key in request.query_params
        )
    ):
        return _invalid_request()
    container: AppContainer = request.app.state.container
    with collection_errors():
        report = container.collection_duplicates.inspect(
            collection_id,
            collection_revision=collection_revision,
            title_threshold=title_threshold,
        )
        content = report.to_json() if format == "json" else report.to_markdown()
    extension = "json" if format == "json" else "md"
    return Response(
        content=content,
        media_type="application/json" if format == "json" else "text/markdown",
        headers={
            **COLLECTION_HEADERS,
            "Content-Disposition": f'attachment; filename="duplicates-{collection_id}'
            f'-r{report.collection_revision}.{extension}"',
        },
    )
