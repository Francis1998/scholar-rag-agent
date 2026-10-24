"""Bounded GET-only human progress downloads using the collection error contract."""

import logging
from collections.abc import Callable, Coroutine
from typing import Annotated
from urllib.parse import unquote_to_bytes

from fastapi import APIRouter, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from starlette.types import Receive, Scope, Send

from api.collections import COLLECTION_HEADERS, collection_errors
from api.dependencies import AppContainer
from storage.paper_collections import MAX_REVISION, CollectionId
from storage.screening_report import (
    ScreeningReport,
    ScreeningReportFormat,
    export_screening_report,
)

logger = logging.getLogger(__name__)
MAX_QUERY_BYTES = 16_384


def _invalid_request() -> JSONResponse:
    logger.warning("Screening report failed: invalid_screening_report_request")
    return JSONResponse(
        {
            "detail": {
                "code": "invalid_screening_report_request",
                "message": "Use an exact collection ID, one positive ASCII integer "
                "collection_revision and format json or markdown. Unknown, duplicate, "
                "invalid Unicode/NUL and oversized parameters fail.",
            }
        },
        status_code=422,
        headers=COLLECTION_HEADERS,
    )


class _ReportRoute(APIRoute):
    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None:
        if self.methods is not None and scope["method"] not in self.methods:
            logger.warning("Screening report failed: screening_report_method_not_allowed")
            await JSONResponse(
                {
                    "detail": {
                        "code": "screening_report_method_not_allowed",
                        "message": "Screening report downloads require GET; no operation occurred.",
                    }
                },
                status_code=405,
                headers={**COLLECTION_HEADERS, "Allow": "GET"},
            )(scope, receive, send)
            return
        await super().handle(scope, receive, send)

    def get_route_handler(self) -> Callable[[Request], Coroutine[None, None, Response]]:
        handler = super().get_route_handler()

        async def read(request: Request) -> Response:
            try:
                raw = request.scope.get("query_string", b"")
                if len(raw) > MAX_QUERY_BYTES:
                    return _invalid_request()
                for value in (raw, request.scope.get("raw_path", b"")):
                    if "\0" in unquote_to_bytes(value).decode("utf-8"):
                        return _invalid_request()
                if any(
                    key not in {"collection_revision", "format"}
                    or len(request.query_params.getlist(key)) != 1
                    for key in request.query_params
                ):
                    return _invalid_request()
                revision = request.query_params.get("collection_revision", "")
                if not revision.isascii() or not revision.isdecimal() or len(revision) > 19:
                    return _invalid_request()
                return await handler(request)
            except (RequestValidationError, UnicodeError):
                return _invalid_request()

        return read


router = APIRouter(route_class=_ReportRoute, tags=["Human screening report"])


@router.get(
    "/collections/{collection_id}/screening-report",
    response_model=ScreeningReport,
    responses={
        200: {"content": {"text/markdown": {"schema": {"type": "string"}}}},
        404: {"description": "Exact collection missing"},
        405: {"description": "Downloads require GET"},
        409: {"description": "Revision conflict, missing member or invalid saved data"},
        413: {"description": "Complete snapshot or serialized report exceeds its UTF-8 byte cap"},
        422: {"description": "Invalid collection ID, revision, format or query parameters"},
        503: {"description": "Existing collection or screening storage unavailable"},
    },
)
def download_report(
    request: Request,
    collection_id: CollectionId,
    collection_revision: Annotated[int, Query(ge=1, le=MAX_REVISION)],
    format: ScreeningReportFormat = "json",
) -> Response:
    """Outside the screening PUT wildcard so report URLs can never record a decision."""
    container: AppContainer = request.app.state.container
    with collection_errors():
        download = export_screening_report(
            container.paper_screening,
            collection_id,
            collection_revision=collection_revision,
            format=format,
        )
    return Response(
        content=download.content,
        media_type=download.media_type,
        headers={
            **COLLECTION_HEADERS,
            "Content-Disposition": f'attachment; filename="{download.filename}"',
        },
    )
