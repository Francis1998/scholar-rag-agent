"""Read-only JSON ordering of papers still needing a human screening decision."""

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
from storage.screening_priority import ScreeningPriority, prioritize_screening

logger = logging.getLogger(__name__)
MAX_QUERY_BYTES = 16_384


def _invalid_request() -> JSONResponse:
    logger.warning("Screening priority failed: invalid_screening_priority_request")
    return JSONResponse(
        {
            "detail": {
                "code": "invalid_screening_priority_request",
                "message": "Use an exact collection ID and one positive ASCII integer "
                "collection_revision. Unknown, duplicate, invalid Unicode/NUL and oversized "
                "parameters fail.",
            }
        },
        status_code=422,
        headers=COLLECTION_HEADERS,
    )


class _PriorityRoute(APIRoute):
    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None:
        if self.methods is not None and scope["method"] not in self.methods:
            logger.warning("Screening priority failed: screening_priority_method_not_allowed")
            await JSONResponse(
                {
                    "detail": {
                        "code": "screening_priority_method_not_allowed",
                        "message": "The screening ordering is read with GET; nothing was "
                        "recorded. Use the screening PUT route to record a human decision.",
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
                    key != "collection_revision" or len(request.query_params.getlist(key)) != 1
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


router = APIRouter(route_class=_PriorityRoute, tags=["Human screening prioritization"])


@router.get(
    "/collections/{collection_id}/screening-priority",
    response_model=ScreeningPriority,
    responses={
        404: {"description": "Exact collection missing"},
        405: {"description": "Read with GET; nothing is recorded"},
        409: {"description": "Revision conflict, missing member or invalid saved data"},
        413: {"description": "Complete ordering exceeds its UTF-8 byte limit"},
        422: {"description": "Invalid collection ID, revision or query parameters"},
        503: {"description": "Existing collection or screening storage unavailable"},
    },
)
def screening_priority(
    request: Request,
    response: Response,
    collection_id: CollectionId,
    collection_revision: Annotated[int, Query(ge=1, le=MAX_REVISION)],
) -> ScreeningPriority:
    """Order pending members most-likely-relevant first; never record or suggest a decision.

    Outside the screening PUT wildcard, so this URL can never record a decision for a paper
    whose ID happens to be ``priority``.
    """
    container: AppContainer = request.app.state.container
    with collection_errors():
        result = prioritize_screening(
            container.paper_screening, collection_id, collection_revision=collection_revision
        )
    response.headers.update(COLLECTION_HEADERS)
    return result
