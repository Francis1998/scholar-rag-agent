"""Read-only discovery of persisted run IDs and bounded query summaries."""

import logging
from collections.abc import Callable, Coroutine
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute

from agent.models import AgentState
from api.dependencies import AppContainer
from api.native_forms import parse_fields
from storage.run_history import (
    DEFAULT_PAGE_SIZE,
    MAX_EVENT_ID,
    MAX_PAGE_SIZE,
    QuestionFilter,
    RunHistoryError,
    RunHistoryPage,
)

logger = logging.getLogger(__name__)
MAX_SEARCH_QUERY_BYTES = 16_384


def _invalid_search() -> JSONResponse:
    logger.warning("Run history search failed: invalid_run_search_request")
    return JSONResponse(
        status_code=422,
        content={
            "detail": {
                "code": "invalid_run_search_request",
                "message": "Use a nonblank literal question of 1-200 Unicode code points "
                "and valid limit/state/cursor values. Search accepts only those four fields, "
                "once each, in at most 16384 UTF-8 URL-encoded query bytes.",
            }
        },
        headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
    )


class _RunHistoryRoute(APIRoute):
    def get_route_handler(self) -> Callable[[Request], Coroutine[None, None, Response]]:
        handler = super().get_route_handler()

        async def read(request: Request) -> Response:
            if "question" not in request.query_params:
                return await handler(request)
            raw = request.scope.get("query_string", b"")
            if len(raw) > MAX_SEARCH_QUERY_BYTES:
                return _invalid_search()
            try:
                fields = parse_fields(raw, max_fields=4)
            except ValueError:
                return _invalid_search()
            if fields.keys() - {"question", "state", "limit", "cursor"}:
                return _invalid_search()
            try:
                return await handler(request)
            except RequestValidationError:
                return _invalid_search()

        return read


router = APIRouter(route_class=_RunHistoryRoute)


@router.get(
    "/runs",
    response_model=RunHistoryPage,
    responses={
        409: {"description": "Malformed or unsupported saved run summary"},
        422: {"description": "Invalid history/search fields, Unicode, URL encoding or bounds"},
    },
)
def list_runs(
    request: Request,
    response: Response,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = DEFAULT_PAGE_SIZE,
    cursor: Annotated[int | None, Query(ge=1, le=MAX_EVENT_ID)] = None,
    state: AgentState | None = None,
    question: Annotated[
        QuestionFilter | None,
        Query(
            description="Case-sensitive literal substring of the complete first PLANNING "
            "question, not its 300-character preview or the answer. No trimming, "
            "normalization, wildcards or relevance ranking. Omit for all questions."
        ),
    ] = None,
) -> RunHistoryPage:
    """List newest-created runs; recorded state does not imply active execution."""
    container: AppContainer = request.app.state.container
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    try:
        return container.run_history.list_runs(
            limit=limit, cursor=cursor, state=state, question=question
        )
    except RunHistoryError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": exc.code, "message": str(exc)},
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        ) from exc
