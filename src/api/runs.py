"""Read-only discovery of persisted run IDs and bounded query summaries."""

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request, Response

from agent.models import AgentState
from api.dependencies import AppContainer
from storage.run_history import (
    DEFAULT_PAGE_SIZE,
    MAX_EVENT_ID,
    MAX_PAGE_SIZE,
    RunHistoryError,
    RunHistoryPage,
)

router = APIRouter()


@router.get(
    "/runs",
    response_model=RunHistoryPage,
    responses={409: {"description": "Malformed or unsupported saved run summary"}},
)
def list_runs(
    request: Request,
    response: Response,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = DEFAULT_PAGE_SIZE,
    cursor: Annotated[int | None, Query(ge=1, le=MAX_EVENT_ID)] = None,
    state: AgentState | None = None,
) -> RunHistoryPage:
    """List newest-created runs; recorded state does not imply active execution."""
    container: AppContainer = request.app.state.container
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    try:
        return container.run_history.list_runs(limit=limit, cursor=cursor, state=state)
    except RunHistoryError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": exc.code, "message": str(exc)},
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        ) from exc
