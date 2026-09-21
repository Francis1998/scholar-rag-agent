"""Generation-free retrieval inspection, separate from the persisted agent-run API."""

import logging

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import ConfigDict, field_validator

from agent.retrieval_preview import RetrievalPreview, RetrievalPreviewError
from api.dependencies import AppContainer
from api.schemas import QueryRequest
from retrieval.scope import scope_arguments

logger = logging.getLogger(__name__)
router = APIRouter()
_PRIVATE_HEADERS = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}


class RetrievalRequest(QueryRequest):
    """The query and optional document selection; no generation or paging options."""

    model_config = ConfigDict(extra="forbid")

    @field_validator("query")
    @classmethod
    def reject_blank_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank.")
        return value


@router.post(
    "/retrieve",
    response_model=RetrievalPreview,
    responses={
        409: {"description": "Configured retrieval would invoke an LLM"},
        422: {"description": "Invalid query, scope, or unsupported request fields"},
        500: {"description": "Planning, retrieval, scope, or context-capture failure"},
        504: {"description": "Retrieval or context-preparation timeout"},
    },
)
async def retrieve(
    request: Request, payload: RetrievalRequest, response: Response
) -> RetrievalPreview:
    """Return bounded prepared evidence without creating a run or invoking a generator."""
    container: AppContainer = request.app.state.container
    try:
        preview = await container.runner.preview(
            payload.query, **scope_arguments(payload.document_ids)
        )
    except RetrievalPreviewError as exc:
        logger.warning("Retrieval preview failed: %s", exc.code)
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": str(exc)},
            headers=_PRIVATE_HEADERS,
        ) from exc
    response.headers.update(_PRIVATE_HEADERS)
    return preview
