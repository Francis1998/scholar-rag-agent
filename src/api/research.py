"""Model-free, bounded research inspection downloads."""

import json
import logging
from collections.abc import Callable, Coroutine
from typing import Literal

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute

from agent.research_worksheet import ResearchWorksheet, WorksheetError, WorksheetRequest
from agent.retrieval_comparison import (
    RetrievalComparison,
    RetrievalComparisonError,
    RetrievalComparisonRequest,
)
from api.dependencies import AppContainer
from storage.paper_collections import CollectionError

logger = logging.getLogger(__name__)
_HEADERS = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}


class _ResearchRoute(APIRoute):
    """Do not echo private or unencodable request input in research validation errors."""

    def get_route_handler(self) -> Callable[[Request], Coroutine[None, None, Response]]:
        handler = super().get_route_handler()

        async def validated(request: Request) -> Response:
            try:
                return await handler(request)
            except RequestValidationError as exc:
                logger.warning("Research request validation failed")
                errors = [
                    {key: error[key] for key in ("loc", "type", "msg")} for error in exc.errors()
                ]
                return Response(
                    content=json.dumps({"detail": errors}, ensure_ascii=True),
                    status_code=422,
                    media_type="application/json",
                    headers=_HEADERS,
                )

        return validated


router = APIRouter(route_class=_ResearchRoute)


@router.post(
    "/research/worksheet",
    response_model=ResearchWorksheet,
    responses={
        200: {"content": {"text/markdown": {"schema": {"type": "string"}}}},
        404: {"description": "Unknown collection"},
        409: {
            "description": "Invalid/stale collection, disappeared papers, or generative retrieval"
        },
        413: {"description": "Serialized worksheet exceeds the aggregate download limit"},
        422: {"description": "Invalid, unknown, ambiguous, unscoped, or oversized selection"},
        500: {"description": "Failed cell or invalid retrieval provenance; no partial result"},
        503: {"description": "Selection storage unavailable"},
        504: {"description": "Overall worksheet or preview phase deadline exceeded"},
    },
)
async def research_worksheet(
    request: Request,
    payload: WorksheetRequest,
    format: Literal["json", "markdown"] = "json",
) -> Response:
    """Inspect each question against each selected paper; never run an answering agent."""
    container: AppContainer = request.app.state.container
    try:
        result = await container.worksheets.build(payload)
        content = result.to_markdown() if format == "markdown" else result.to_json()
    except (WorksheetError, CollectionError) as exc:
        logger.warning("Research worksheet failed: %s", exc.code)
        detail: dict[str, object] = {"code": exc.code, "message": str(exc)}
        if isinstance(exc, CollectionError) and exc.missing_document_ids:
            detail["missing_document_ids"] = list(exc.missing_document_ids)
        if isinstance(exc, WorksheetError):
            if exc.question_index is not None:
                detail["question_index"] = exc.question_index
            if exc.document_index is not None:
                detail["document_index"] = exc.document_index
        raise HTTPException(status_code=exc.status_code, detail=detail, headers=_HEADERS) from exc
    extension = "md" if format == "markdown" else "json"
    return Response(
        content=content,
        media_type="text/markdown" if format == "markdown" else "application/json",
        headers={
            **_HEADERS,
            "Content-Disposition": f'attachment; filename="worksheet.{extension}"',
        },
    )


@router.post(
    "/research/compare-retrieval",
    response_model=RetrievalComparison,
    tags=["research"],
    responses={
        200: {"content": {"text/markdown": {"schema": {"type": "string"}}}},
        404: {"description": "Unknown collection"},
        409: {"description": "Invalid/broken collection or generative retrieval"},
        413: {"description": "Complete JSON or Markdown exceeds 256 KiB; no partial comparison"},
        422: {"description": "Invalid query, shared scope, or the two named policy variants"},
        500: {"description": "Preview, scope, or provenance failure; no partial comparison"},
        503: {"description": "Collection storage unavailable"},
        504: {"description": "Overall comparison or preview phase deadline exceeded"},
    },
)
async def compare_retrieval(
    request: Request,
    payload: RetrievalComparisonRequest,
    format: Literal["json", "markdown"] = "json",
) -> Response:
    """Inspect baseline and candidate policies on the current corpus, without generation.

    Both variants share one query and one resolved scope. Full preview evidence and exact
    source differences are returned together, or not at all. This is not a controlled
    experiment, a corpus snapshot, or a retrieval-quality metric.
    """
    container: AppContainer = request.app.state.container
    try:
        result = await container.retrieval_comparator.compare(payload)
        content = result.to_markdown() if format == "markdown" else result.to_json()
    except (RetrievalComparisonError, CollectionError) as exc:
        logger.warning("Retrieval comparison failed: %s", exc.code)
        detail: dict[str, object] = {"code": exc.code, "message": str(exc)}
        if isinstance(exc, RetrievalComparisonError) and exc.variant is not None:
            detail["variant"] = exc.variant
        raise HTTPException(status_code=exc.status_code, detail=detail, headers=_HEADERS) from exc
    extension = "md" if format == "markdown" else "json"
    return Response(
        content=content,
        media_type="text/markdown" if format == "markdown" else "application/json",
        headers={
            **_HEADERS,
            "Content-Disposition": f'attachment; filename="retrieval-comparison.{extension}"',
        },
    )
