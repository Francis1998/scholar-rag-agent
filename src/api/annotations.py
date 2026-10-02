"""Append and read human notes on exact, frozen saved-evidence spans."""

import logging
from collections.abc import Callable, Coroutine
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from pydantic import BeforeValidator
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.dependencies import AppContainer
from storage.evidence_annotations import (
    AnnotationIdentity,
    AnnotationPage,
    AnnotationSubmission,
    SavedEvidenceAnnotation,
)
from storage.evidence_export import EvidenceExportError
from storage.run_history import DEFAULT_PAGE_SIZE, MAX_EVENT_ID, MAX_PAGE_SIZE

logger = logging.getLogger(__name__)
_HEADERS = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}
_ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    404: {"description": "Unknown saved run or frozen chunk"},
    409: {
        "description": "UUID conflict or incomplete, failed, legacy, corrupt, over-limit records"
    },
    413: {"description": "Annotation page exceeds the UTF-8 response-byte limit"},
    422: {"description": "Invalid request fields, pagination, or exact source selector"},
    503: {"description": "Saved annotation or evidence storage unavailable"},
}


def _invalid_request() -> HTTPException:
    logger.warning("Evidence annotation failed: invalid_annotation_request")
    return HTTPException(
        status_code=422,
        detail={
            "code": "invalid_annotation_request",
            "message": "Annotation fields, run ID, or pagination are invalid.",
        },
        headers=_HEADERS,
    )


def _integer_query(value: object) -> int | None:
    if value is None:
        return None
    if type(value) is int:
        return value
    if (
        not isinstance(value, str)
        or not value.isascii()
        or not value.isdecimal()
        or len(value) > 19
    ):
        raise ValueError("Pagination must use unsigned decimal integers.")
    return int(value)


class _AnnotationRoute(APIRoute):
    def get_route_handler(self) -> Callable[[Request], Coroutine[None, None, Response]]:
        handler = super().get_route_handler()

        async def validated(request: Request) -> Response:
            allowed = {"limit", "cursor"} if request.method == "GET" else set()
            if any(
                key not in allowed or len(request.query_params.getlist(key)) != 1
                for key in request.query_params
            ):
                raise _invalid_request()
            try:
                return await handler(request)
            except RequestValidationError as exc:
                raise _invalid_request() from exc
            except StarletteHTTPException as exc:
                if exc.status_code == 400:
                    raise _invalid_request() from exc
                raise
            except EvidenceExportError as exc:
                logger.warning("Evidence annotation failed: %s", exc.code)
                raise HTTPException(
                    status_code=exc.status_code,
                    detail={"code": exc.code, "message": str(exc)},
                    headers=_HEADERS,
                ) from exc

        return validated


router = APIRouter(route_class=_AnnotationRoute, tags=["evidence annotations"])


@router.post(
    "/runs/{run_id}/annotations",
    response_model=SavedEvidenceAnnotation,
    status_code=201,
    responses={
        **_ERROR_RESPONSES,
        200: {"model": SavedEvidenceAnnotation, "description": "Identical retry; original record"},
    },
)
def create_annotation(
    request: Request, run_id: AnnotationIdentity, submission: AnnotationSubmission
) -> Response:
    """Append an opinion, not an authenticated review or a scientific verification."""
    container: AppContainer = request.app.state.container
    saved, created = container.evidence_annotations.create(run_id, submission)
    return Response(
        content=saved.model_dump_json(),
        status_code=201 if created else 200,
        media_type="application/json",
        headers=_HEADERS,
    )


@router.get("/runs/{run_id}/annotations", response_model=AnnotationPage, responses=_ERROR_RESPONSES)
def list_annotations(
    request: Request,
    run_id: AnnotationIdentity,
    limit: Annotated[
        int, Query(ge=1, le=MAX_PAGE_SIZE), BeforeValidator(_integer_query)
    ] = DEFAULT_PAGE_SIZE,
    cursor: Annotated[
        int | None, Query(ge=1, le=MAX_EVENT_ID), BeforeValidator(_integer_query)
    ] = None,
) -> Response:
    """Read newest-first history; next_cursor exclusively selects older sequences."""
    container: AppContainer = request.app.state.container
    page = container.evidence_annotations.list_annotations(run_id, limit=limit, cursor=cursor)
    return Response(content=page.model_dump_json(), media_type="application/json", headers=_HEADERS)
