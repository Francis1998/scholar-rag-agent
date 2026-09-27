"""Record and inspect human opinions without altering the saved answer or run events."""

from collections.abc import Callable, Coroutine
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute

from agent.review_models import (
    DEFAULT_REVIEW_LIMIT,
    MAX_REVIEW_LIMIT,
    MAX_REVIEW_SEQUENCE,
    AnswerReviewError,
    ReviewHistoryPage,
    ReviewIdentity,
    ReviewSubmission,
    SavedAnswerReview,
)
from api.dependencies import AppContainer

_HEADERS = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}
_ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    404: {"description": "Unknown saved run"},
    409: {"description": "Incomplete, failed, legacy, corrupt evidence, or corrupt reviews"},
    503: {"description": "Review or evidence storage unavailable"},
}


class _ReviewRoute(APIRoute):
    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        handler = super().get_route_handler()

        async def validate(request: Request) -> Response:
            try:
                return await handler(request)
            except RequestValidationError as exc:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "code": "invalid_review_request",
                        "message": "Review fields, run ID, or pagination parameters are invalid.",
                    },
                    headers=_HEADERS,
                ) from exc

        return validate


router = APIRouter(route_class=_ReviewRoute)


def _http_error(exc: AnswerReviewError) -> HTTPException:
    return HTTPException(
        status_code=exc.status_code,
        detail={"code": exc.code, "message": str(exc)},
        headers=_HEADERS,
    )


@router.post(
    "/runs/{run_id}/reviews",
    response_model=SavedAnswerReview,
    status_code=201,
    responses={
        **_ERROR_RESPONSES,
        200: {"model": SavedAnswerReview, "description": "Identical retry; original record"},
        409: {"description": "Review ID conflict or invalid saved evidence/review"},
        422: {"description": "Invalid review fields or out-of-bundle cited chunk IDs"},
    },
)
def create_review(
    request: Request,
    response: Response,
    run_id: ReviewIdentity,
    submission: ReviewSubmission,
) -> SavedAnswerReview:
    """Append an opinion; accepted does not verify facts or identify an authenticated reviewer."""
    container: AppContainer = request.app.state.container
    response.headers.update(_HEADERS)
    try:
        saved, created = container.answer_reviews.create(run_id, submission)
    except AnswerReviewError as exc:
        raise _http_error(exc) from exc
    response.status_code = 201 if created else 200
    return saved


@router.get(
    "/runs/{run_id}/reviews",
    response_model=ReviewHistoryPage,
    responses=_ERROR_RESPONSES,
)
def list_reviews(
    request: Request,
    response: Response,
    run_id: ReviewIdentity,
    limit: Annotated[int, Query(ge=1, le=MAX_REVIEW_LIMIT)] = DEFAULT_REVIEW_LIMIT,
    cursor: Annotated[int | None, Query(ge=1, le=MAX_REVIEW_SEQUENCE)] = None,
) -> ReviewHistoryPage:
    """Read newest-first history; use next_cursor unchanged for older records."""
    container: AppContainer = request.app.state.container
    response.headers.update(_HEADERS)
    try:
        return container.answer_reviews.list_reviews(run_id, limit=limit, cursor=cursor)
    except AnswerReviewError as exc:
        raise _http_error(exc) from exc
