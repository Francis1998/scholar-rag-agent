"""Read-only JSON stopping-rule assessment for prioritized human screening; never a decision."""

import logging
import re
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
from retrieval.scope import MAX_DOCUMENT_IDS
from storage.paper_collections import MAX_REVISION, CollectionId
from storage.screening_stopping import (
    DEFAULT_CONFIDENCE,
    DEFAULT_MIN_SCREENED_PERCENT,
    DEFAULT_TARGET_RECALL,
    ScreeningStoppingAssessment,
    assess_screening_stopping,
)

logger = logging.getLogger(__name__)
MAX_QUERY_BYTES = 32_768
INTEGER_PARAMETERS = frozenset(
    {
        "collection_revision",
        "irrelevant_run",
        "irrelevant_run_percent",
        "min_screened_percent",
        "estimated_relevant",
        "sample_seed",
        "sample_after",
    }
)
DECIMAL_PARAMETERS = frozenset({"target_recall", "confidence"})
PARAMETERS = INTEGER_PARAMETERS | DECIMAL_PARAMETERS | {"sample_urn_sha256", "key_document_id"}
_DECIMAL = re.compile(r"[01](?:\.[0-9]{1,6})?")


def _invalid_request() -> JSONResponse:
    logger.warning("Screening stopping failed: invalid_screening_stopping_request")
    return JSONResponse(
        {
            "detail": {
                "code": "invalid_screening_stopping_request",
                "message": "Use an exact collection ID, one positive ASCII integer "
                "collection_revision and optional documented parameters: decimals with at "
                "most 6 places, ASCII integers, at most 100 unique key_document_id values. "
                "Unknown, duplicate, conflicting, invalid Unicode/NUL and oversized "
                "parameters fail.",
            }
        },
        status_code=422,
        headers=COLLECTION_HEADERS,
    )


def valid_query(request: Request) -> bool:
    """Exact parameter names, counts and ASCII number formats before any database read."""
    raw = request.scope.get("query_string", b"")
    if len(raw) > MAX_QUERY_BYTES:
        return False
    for value in (raw, request.scope.get("raw_path", b"")):
        if "\0" in unquote_to_bytes(value).decode("utf-8"):
            return False
    parameters = request.query_params
    for key in parameters:
        values = parameters.getlist(key)
        if key not in PARAMETERS or (key != "key_document_id" and len(values) != 1):
            return False
        if key in INTEGER_PARAMETERS and not all(
            value.isascii() and value.isdecimal() and len(value) <= 19 for value in values
        ):
            return False
        if key in DECIMAL_PARAMETERS and not _DECIMAL.fullmatch(values[0]):
            return False
    return len(parameters.getlist("key_document_id")) <= MAX_DOCUMENT_IDS


class _StoppingRoute(APIRoute):
    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None:
        if self.methods is not None and scope["method"] not in self.methods:
            logger.warning("Screening stopping failed: screening_stopping_method_not_allowed")
            await JSONResponse(
                {
                    "detail": {
                        "code": "screening_stopping_method_not_allowed",
                        "message": "The stopping assessment is read with GET; nothing was "
                        "recorded, excluded or stopped. Use the screening PUT route to record "
                        "a human decision.",
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
                if not valid_query(request):
                    return _invalid_request()
                return await handler(request)
            except (RequestValidationError, UnicodeError):
                return _invalid_request()

        return read


router = APIRouter(route_class=_StoppingRoute, tags=["Human screening stopping rules"])


@router.get(
    "/collections/{collection_id}/screening-stopping",
    response_model=ScreeningStoppingAssessment,
    responses={
        404: {"description": "Exact collection missing"},
        405: {"description": "Read with GET; nothing is recorded, excluded or stopped"},
        409: {
            "description": "Revision conflict, missing member, invalid saved data, tied sample "
            "boundary or changed sample urn"
        },
        413: {"description": "Retained history or complete assessment exceeds its bound"},
        422: {
            "description": "Invalid collection ID, revision, parameters, key paper or "
            "out-of-range value"
        },
        503: {"description": "Existing collection or screening storage unavailable"},
    },
)
def screening_stopping(
    request: Request,
    response: Response,
    collection_id: CollectionId,
    collection_revision: Annotated[int, Query(ge=1, le=MAX_REVISION)],
    target_recall: Annotated[float, Query()] = DEFAULT_TARGET_RECALL,
    confidence: Annotated[float, Query()] = DEFAULT_CONFIDENCE,
    irrelevant_run: Annotated[int | None, Query()] = None,
    irrelevant_run_percent: Annotated[int | None, Query()] = None,
    min_screened_percent: Annotated[int, Query()] = DEFAULT_MIN_SCREENED_PERCENT,
    estimated_relevant: Annotated[int | None, Query()] = None,
    key_document_id: Annotated[list[str] | None, Query()] = None,
    sample_seed: Annotated[int | None, Query()] = None,
    sample_after: Annotated[int | None, Query()] = None,
    sample_urn_sha256: Annotated[str | None, Query()] = None,
) -> ScreeningStoppingAssessment:
    """Assess advisory stopping rules over recorded decisions; never decide or stop.

    Outside the screening PUT wildcard, so this URL can never record a decision for a paper
    whose ID happens to be ``stopping``.
    """
    container: AppContainer = request.app.state.container
    with collection_errors():
        result = assess_screening_stopping(
            container.paper_screening,
            collection_id,
            collection_revision=collection_revision,
            target_recall=target_recall,
            confidence=confidence,
            irrelevant_run=irrelevant_run,
            irrelevant_run_percent=irrelevant_run_percent,
            min_screened_percent=min_screened_percent,
            estimated_relevant=estimated_relevant,
            key_document_ids=key_document_id or (),
            sample_seed=sample_seed,
            sample_after=sample_after,
            sample_urn_sha256=sample_urn_sha256,
        )
    response.headers.update(COLLECTION_HEADERS)
    return result
