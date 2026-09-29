"""Read-only saved-evidence drift, never retrieval, generation, or corpus maintenance."""

import logging
from collections.abc import Callable, Coroutine
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute

from api.dependencies import AppContainer
from storage.corpus_drift import CorpusDriftReport, DriftIdentity
from storage.evidence_export import EvidenceExportError

logger = logging.getLogger(__name__)
_HEADERS = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}


class _DriftRoute(APIRoute):
    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        handler = super().get_route_handler()

        async def validate(request: Request) -> Response:
            try:
                return await handler(request)
            except RequestValidationError as exc:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "code": "invalid_drift_request",
                        "message": "The run ID must contain 1 to 256 valid characters.",
                    },
                    headers=_HEADERS,
                ) from exc

        return validate


router = APIRouter(route_class=_DriftRoute)


@router.get(
    "/runs/{run_id}/corpus-drift",
    response_model=CorpusDriftReport,
    responses={
        404: {"description": "Unknown saved run"},
        409: {"description": "Invalid/unavailable frozen evidence, corrupt chunk, or read limit"},
        422: {"description": "Invalid run identity"},
        503: {"description": "Evidence or corpus storage unavailable"},
    },
)
def corpus_drift(request: Request, response: Response, run_id: DriftIdentity) -> CorpusDriftReport:
    """Compare a completed run's chunks to current persisted records in frozen rank order."""
    container: AppContainer = request.app.state.container
    response.headers.update(_HEADERS)
    try:
        return container.corpus_drift.report(run_id)
    except EvidenceExportError as exc:
        logger.warning("Corpus drift failed: %s", exc.code)
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": str(exc)},
            headers=_HEADERS,
        ) from exc
