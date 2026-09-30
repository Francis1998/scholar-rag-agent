"""Read-only BibTeX and provenance downloads for completed saved-run citations."""

import logging
from collections.abc import Callable, Coroutine

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute

from api.dependencies import AppContainer
from storage.evidence_export import EvidenceExportError
from storage.saved_bibliography import (
    BibliographyFormat,
    BibliographyIdentity,
    SavedBibliography,
)

logger = logging.getLogger(__name__)
_HEADERS = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}


def _invalid_request() -> HTTPException:
    return HTTPException(
        status_code=422,
        detail={
            "code": "invalid_bibliography_request",
            "message": "The run ID must contain 1 to 256 valid characters; "
            "supply at most one format parameter, either bibtex or json.",
        },
        headers=_HEADERS,
    )


class _BibliographyRoute(APIRoute):
    def get_route_handler(self) -> Callable[[Request], Coroutine[None, None, Response]]:
        handler = super().get_route_handler()

        async def validated(request: Request) -> Response:
            if len(request.query_params.getlist("format")) > 1 or any(
                name != "format" for name in request.query_params
            ):
                raise _invalid_request()
            try:
                return await handler(request)
            except RequestValidationError as exc:
                raise _invalid_request() from exc

        return validated


router = APIRouter(route_class=_BibliographyRoute)


@router.get(
    "/runs/{run_id}/bibliography",
    response_model=SavedBibliography,
    responses={
        200: {"content": {"application/x-bibtex": {"schema": {"type": "string"}}}},
        404: {"description": "Unknown saved run"},
        409: {"description": "Invalid, failed, incomplete, legacy, or over-limit saved evidence"},
        413: {"description": "Serialized bibliography exceeds the download byte limit"},
        422: {"description": "Invalid run ID or format parameters"},
        503: {"description": "Saved evidence storage unavailable"},
    },
)
def bibliography(
    request: Request, run_id: BibliographyIdentity, format: BibliographyFormat = "bibtex"
) -> Response:
    """Export only saved final-answer citations; never retrieve, generate, enrich, or write."""
    container: AppContainer = request.app.state.container
    try:
        result = container.saved_bibliography.export(run_id)
        content = result.to_json() if format == "json" else result.to_bibtex()
    except EvidenceExportError as exc:
        logger.warning("Saved bibliography failed: %s", exc.code)
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": str(exc)},
            headers=_HEADERS,
        ) from exc
    extension = "json" if format == "json" else "bib"
    return Response(
        content=content,
        media_type="application/json"
        if format == "json"
        else "application/x-bibtex; charset=utf-8",
        headers={
            **_HEADERS,
            "Content-Disposition": f'attachment; filename="bibliography.{extension}"',
        },
    )
