"""Complete portable notebooks, separate from the existing paginated opinion APIs."""

import logging
from collections.abc import Callable, Coroutine
from typing import Annotated
from urllib.parse import unquote_to_bytes

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from starlette.types import Receive, Scope, Send

from api.dependencies import AppContainer
from storage.evidence_export import EvidenceExportError
from storage.review_notebook import NotebookFormat, ReviewNotebook
from storage.run_history import Identity

logger = logging.getLogger(__name__)
MAX_QUERY_BYTES = 16384
_HEADERS = {
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}


def _invalid_request() -> HTTPException:
    logger.warning("Notebook export failed: invalid_notebook_export_request")
    return HTTPException(
        422,
        detail={
            "code": "invalid_notebook_export_request",
            "message": "Use an exact run ID (1-256 Unicode characters) and one format "
            "(json or markdown). Unknown, duplicate, malformed or oversized parameters fail.",
        },
        headers=_HEADERS,
    )


class _NotebookExportRoute(APIRoute):
    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None:
        if self.methods is not None and scope["method"] not in self.methods:
            logger.warning("Notebook export failed: notebook_export_method_not_allowed")
            response = JSONResponse(
                {
                    "detail": {
                        "code": "notebook_export_method_not_allowed",
                        "message": "Notebook downloads require GET; no operation was executed.",
                    }
                },
                status_code=405,
                headers={**_HEADERS, "Allow": ", ".join(sorted(self.methods))},
            )
            await response(scope, receive, send)
            return
        await super().handle(scope, receive, send)

    def get_route_handler(self) -> Callable[[Request], Coroutine[None, None, Response]]:
        handler = super().get_route_handler()

        async def read(request: Request) -> Response:
            raw = request.scope.get("query_string", b"")
            allowed = {"format", "run_id"} if self.path.startswith("/explore/") else {"format"}
            try:
                if len(raw) > MAX_QUERY_BYTES or any(
                    key not in allowed or len(request.query_params.getlist(key)) != 1
                    for key in request.query_params
                ):
                    raise RequestValidationError([])
                unquote_to_bytes(raw).decode("utf-8")
                unquote_to_bytes(request.scope.get("raw_path", b"")).decode("utf-8")
                return await handler(request)
            except (RequestValidationError, UnicodeError) as exc:
                raise _invalid_request() from exc
            except EvidenceExportError as exc:
                logger.warning("Notebook export failed: %s", exc.code)
                raise HTTPException(
                    exc.status_code,
                    detail={"code": exc.code, "message": str(exc)},
                    headers=_HEADERS,
                ) from exc

        return read


router = APIRouter(route_class=_NotebookExportRoute, tags=["Human review notebook"])
_RESPONSES: dict[int | str, dict[str, object]] = {
    200: {"content": {"text/markdown": {"schema": {"type": "string"}}}},
    404: {"description": "No saved events for this exact run"},
    409: {"description": "Incomplete, failed, legacy or corrupt saved records"},
    413: {
        "description": "Whole notebook exceeds fixed record or UTF-8 byte limits; no partial export"
    },
    422: {"description": "Invalid exact run ID or download parameters"},
    503: {"description": "Existing evidence, review or annotation storage unavailable"},
}


def _download(request: Request, run_id: str, format: NotebookFormat) -> Response:
    container: AppContainer = request.app.state.container
    result = container.review_notebook.export(run_id)
    return Response(
        result.to_json() if format == "json" else result.to_markdown(),
        media_type="application/json" if format == "json" else "text/markdown",
        headers={
            **_HEADERS,
            "Content-Disposition": f'attachment; filename="{result.filename(format)}"',
        },
    )


@router.get("/runs/{run_id}/notebook", response_model=ReviewNotebook, responses=_RESPONSES)
def export_notebook(
    request: Request, run_id: Identity, format: NotebookFormat = "json"
) -> Response:
    """Download all retained human opinions with frozen evidence in one read snapshot."""
    return _download(request, run_id, format)


@router.get("/explore/run/notebook/export", response_model=ReviewNotebook, responses=_RESPONSES)
def download_notebook(
    request: Request, run_id: Annotated[Identity, Query()], format: NotebookFormat = "json"
) -> Response:
    """Query-addressed equivalent preserves opaque IDs that cannot be path segments."""
    return _download(request, run_id, format)
