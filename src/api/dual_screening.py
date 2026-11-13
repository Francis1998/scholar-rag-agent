"""JSON routes for independent, optionally blind, multi-reviewer screening.

Every route sits outside the legacy ``/screening/{document_id:path}`` PUT wildcard, so no
team or reviewer URL can ever record an unlabeled decision for a paper with a lookalike ID.
"""

import logging
from collections.abc import Callable, Coroutine
from typing import Annotated, Literal
from urllib.parse import unquote_to_bytes

from fastapi import APIRouter, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from starlette.types import Receive, Scope, Send

from api.collections import COLLECTION_HEADERS, collection_errors
from api.dependencies import AppContainer
from storage.document_catalog import Identity
from storage.dual_screening import (
    AgreementFormat,
    AgreementReport,
    FinalDecisionExport,
    NamedReviewer,
    ProtocolChange,
    ResolutionSubmission,
    ReviewerHistory,
    ReviewerPriority,
    ReviewerQueue,
    ReviewerReview,
    ReviewerSubmission,
    ScreeningConflicts,
    ScreeningProtocol,
    ScreeningResolution,
    TeamExportFormat,
)
from storage.paper_collections import MAX_PAGE_SIZE, MAX_REVISION, CollectionId
from storage.paper_screening import ScreeningDownload, ScreeningStatus
from storage.source_context import ContextDocumentIdentity

logger = logging.getLogger(__name__)
MAX_QUERY_BYTES = 16_384
_PREFIX = "/collections/{collection_id}"
_QUERY_FIELDS = {
    f"{_PREFIX}/screening-protocol": frozenset(),
    f"{_PREFIX}/reviewer-screening": frozenset(
        {"reviewer", "collection_revision", "blind", "limit", "cursor", "status"}
    ),
    f"{_PREFIX}/reviewer-screening-history": frozenset(
        {"reviewer", "collection_revision", "document_id", "limit", "cursor"}
    ),
    f"{_PREFIX}/reviewer-screening-priority": frozenset({"reviewer", "collection_revision"}),
    f"{_PREFIX}/screening-conflicts": frozenset({"collection_revision", "limit", "cursor"}),
    f"{_PREFIX}/screening-agreement": frozenset({"collection_revision", "format"}),
    f"{_PREFIX}/screening-final-decisions": frozenset({"collection_revision", "format"}),
}
_ALLOWED = {f"{_PREFIX}/screening-protocol": "GET, PUT"}
# Document-ID cursors stay strings; only revision cursors are integers.
_INTEGERS = {f"{_PREFIX}/reviewer-screening-history": ("collection_revision", "limit", "cursor")}
_RESPONSES: dict[int | str, dict[str, object]] = {
    404: {"description": "Exact collection missing"},
    405: {"description": "Unsupported method; nothing was recorded"},
    409: {
        "description": "Blind mode on, revision/basis conflict, no disagreement, reviewer limit "
        "or invalid saved data"
    },
    413: {"description": "Complete response exceeds its UTF-8 byte limit; nothing partial"},
    422: {"description": "Invalid collection, reviewer label, member, revision or fields"},
    503: {"description": "Existing collection or screening storage unavailable"},
}


def _invalid_request() -> JSONResponse:
    logger.warning("Independent screening failed: invalid_reviewer_screening_request")
    return JSONResponse(
        {
            "detail": {
                "code": "invalid_reviewer_screening_request",
                "message": "Use an exact collection ID and member, positive ASCII integer "
                "revisions, and a pseudonymous reviewer label of 1-32 lowercase letters, "
                "digits, '.', '_' or '-' ('default' is reserved for existing unlabeled "
                "decisions). Unknown, duplicate, invalid Unicode/NUL and oversized fields fail; "
                "no input is echoed and nothing was recorded.",
            }
        },
        status_code=422,
        headers=COLLECTION_HEADERS,
    )


class _DualScreeningRoute(APIRoute):
    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None:
        if self.methods is not None and scope["method"] not in self.methods:
            logger.warning("Independent screening failed: reviewer_screening_method_not_allowed")
            allowed = _ALLOWED.get(self.path, ", ".join(sorted(self.methods)))
            await JSONResponse(
                {
                    "detail": {
                        "code": "reviewer_screening_method_not_allowed",
                        "message": f"Use {allowed}; nothing was recorded.",
                    }
                },
                status_code=405,
                headers={**COLLECTION_HEADERS, "Allow": allowed},
            )(scope, receive, send)
            return
        await super().handle(scope, receive, send)

    def get_route_handler(self) -> Callable[[Request], Coroutine[None, None, Response]]:
        handler = super().get_route_handler()
        fields = _QUERY_FIELDS.get(self.path)

        async def checked(request: Request) -> Response:
            try:
                raw = request.scope.get("query_string", b"")
                if len(raw) > MAX_QUERY_BYTES:
                    return _invalid_request()
                for value in (raw, request.scope.get("raw_path", b"")):
                    if "\0" in unquote_to_bytes(value).decode("utf-8"):
                        return _invalid_request()
                if request.method == "GET" and fields is not None:
                    if any(
                        key not in fields or len(request.query_params.getlist(key)) != 1
                        for key in request.query_params
                    ):
                        return _invalid_request()
                    for key in _INTEGERS.get(self.path, ("collection_revision", "limit")):
                        value = request.query_params.get(key)
                        if value is not None and (
                            not value.isascii() or not value.isdecimal() or len(value) > 19
                        ):
                            return _invalid_request()
                return await handler(request)
            except (RequestValidationError, UnicodeError):
                return _invalid_request()

        return checked


router = APIRouter(route_class=_DualScreeningRoute, tags=["Independent dual-reviewer screening"])
_Reviewer = Annotated[NamedReviewer, Query()]
_CollectionRevision = Annotated[int, Query(ge=1, le=MAX_REVISION)]
_Blind = Annotated[Literal["true", "false"] | None, Query()]


def _container(request: Request) -> AppContainer:
    container: AppContainer = request.app.state.container
    return container


def _attachment(download: ScreeningDownload) -> Response:
    return Response(
        content=download.content,
        media_type=download.media_type,
        headers={
            **COLLECTION_HEADERS,
            "Content-Disposition": f'attachment; filename="{download.filename}"',
        },
    )


@router.get(f"{_PREFIX}/screening-protocol", response_model=ScreeningProtocol, responses=_RESPONSES)
def read_protocol(
    request: Request, response: Response, collection_id: CollectionId
) -> ScreeningProtocol:
    """Blind mode and the exclusion-reason rule; revision 0 is the unsaved blind default."""
    with collection_errors():
        result = _container(request).dual_screening.protocol(collection_id)
    response.headers.update(COLLECTION_HEADERS)
    return result


@router.put(f"{_PREFIX}/screening-protocol", response_model=ScreeningProtocol, responses=_RESPONSES)
def change_protocol(
    request: Request, response: Response, collection_id: CollectionId, payload: ProtocolChange
) -> ScreeningProtocol:
    """Append a protocol revision, for example to unblind after independent screening."""
    with collection_errors():
        result = _container(request).dual_screening.set_protocol(collection_id, payload)
    response.headers.update(COLLECTION_HEADERS)
    return result


@router.get(f"{_PREFIX}/reviewer-screening", response_model=ReviewerQueue, responses=_RESPONSES)
def reviewer_queue(
    request: Request,
    response: Response,
    collection_id: CollectionId,
    reviewer: _Reviewer,
    collection_revision: _CollectionRevision,
    blind: _Blind = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    cursor: Annotated[Identity | None, Query()] = None,
    status: Annotated[ScreeningStatus | None, Query()] = None,
) -> ReviewerQueue:
    """Your own statuses; in blind mode other reviewers appear only after you decide."""
    with collection_errors():
        result = _container(request).dual_screening.reviewer_queue(
            collection_id,
            reviewer=reviewer,
            collection_revision=collection_revision,
            blind=None if blind is None else blind == "true",
            limit=limit,
            cursor=cursor,
            status=status,
        )
    response.headers.update(COLLECTION_HEADERS)
    return result


@router.put(
    f"{_PREFIX}/reviewer-screening/{{document_id:path}}",
    response_model=ReviewerReview,
    responses=_RESPONSES,
)
def submit_reviewer_decision(
    request: Request,
    response: Response,
    collection_id: CollectionId,
    document_id: Identity,
    payload: ReviewerSubmission,
) -> ReviewerReview:
    """Record one named reviewer's decision; returns only that reviewer's own record."""
    with collection_errors():
        result = _container(request).dual_screening.submit(collection_id, document_id, payload)
    response.headers.update(COLLECTION_HEADERS)
    return result


@router.get(
    f"{_PREFIX}/reviewer-screening-history",
    response_model=ReviewerHistory,
    responses=_RESPONSES,
)
def reviewer_history(
    request: Request,
    response: Response,
    collection_id: CollectionId,
    reviewer: _Reviewer,
    document_id: Annotated[ContextDocumentIdentity, Query()],
    collection_revision: _CollectionRevision,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = 20,
    cursor: Annotated[int | None, Query(ge=1, le=MAX_REVISION)] = None,
) -> ReviewerHistory:
    """Your own append-only decisions for one member; other reviewers are never included."""
    with collection_errors():
        result = _container(request).dual_screening.reviewer_history(
            collection_id,
            document_id,
            reviewer=reviewer,
            collection_revision=collection_revision,
            limit=limit,
            cursor=cursor,
        )
    response.headers.update(COLLECTION_HEADERS)
    return result


@router.get(
    f"{_PREFIX}/reviewer-screening-priority",
    response_model=ReviewerPriority,
    responses=_RESPONSES,
)
def reviewer_priority(
    request: Request,
    response: Response,
    collection_id: CollectionId,
    reviewer: _Reviewer,
    collection_revision: _CollectionRevision,
) -> ReviewerPriority:
    """Most likely relevant first, trained only on your own current decisions."""
    with collection_errors():
        result = _container(request).dual_screening.reviewer_priority(
            collection_id, reviewer=reviewer, collection_revision=collection_revision
        )
    response.headers.update(COLLECTION_HEADERS)
    return result


@router.get(
    f"{_PREFIX}/screening-conflicts", response_model=ScreeningConflicts, responses=_RESPONSES
)
def screening_conflicts(
    request: Request,
    response: Response,
    collection_id: CollectionId,
    collection_revision: _CollectionRevision,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    cursor: Annotated[Identity | None, Query()] = None,
) -> ScreeningConflicts:
    """Current disagreements and their resolutions; refused while blind mode is on."""
    with collection_errors():
        result = _container(request).dual_screening.conflicts(
            collection_id, collection_revision=collection_revision, limit=limit, cursor=cursor
        )
    response.headers.update(COLLECTION_HEADERS)
    return result


@router.put(
    f"{_PREFIX}/screening-resolutions/{{document_id:path}}",
    response_model=ScreeningResolution,
    responses=_RESPONSES,
)
def resolve_conflict(
    request: Request,
    response: Response,
    collection_id: CollectionId,
    document_id: Identity,
    payload: ResolutionSubmission,
) -> ScreeningResolution:
    """Append a final decision for one disagreement; reviewer decisions stay unchanged."""
    with collection_errors():
        result = _container(request).dual_screening.resolve(collection_id, document_id, payload)
    response.headers.update(COLLECTION_HEADERS)
    return result


@router.get(
    f"{_PREFIX}/screening-agreement",
    response_model=AgreementReport,
    responses={
        **_RESPONSES,
        200: {
            "content": {
                "text/markdown": {"schema": {"type": "string"}},
                "text/csv": {"schema": {"type": "string"}},
            }
        },
    },
)
def screening_agreement(
    request: Request,
    collection_id: CollectionId,
    collection_revision: _CollectionRevision,
    format: AgreementFormat = "json",
) -> Response:
    """Cohen's kappa with a 95% CI, PABAK and conflict counts as JSON, Markdown or CSV."""
    with collection_errors():
        download = _container(request).dual_screening.export_agreement(
            collection_id, collection_revision=collection_revision, format=format
        )
    return _attachment(download)


@router.get(
    f"{_PREFIX}/screening-final-decisions",
    response_model=FinalDecisionExport,
    responses={
        **_RESPONSES,
        200: {"content": {"text/csv": {"schema": {"type": "string"}}}},
    },
)
def screening_final_decisions(
    request: Request,
    collection_id: CollectionId,
    collection_revision: _CollectionRevision,
    format: TeamExportFormat = "json",
) -> Response:
    """Every member's team outcome, resolutions first, beside every reviewer's decision."""
    with collection_errors():
        download = _container(request).dual_screening.export_final_decisions(
            collection_id, collection_revision=collection_revision, format=format
        )
    return _attachment(download)
