"""Explicit same-origin native writes over the existing human screening service."""

import json
import sqlite3
from collections.abc import Iterator
from typing import Annotated, Self

from fastapi import APIRouter, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator
from starlette.concurrency import run_in_threadpool
from starlette.types import Receive, Scope, Send

from api.collection_explorer import _ERRORS, _history_review
from api.dependencies import AppContainer
from api.explorer import (
    _HEADERS,
    _collection_link,
    _CollectionReturn,
    _CollectionReturnToken,
    _decode_collection_return,
    _error,
    _escaped,
    _identity,
    _link,
    _literal,
    _page,
    _PageTooLargeError,
    _return_pairs,
    _UnrepresentableTextError,
    _url,
)
from api.native_forms import FormTooLargeError, parse_fields, read_body, require_urlencoded
from storage.paper_collections import MAX_REVISION, CollectionError, CollectionId, Revision
from storage.paper_screening import (
    MAX_REASON_CHARACTERS,
    Decision,
    Reason,
    ScreeningHistory,
    ScreeningSubmission,
)
from storage.source_context import ContextDocumentIdentity

PATH = "/explore/screening"
MAX_FORM_BYTES = 65_536
MAX_QUERY_BYTES = 65_536
HISTORY_LIMIT = 5


def _number(value: object) -> object:
    if isinstance(value, str) and value.isascii() and value.isdecimal() and len(value) <= 19:
        return int(value)
    return value


_Revision = Annotated[Revision, BeforeValidator(_number)]
_ExpectedRevision = Annotated[
    int, Field(strict=True, ge=0, le=MAX_REVISION), BeforeValidator(_number)
]


class _Selection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    collection_id: CollectionId
    collection_revision: _Revision
    document_id: ContextDocumentIdentity
    collection_return: _CollectionReturnToken | None = None

    @model_validator(mode="after")
    def matching_return(self) -> Self:
        if self.collection_return is not None:
            navigation = _decode_collection_return(self.collection_return)
            if (
                navigation.collection_id != self.collection_id
                or navigation.collection_revision != self.collection_revision
            ):
                raise ValueError("Return state must match this exact collection revision.")
        return self

    def navigation(self) -> str:
        return (
            self.collection_return
            or _CollectionReturn(
                collection_id=self.collection_id, collection_revision=self.collection_revision
            ).token()
        )

    def parameters(self) -> dict[str, str | int | None]:
        return self.model_dump()


class _Open(_Selection):
    saved_revision: _Revision | None = None


class _Form(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    selection: str = Field(max_length=16_384)
    expected_decision_revision: _ExpectedRevision
    decision: Decision
    reason: Reason


def _same_origin(request: Request) -> None:
    origins = request.headers.getlist("origin")
    sites = request.headers.getlist("sec-fetch-site")
    if (
        len(request.headers.getlist("host")) != 1
        or origins != [f"{request.url.scheme}://{request.url.netloc}"]
        or (sites and sites != ["same-origin"])
    ):
        raise CollectionError("screening_form_origin", "", 403)


_FAILURES = {
    **_ERRORS,
    "invalid_screening_form": (
        "Use the exact opened selection, both integer revisions, include/exclude/unsure, "
        "and a nonblank reason of 1-1000 characters without NUL. Unknown/duplicate fields, "
        "malformed encoding, invalid UTF-8, GET bodies and POST query fields are rejected. "
        "No input is echoed and no decision was written."
    ),
    "screening_form_too_large": "The URL or form body exceeds 65536 bytes. Nothing was written.",
    "screening_form_encoding": "Use a UTF-8 URL-encoded form without content encoding.",
    "screening_revision_conflict": (
        "Another decision was recorded after this form opened. Nothing was overwritten. "
        "Reopen the member from its collection, inspect the latest decision, then decide again."
    ),
    "screening_revision_exhausted": "The decision revision cannot increase. Nothing was written.",
    "screening_form_method": (
        "Use GET to inspect or an explicit same-origin POST to record a decision."
    ),
    "screening_form_origin": (
        "The submission did not come from this exact origin (scheme, host and port), or "
        "its Origin/Fetch Metadata headers are missing, ambiguous or cross-origin. "
        "Open the local form again. No decision was written."
    ),
    "html_page_too_large": "The complete escaped page exceeds 1 MiB. No partial form is shown.",
    "html_text_not_representable": (
        "Saved text cannot be displayed faithfully as UTF-8 HTML. Use the JSON/Python readers."
    ),
}


def _failure(status: int, code: str, root_path: str) -> HTMLResponse:
    return _error(
        status,
        code,
        "Human screening unavailable",
        _FAILURES.get(code, "The screening service failed. No success is being reported."),
        navigation=_link(
            "back-to-collections", root_path + "/explore/collections", "Reopen a saved collection"
        ),
        screening=True,
    )


class _ScreeningExplorerRoute(APIRoute):
    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None:
        if self.methods is not None and scope["method"] not in self.methods:
            response = _failure(
                405, "screening_form_method", scope.get("root_path", "").rstrip("/")
            )
            response.headers["Allow"] = "GET, POST"
            await response(scope, receive, send)
            return
        await super().handle(scope, receive, send)


def _fragments(
    history: ScreeningHistory, selection: _Selection, saved_revision: int | None, root_path: str
) -> Iterator[str]:
    yield '<nav aria-label="Screening navigation">'
    yield _collection_link(selection.navigation(), root_path)
    yield _link(
        "screening-paper",
        _url(
            root_path + "/explore/document",
            {"document_id": selection.document_id, "collection_return": selection.navigation()},
        ),
        "Read this exact paper / choose source context",
    )
    yield _link(
        "screening-history",
        _url(root_path + "/explore/screening-history", selection.parameters()),
        "Inspect all retained decision history",
    )
    yield "</nav>"
    yield _literal(history.collection_name, "collection-name")
    yield _identity(history.collection_id, "collection-id")
    yield _identity(history.document_id, "document-id")
    yield (
        f"<p>Collection revision: {history.collection_revision}; current human status: "
        f'<strong class="current-status">{history.current_status}</strong>.</p>'
    )
    if saved_revision is not None:
        if any(review.revision == saved_revision for review in history.items):
            yield (
                f'<p class="notice" role="status" id="screening-saved">'
                f"Stored decision revision {saved_revision} confirmed.</p>"
            )
        else:
            yield (
                '<p class="notice">The requested confirmation is not in this recent-history '
                "window. No submission success is inferred from the URL. Inspect retained "
                "history before retrying.</p>"
            )
    yield "<h2>Latest persisted decision</h2>"
    if history.latest_review is None:
        yield "<p>No human decision is recorded for this member.</p>"
    else:
        yield from _history_review(history.latest_review, "latest-review")
    if history.current_status == "stale":
        yield (
            '<p class="notice">This opinion belongs to an earlier collection revision, '
            "not a current include. Reassessment requires your explicit new decision.</p>"
        )
    yield (
        "<h2>Record your decision</h2>"
        '<p id="screening-help">Read the paper and source context first. Choose a decision and '
        "write your own reason each time; no decision or reason is preselected. This appends "
        "one human opinion, not an authenticated approval or scientific validation. "
        "Membership and retrieval selection do not change.</p>"
        '<form id="screening-form" method="post" enctype="application/x-www-form-urlencoded" '
        f'action="{_escaped(root_path + PATH)}" accept-charset="UTF-8" autocomplete="off" '
        'aria-describedby="screening-help">'
    )
    token = json.dumps(selection.model_dump(), ensure_ascii=True, separators=(",", ":"))
    expected = history.latest_review.revision if history.latest_review else 0
    yield (
        f'<input type="hidden" name="selection" value="{_escaped(token)}">'
        f'<input type="hidden" name="expected_decision_revision" value="{expected}">'
        '<div><label for="decision">Human decision</label>'
        '<select id="decision" name="decision" required>'
        '<option value="" selected>Choose explicitly</option>'
        '<option value="include">Include</option><option value="exclude">Exclude</option>'
        '<option value="unsure">Unsure</option></select></div>'
        '<div class="worksheet-input"><label for="reason">Reason for this decision</label>'
        f'<textarea id="reason" name="reason" required maxlength="{MAX_REASON_CHARACTERS}" '
        'aria-describedby="reason-help" spellcheck="false"></textarea>'
        '<p id="reason-help">1-1000 characters, nonblank; no credentials or personal data. '
        "Text is saved as submitted, without trimming. Browsers serialize line breaks as CRLF; "
        "the server's character bound is authoritative.</p></div>"
        '<div class="actions"><button type="submit">Record human decision</button></div></form>'
        f"<h2>Recent retained history (up to {HISTORY_LIMIT})</h2>"
    )
    yield (
        f'<p>Decision revisions not retained in history: <span id="unretained-revisions">'
        f"{history.unretained_decision_revisions}</span>. No old decisions were backfilled. "
        "Original reasons and timestamps remain; newest decision revision first. "
        "A historical include is not a current inclusion.</p>"
    )
    for review in history.items:
        yield f"<article><h3>Decision revision {review.revision}</h3>"
        yield from _history_review(review, "history")
        yield "</article>"
    if not history.items:
        yield (
            "<p>No retained history yet; a legacy latest-only opinion is not a recovered event.</p>"
        )


def _open(container: AppContainer, options: _Open, root_path: str) -> HTMLResponse:
    selection = _Selection.model_validate(options.model_dump(exclude={"saved_revision"}))
    history = container.paper_screening.list_history(
        selection.collection_id,
        selection.document_id,
        collection_revision=selection.collection_revision,
        limit=HISTORY_LIMIT,
    )
    return _page(
        "Human paper-screening workbench",
        _fragments(history, selection, options.saved_revision, root_path),
        screening=True,
    )


router = APIRouter(route_class=_ScreeningExplorerRoute, tags=["Human screening browser"])


@router.get(PATH, response_class=HTMLResponse)
@router.post(
    PATH,
    response_class=HTMLResponse,
    responses={
        303: {"description": "One decision persisted; local revision-confirmation redirect"},
        403: {"description": "Missing, ambiguous or cross-origin browser origin"},
        404: {"description": "Exact collection not found"},
        405: {"description": "Only GET and explicit POST are supported"},
        409: {"description": "Stale revision, missing corpus member or invalid saved metadata"},
        413: {"description": "Bounded body, history or complete HTML exceeded"},
        415: {"description": "Unsupported form encoding"},
        422: {"description": "Invalid exact selection, decision, reason or ambiguous fields"},
        503: {"description": "Existing screening storage unavailable"},
    },
)
async def explore_screening(request: Request) -> Response:
    root_path = request.scope.get("root_path", "").rstrip("/")
    try:
        if request.method == "POST":
            _same_origin(request)
            try:
                require_urlencoded(request)
            except ValueError as exc:
                raise CollectionError("screening_form_encoding", "", 415) from exc
        raw_query = request.scope.get("query_string", b"")
        if len(raw_query) > MAX_QUERY_BYTES:
            raise FormTooLargeError()
        if request.method == "POST" and raw_query:
            raise ValueError("POST fields belong only in the body.")
        body = await read_body(request, max_bytes=MAX_FORM_BYTES)
        container: AppContainer = request.app.state.container
        if request.method == "GET":
            if body:
                raise ValueError("GET cannot accept a body.")
            options = _Open.model_validate(parse_fields(raw_query, max_fields=5))
            return await run_in_threadpool(_open, container, options, root_path)
        form = _Form.model_validate(parse_fields(body, max_fields=4))
        selection = _Selection.model_validate(
            json.loads(form.selection, object_pairs_hook=_return_pairs)
        )
        payload = ScreeningSubmission(
            collection_revision=selection.collection_revision,
            expected_decision_revision=form.expected_decision_revision,
            decision=form.decision,
            reason=form.reason,
        )
        # Construct the bounded local response before the only write; never echo a reason in URLs.
        redirect = RedirectResponse(
            _url(
                root_path + PATH,
                {
                    **selection.parameters(),
                    "saved_revision": payload.expected_decision_revision + 1,
                },
            ),
            status_code=303,
            headers=_HEADERS,
        )
        await run_in_threadpool(
            container.paper_screening.submit,
            selection.collection_id,
            selection.document_id,
            payload,
        )
        return redirect
    except CollectionError as exc:
        return _failure(exc.status_code, exc.code, root_path)
    except sqlite3.Error:
        return _failure(503, "screening_storage_unavailable", root_path)
    except FormTooLargeError:
        return _failure(413, "screening_form_too_large", root_path)
    except _PageTooLargeError:
        return _failure(413, "html_page_too_large", root_path)
    except _UnrepresentableTextError:
        return _failure(409, "html_text_not_representable", root_path)
    except (ValueError, RecursionError):
        return _failure(422, "invalid_screening_form", root_path)
