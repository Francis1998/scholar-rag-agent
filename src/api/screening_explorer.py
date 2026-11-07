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
    _WRITE_HEADERS,
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
    _truncation,
    _UnrepresentableTextError,
    _url,
)
from api.native_forms import (
    FormTooLargeError,
    parse_fields,
    read_body,
    require_same_origin,
    require_urlencoded,
    unsigned_decimal,
)
from storage.document_catalog import MAX_TITLE_CHARACTERS
from storage.paper_collections import MAX_REVISION, CollectionError, CollectionId, Revision
from storage.paper_screening import (
    MAX_REASON_CHARACTERS,
    Decision,
    Reason,
    ScreeningHistory,
    ScreeningSubmission,
)
from storage.screening_priority import (
    MAX_FEATURE_TEXT_CHARACTERS,
    PriorityItem,
    ScreeningPriority,
    prioritize_screening,
)
from storage.source_context import ContextDocumentIdentity

PATH = "/explore/screening"
PRIORITY_PATH = "/explore/screening-priority"
MAX_FORM_BYTES = 65_536
MAX_QUERY_BYTES = 65_536
HISTORY_LIMIT = 5


_Revision = Annotated[Revision, BeforeValidator(unsigned_decimal)]
_ExpectedRevision = Annotated[
    int, Field(strict=True, ge=0, le=MAX_REVISION), BeforeValidator(unsigned_decimal)
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


class _PriorityOpen(BaseModel):
    """One exact collection revision; return state must name that same revision."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    collection_id: CollectionId
    collection_revision: _Revision
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


class _Form(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    selection: str = Field(max_length=16_384)
    expected_decision_revision: _ExpectedRevision
    decision: Decision
    reason: Reason


def _same_origin(request: Request) -> None:
    try:
        require_same_origin(request)
    except ValueError as exc:
        raise CollectionError("screening_form_origin", "", 403) from exc


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


_PRIORITY_FAILURES = {
    **_FAILURES,
    "invalid_screening_priority_request": (
        "Use an exact collection_id, a positive integer collection_revision and optional "
        "return state for that same revision. Unknown/duplicate fields, malformed encoding, "
        "invalid UTF-8 and GET bodies are rejected. Nothing was recorded."
    ),
    "screening_priority_request_too_large": ("The URL exceeds 65536 bytes. Nothing was recorded."),
    "screening_priority_method": (
        "The prioritized queue is read with GET. Record a decision only through the explicit "
        "human screening form. Nothing was recorded."
    ),
    "screening_priority_too_large": (
        "The complete ordering exceeds 524288 UTF-8 bytes. No partial ordering is shown."
    ),
    "invalid_screening_priority": (
        "The screening snapshot could not be ordered consistently. Nothing was saved or repaired."
    ),
    "invalid_document_record": (
        "A current member's stored title or text is invalid. Nothing was repaired; use the "
        "JSON/Python readers to inspect it."
    ),
}


def _priority_failure(status: int, code: str, root_path: str) -> HTMLResponse:
    return _error(
        status,
        code,
        "Prioritized screening queue unavailable",
        _PRIORITY_FAILURES.get(code, "The ordering could not be read. No success is reported."),
        navigation=_link(
            "back-to-collections", root_path + "/explore/collections", "Reopen a saved collection"
        ),
        prioritization=True,
    )


class _ScreeningExplorerRoute(APIRoute):
    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None:
        if self.methods is not None and scope["method"] not in self.methods:
            root_path = scope.get("root_path", "").rstrip("/")
            if self.path == PRIORITY_PATH:
                response = _priority_failure(405, "screening_priority_method", root_path)
                response.headers["Allow"] = "GET"
            else:
                response = _failure(405, "screening_form_method", root_path)
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
    yield _link(
        "screening-priority",
        _url(
            root_path + PRIORITY_PATH,
            {
                "collection_id": selection.collection_id,
                "collection_revision": selection.collection_revision,
                "collection_return": selection.navigation(),
            },
        ),
        "Most likely relevant first: reopen the prioritized queue",
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
            headers=_WRITE_HEADERS,
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


def _priority_item(
    item: PriorityItem, prefix: str, options: _PriorityOpen, root_path: str
) -> Iterator[str]:
    yield (
        f'<article class="priority-item"><h3>Rank <span class="priority-rank">{item.rank}</span>'
        "</h3>"
    )
    yield _identity(item.document_id, "document-id")
    yield _literal(item.title, "member-title")
    yield _truncation("Title", item.title_truncated, MAX_TITLE_CHARACTERS)
    yield f'<p>Human screening status: <strong class="member-status">{item.status}</strong>.</p>'
    if item.score is None or item.log_odds is None:
        yield '<p class="priority-unscored">No score: the ordering model is not trained.</p>'
    else:
        yield (
            "<p>Relevance score (uncalibrated Naive Bayes P(relevant)): "
            f'<strong class="priority-score">{item.score:.3f}</strong>; log-odds '
            f'<span class="priority-log-odds">{item.log_odds:+.3f}</span>.</p>'
        )
        if item.top_terms:
            yield '<p>Top terms pushing toward relevant:</p><ul class="top-terms">'
            for term in item.top_terms:
                yield (
                    f'<li><code class="top-term" translate="no">{_escaped(term.term)}</code> '
                    f'<span class="term-contribution">+{term.contribution:.3f}</span></li>'
                )
            yield "</ul>"
        else:
            yield "<p>No term in this paper pushes it toward relevant.</p>"
    navigation = options.navigation()
    yield _link(
        f"{prefix}screen-member-{item.rank}",
        _url(
            root_path + PATH,
            {
                "collection_id": options.collection_id,
                "collection_revision": options.collection_revision,
                "document_id": item.document_id,
                "collection_return": navigation,
            },
        ),
        "Open explicit human screening form",
    )
    yield _link(
        f"{prefix}read-member-{item.rank}",
        _url(
            root_path + "/explore/document",
            {"document_id": item.document_id, "collection_return": navigation},
        ),
        "Read this exact paper / choose source context",
    )
    yield "</article>"


def _priority_fragments(
    priority: ScreeningPriority, options: _PriorityOpen, root_path: str
) -> Iterator[str]:
    yield '<nav aria-label="Screening navigation">'
    yield _collection_link(options.navigation(), root_path)
    yield _link(
        "priority-json",
        _url(
            f"{root_path}/collections/{priority.collection_id}/screening-priority",
            {"collection_revision": priority.collection_revision},
        ),
        "Read this ordering as JSON (a separate read)",
    )
    yield "</nav>"
    yield _literal(priority.collection_name, "collection-name")
    yield _identity(priority.collection_id, "collection-id")
    yield (
        f"<p>Collection revision: {priority.collection_revision}; current members: "
        f"{priority.total_documents}.</p>"
        '<p class="notice" id="priority-guarantee">Ordering only. Nothing on this page records, '
        "changes or suggests a decision. Open each paper, read it and decide yourself: a high "
        "score is not an include and a low score is not an exclude. The collection page keeps "
        "its saved membership order.</p>"
    )
    counts = priority.counts
    if priority.model_status == "ready":
        yield (
            '<p id="model-status">Model status: <strong class="model-status">ready</strong>. '
            f"Trained on {priority.training_relevant} current include(s) and "
            f"{priority.training_irrelevant} current exclude(s) using "
            f"{priority.feature_terms} feature terms. Papers needing a decision are ordered "
            "most likely relevant first.</p>"
        )
    elif priority.model_status == "needs_include_and_exclude":
        yield (
            '<p id="model-status">Model status: <strong class="model-status">'
            "needs_include_and_exclude</strong>. Record at least one current include and one "
            f"current exclude (now {counts.include} and {counts.exclude}). Until then papers "
            "stay in saved collection order and no score is invented.</p>"
        )
    else:
        yield (
            '<p id="model-status">Model status: <strong class="model-status">'
            "no_training_terms</strong>. Your labeled papers contain no usable terms, so papers "
            "stay in saved collection order and no score is invented.</p>"
        )
    yield "<h2>Current human screening counts (all members)</h2><dl>"
    for status, count in counts.model_dump().items():
        yield f'<dt>{status}</dt><dd id="count-{status}">{count}</dd>'
    yield "</dl>"
    order = (
        "most likely relevant first"
        if priority.ordering == "most_likely_relevant_first"
        else "saved collection order"
    )
    yield f"<h2>Papers needing your decision: {len(priority.items)}, {order}</h2>"
    yield (
        "<p>Unscreened and stale members. A stale member's earlier opinion belongs to an older "
        "collection revision and is not a training label.</p>"
    )
    if not priority.items:
        yield "<p>No unscreened or stale members remain. Nothing was hidden.</p>"
    for item in priority.items:
        yield from _priority_item(item, "", options, root_path)
    yield f"<h2>Unsure papers: {len(priority.unsure_items)}, ranked separately</h2>"
    yield "<p>Your unsure decisions stay unsure; they are neither labels nor hidden.</p>"
    if not priority.unsure_items:
        yield "<p>No current unsure decisions.</p>"
    for item in priority.unsure_items:
        yield from _priority_item(item, "unsure-", options, root_path)
    method = priority.method
    yield (
        "<h2>How this order is computed</h2><details><summary>Method and limits</summary>"
        f"<p>Features: {_escaped(method.features)}</p>"
        f"<p>Weighting: {_escaped(method.weighting)}</p>"
        f"<p>Classifier: {_escaped(method.classifier)} Laplace alpha = {method.alpha}.</p>"
        f"<p>Class prior: {_escaped(method.class_prior)}</p>"
        f"<p>Score: {_escaped(method.score)}</p>"
        f"<p>Ties: {_escaped(method.tie_break)}</p>"
        f"<p>Only the title and the first {MAX_FEATURE_TEXT_CHARACTERS} characters of stored "
        "text are read; nothing is sent anywhere.</p>"
        '<p id="stopping-signal">No stopping signal is shown. Decisions keep per-paper '
        "revisions and wall-clock times, not a reliable global decision order, so a count of "
        "consecutive excludes since the last include would not be honest. When to stop "
        "screening remains your judgment; prioritization gives no recall guarantee.</p>"
        "</details>"
    )


def _priority_page(container: AppContainer, options: _PriorityOpen, root_path: str) -> HTMLResponse:
    priority = prioritize_screening(
        container.paper_screening,
        options.collection_id,
        collection_revision=options.collection_revision,
    )
    return _page(
        "Most likely relevant first",
        _priority_fragments(priority, options, root_path),
        prioritization=True,
    )


@router.get(
    PRIORITY_PATH,
    response_class=HTMLResponse,
    responses={
        404: {"description": "Exact collection not found"},
        405: {"description": "Only GET is supported; nothing is recorded"},
        409: {"description": "Stale revision, missing member or invalid saved data/HTML text"},
        413: {"description": "URL, complete ordering or escaped HTML exceeds its bound"},
        422: {"description": "Invalid exact collection, revision, return state or fields"},
        503: {"description": "Existing collection or screening storage unavailable"},
    },
)
async def explore_screening_priority(request: Request) -> Response:
    """Opt-in most-likely-relevant-first queue; read-only, never a decision or suggestion."""
    root_path = request.scope.get("root_path", "").rstrip("/")
    try:
        raw_query = request.scope.get("query_string", b"")
        if len(raw_query) > MAX_QUERY_BYTES:
            raise FormTooLargeError()
        if await read_body(request, max_bytes=MAX_FORM_BYTES):
            raise ValueError("GET cannot accept a body.")
        options = _PriorityOpen.model_validate(parse_fields(raw_query, max_fields=3))
        container: AppContainer = request.app.state.container
        return await run_in_threadpool(_priority_page, container, options, root_path)
    except CollectionError as exc:
        return _priority_failure(exc.status_code, exc.code, root_path)
    except sqlite3.Error:
        return _priority_failure(503, "screening_storage_unavailable", root_path)
    except FormTooLargeError:
        return _priority_failure(413, "screening_priority_request_too_large", root_path)
    except _PageTooLargeError:
        return _priority_failure(413, "html_page_too_large", root_path)
    except _UnrepresentableTextError:
        return _priority_failure(409, "html_text_not_representable", root_path)
    except (ValueError, RecursionError):
        return _priority_failure(422, "invalid_screening_priority_request", root_path)
