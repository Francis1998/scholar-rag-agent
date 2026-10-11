"""Native, bounded saved-run discovery and frozen-evidence reading without execution."""

import sqlite3
from collections.abc import Callable, Coroutine, Iterator
from typing import Annotated
from urllib.parse import quote, unquote_to_bytes

from fastapi import APIRouter, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, BeforeValidator, ConfigDict
from starlette.types import Receive, Scope, Send

from agent.evidence_html import HTML_CSP, BrowserNavigation, render_html
from agent.models import AgentState
from api.dependencies import AppContainer
from api.explorer import (
    _CatalogLimit,
    _error,
    _escaped,
    _identity,
    _link,
    _literal,
    _OptionalSource,
    _OptionalTitle,
    _page,
    _PageTooLargeError,
    _pagination,
    _query_integer,
    _truncation,
    _UnrepresentableTextError,
    _url,
)
from storage.evidence_export import EvidenceExportError
from storage.run_history import (
    DEFAULT_PAGE_SIZE,
    MAX_QUERY_CHARACTERS,
    EventID,
    Identity,
    PageSize,
    RunHistoryError,
    RunHistoryPage,
)
from storage.source_context import ContextDocumentIdentity

MAX_QUERY_BYTES = 16_384
_HistoryLimit = Annotated[PageSize, BeforeValidator(_query_integer)]
_HistoryCursor = Annotated[EventID, BeforeValidator(_query_integer)]


def _all_states(value: object) -> object:
    return None if value == "" else value


class _HistoryParameters(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    limit: _HistoryLimit = DEFAULT_PAGE_SIZE
    cursor: _HistoryCursor | None = None
    state: Annotated[AgentState | None, BeforeValidator(_all_states)] = None
    catalog_limit: _CatalogLimit = DEFAULT_PAGE_SIZE
    source: _OptionalSource = None
    title: _OptionalTitle = None
    catalog_cursor: ContextDocumentIdentity | None = None

    def parameters(self) -> dict[str, str | int | None]:
        return {
            "limit": self.limit,
            "cursor": self.cursor,
            "state": self.state,
            "catalog_limit": self.catalog_limit,
            "source": self.source,
            "title": self.title,
            "catalog_cursor": self.catalog_cursor,
        }

    def catalog_parameters(self) -> dict[str, str | int | None]:
        return {
            "limit": self.catalog_limit,
            "source": self.source,
            "title": self.title,
            "cursor": self.catalog_cursor,
        }


class _RunParameters(_HistoryParameters):
    run_id: Identity


def _navigation(options: _HistoryParameters | None, root_path: str) -> str:
    if options is None:
        return (
            '<nav aria-label="Saved run navigation">'
            + _link("reset-runs", root_path + "/explore/runs", "Reset saved-run selection")
            + _link("back-to-catalog", root_path + "/explore", "Back to current corpus")
            + "</nav>"
        )
    return (
        '<nav aria-label="Saved run navigation">'
        + _link(
            "back-to-runs",
            _url(root_path + "/explore/runs", options.parameters()),
            "Back to saved runs",
        )
        + _link(
            "back-to-catalog",
            _url(root_path + "/explore", options.catalog_parameters()),
            "Back to filtered current corpus",
        )
        + "</nav>"
    )


_EVIDENCE_ERRORS = {
    "run_not_found": "No saved events exist for this exact run ID. No replacement was guessed.",
    "run_incomplete": "No terminal DONE event was saved. This is not a worker-liveness check.",
    "run_failed": "The first terminal event records failure. No successful evidence is available.",
    "snapshot_unavailable": (
        "This legacy or uncaptured run has no saved evidence snapshot. Recorded DONE is not "
        "proof of exportability. Missing evidence is never reconstructed from today's corpus."
    ),
    "invalid_run_record": (
        "Saved evidence is inconsistent, invalid, or unsupported. No partial record is shown."
    ),
    "evidence_read_limit_exceeded": (
        "Saved events exceed the read limits: 100 events, 1 MiB per payload or 8 MiB total "
        "UTF-8 event text. No partial record is shown, including for oversized later events."
    ),
    "html_text_not_representable": (
        "Saved text cannot be represented faithfully as UTF-8 HTML, for example embedded NUL. "
        "Inspect the exact record through the read-only Python/JSON interfaces."
    ),
    "html_export_too_large": (
        "The complete escaped HTML, including navigation, exceeds 4 MiB. Nothing was truncated. "
        "Use the Python exporter or existing JSON/Markdown download for this run."
    ),
    "invalid_evidence_request": "Use an exact run ID containing 1 to 256 Unicode characters.",
    "evidence_storage_unavailable": (
        "Saved evidence storage is unavailable; this is not an empty run."
    ),
}


class _RunExplorerRoute(APIRoute):
    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None:
        if self.methods is not None and scope["method"] not in self.methods:
            response = _error(
                405,
                "saved_run_method_not_allowed",
                "Saved-run browsing requires GET",
                "These pages only read saved records. No operation was executed.",
                navigation=_navigation(None, scope.get("root_path", "").rstrip("/")),
                saved_runs=True,
            )
            response.headers["Allow"] = ", ".join(sorted(self.methods))
            await response(scope, receive, send)
            return
        await super().handle(scope, receive, send)

    def get_route_handler(self) -> Callable[[Request], Coroutine[None, None, Response]]:
        handler = super().get_route_handler()

        async def read(request: Request) -> Response:
            root_path = request.scope.get("root_path", "").rstrip("/")

            def error(status: int, code: str, title: str, message: str) -> HTMLResponse:
                options = getattr(request.state, "saved_run_options", None)
                return _error(
                    status,
                    code,
                    title,
                    message,
                    navigation=_navigation(options, root_path),
                    saved_runs=True,
                )

            try:
                raw = request.scope.get("query_string", b"")
                if len(raw) > MAX_QUERY_BYTES:
                    raise RequestValidationError([])
                try:
                    unquote_to_bytes(raw).decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise RequestValidationError([]) from exc
                if len(request.query_params.multi_items()) != len(request.query_params):
                    raise RequestValidationError([])
                return await handler(request)
            except RequestValidationError:
                return error(
                    422,
                    "invalid_saved_run_request",
                    "Invalid saved-run request",
                    "Use page size 1-100, an exact uppercase recorded state (or blank for all), "
                    "and a positive first-event-ID cursor up to 9223372036854775807. "
                    "The evidence reader requires an exact run ID of 1-256 Unicode characters. "
                    "Only blank source/title return filters mean omission. Empty cursors, "
                    "unknown/duplicate fields, invalid UTF-8 and queries over 16384 bytes fail. "
                    "No invalid filter was ignored or selection widened.",
                )
            except RunHistoryError:
                return error(
                    409,
                    "invalid_run_record",
                    "Saved history cannot be displayed",
                    "A projected summary or its lookahead is invalid. No partial page is shown. "
                    "Inspect saved records with trusted local tools; browsing cannot repair them.",
                )
            except EvidenceExportError as exc:
                message = _EVIDENCE_ERRORS.get(exc.code)
                return error(
                    exc.status_code if message is not None else 409,
                    exc.code if message is not None else "saved_evidence_unavailable",
                    "Frozen evidence unavailable",
                    message
                    or "No complete saved evidence could be read. No partial record is shown.",
                )
            except sqlite3.Error:
                return error(
                    503,
                    "storage_unavailable",
                    "Saved history storage unavailable",
                    "The existing event database could not be read. Check local availability "
                    "and permissions; this is not an empty result.",
                )
            except _PageTooLargeError:
                return error(
                    413,
                    "html_page_too_large",
                    "History exceeds the HTML limit",
                    "The complete escaped history page exceeds 1 MiB. Nothing was truncated "
                    "or skipped; request a smaller page size.",
                )
            except (_UnrepresentableTextError, UnicodeError):
                return error(
                    409,
                    "html_text_not_representable",
                    "History text cannot be displayed faithfully",
                    "A saved preview cannot be represented faithfully as UTF-8 HTML. "
                    "Use the read-only JSON/Python history interface; no partial page is shown.",
                )

        return read


router = APIRouter(route_class=_RunExplorerRoute, tags=["Saved run browser"])
_RESPONSES: dict[int | str, dict[str, str]] = {
    404: {"description": "No events for the exact run ID"},
    405: {"description": "Native browsing accepts only GET"},
    409: {"description": "Invalid/incomplete evidence, read limits or unrepresentable text"},
    413: {"description": "Complete escaped HTML exceeds the fixed page limit"},
    422: {"description": "Invalid query, exact state/ID, cursor, or return navigation"},
    503: {"description": "Saved event storage unavailable"},
}


def _filters(options: _HistoryParameters, root_path: str) -> Iterator[str]:
    yield '<section aria-labelledby="filters"><h2 id="filters">Filter recorded runs</h2>'
    if any(
        isinstance(value, str) and any(character in value for character in "\0\r\n")
        for value in options.parameters().values()
    ):
        yield (
            "<p>Catalog return filters contain control characters that native inputs cannot "
            "preserve. Exact page, evidence and return links remain available. "
            "Reset those filters in the current corpus catalog to edit this form.</p></section>"
        )
        return
    yield (
        '<form id="run-filters" method="get" '
        f'action="{_escaped(root_path + "/explore/runs")}" autocomplete="off">'
        '<div><label for="state">Exact recorded state</label>'
        '<select id="state" name="state" aria-describedby="state-help">'
        f'<option value=""{" selected" if options.state is None else ""}>'
        "All recorded states</option>"
    )
    for state in AgentState:
        selected = " selected" if state == options.state else ""
        yield f'<option value="{state}"{selected}>{state}</option>'
    yield (
        '</select><p id="state-help">Latest saved transition, not a live worker or an '
        'exportability check.</p></div><div><label for="limit">Runs per page</label>'
        f'<input id="limit" name="limit" type="number" min="1" max="100" value="{options.limit}" '
        'required inputmode="numeric" aria-describedby="limit-help">'
        '<p id="limit-help">1-100; newest first-event ID first.</p></div>'
    )
    for name in ("catalog_limit", "source", "title", "catalog_cursor"):
        value = options.parameters()[name]
        if value is not None:
            yield f'<input type="hidden" name="{name}" value="{_escaped(str(value))}">'
    yield (
        '<div class="actions"><button type="submit">Apply recorded-state filter</button>'
        + _link(
            "all-states",
            _url(
                root_path + "/explore/runs", {**options.parameters(), "state": None, "cursor": None}
            ),
            "Show all states",
        )
        + "</div></form><p>Submitting starts a new traversal without the old cursor. "
        "Catalog filters are only return navigation, not run filters.</p></section>"
    )


def _history(page: RunHistoryPage, options: _HistoryParameters, root_path: str) -> Iterator[str]:
    yield _navigation(options, root_path)
    yield from _filters(options, root_path)
    yield (
        '<section aria-labelledby="runs"><h2 id="runs">Persisted research runs</h2>'
        f"<p>{len(page.runs)} runs on this page (limit {options.limit}). "
        "Newest-created first by descending first-event ID, not by timestamp or latest activity. "
        "Pagination is exclusive: only smaller first-event IDs follow the cursor.</p>"
        "<p>Each page is one read snapshot, not a frozen multi-page state selection. "
        "Later transitions can change filter membership; restart to discover newly created runs. "
        "Inspecting evidence checks the capture; recorded DONE alone does not make it "
        "exportable.</p>"
    )
    if not page.runs:
        message = (
            "No matching runs remain after this cursor. Start again with the same selection."
            if options.cursor is not None
            else "No runs match this exact recorded-state selection."
            if options.state is not None
            else "No persisted runs in this database. Browsing never creates a run."
        )
        yield f'<p class="notice">{message}</p>'
    for number, run in enumerate(page.runs, 1):
        yield (
            f'<article aria-labelledby="run-{number}"><h3 id="run-{number}">Saved run {number}</h3>'
        )
        yield "<dl><dt>Run ID (JSON)</dt><dd>" + _identity(run.run_id, "run-id") + "</dd>"
        yield "<dt>Agent ID (JSON)</dt><dd>" + _identity(run.agent_id, "agent-id") + "</dd>"
        yield (
            '<dt>First event ID</dt><dd class="first-event-id">' + str(run.first_event_id) + "</dd>"
        )
        yield '<dt>Recorded state</dt><dd class="recorded-state">'
        yield run.recorded_state.value if run.recorded_state is not None else "Not recorded"
        yield f"</dd><dt>Saved events</dt><dd>{run.event_count}</dd>"
        for label, timestamp in (
            ("First event time", run.started_at),
            ("Last event time", run.updated_at),
        ):
            value = timestamp.isoformat()
            yield f'<dt>{label}</dt><dd><time datetime="{value}">{value}</time></dd>'
        yield "</dl><h4>Recorded query preview</h4>"
        yield (
            "<p>No planning query was recorded.</p>"
            if run.query_summary is None
            else "<p>The recorded query is empty.</p>"
            if run.query_summary == ""
            else _literal(run.query_summary, "query-preview")
        )
        yield _truncation("Query preview", run.query_truncated, MAX_QUERY_CHARACTERS)
        yield _link(
            f"inspect-run-{number}",
            _url(root_path + "/explore/run", {"run_id": run.run_id, **options.parameters()}),
            "Inspect saved evidence",
        )
        yield "</article>"
    yield _pagination(
        _url(root_path + "/explore/runs", {**options.parameters(), "cursor": None}),
        _url(root_path + "/explore/runs", {**options.parameters(), "cursor": page.next_cursor})
        if page.next_cursor is not None
        else None,
    )
    yield "</section>"


@router.get("/explore/runs", response_class=HTMLResponse, responses=_RESPONSES)
def explore_runs(request: Request, options: Annotated[_HistoryParameters, Query()]) -> HTMLResponse:
    """Discover recorded states, never infer liveness or eagerly read evidence payloads."""
    request.state.saved_run_options = options
    container: AppContainer = request.app.state.container
    page = container.run_history.list_runs(
        limit=options.limit, cursor=options.cursor, state=options.state
    )
    return _page(
        "Saved run browser",
        _history(page, options, request.scope.get("root_path", "").rstrip("/")),
        saved_runs=True,
    )


@router.get("/explore/run", response_class=HTMLResponse, responses=_RESPONSES)
def explore_run(request: Request, options: Annotated[_RunParameters, Query()]) -> HTMLResponse:
    """Open a bounded, validated frozen capture inline; never consult current source passages."""
    request.state.saved_run_options = options
    container: AppContainer = request.app.state.container
    bundle = container.saved_run_evidence.export(options.run_id)
    root_path = request.scope.get("root_path", "").rstrip("/")
    export_url = (
        root_path + "/runs/" + quote(options.run_id, safe="") + "/export"
        if "/" not in options.run_id and options.run_id not in {".", ".."}
        else None
    )
    html = render_html(
        bundle,
        navigation=BrowserNavigation(
            history_url=_url(root_path + "/explore/runs", options.parameters()),
            catalog_url=_url(root_path + "/explore", options.catalog_parameters()),
            export_url=export_url,
        ),
    )
    return HTMLResponse(
        html,
        headers={
            "Content-Security-Policy": HTML_CSP + "; frame-ancestors 'none'",
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
            "X-Frame-Options": "DENY",
        },
    )
