"""Script-free collection discovery over the existing bounded screening export snapshot."""

import sqlite3
from collections.abc import Callable, Coroutine, Iterator
from typing import Annotated
from urllib.parse import unquote_to_bytes

from fastapi import APIRouter, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict
from starlette.types import Receive, Scope, Send

from api.dependencies import AppContainer
from api.explorer import (
    _CatalogLimit,
    _CollectionReturn,
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
    _truncation,
    _UnrepresentableTextError,
    _url,
)
from storage.document_catalog import DEFAULT_PAGE_SIZE, MAX_SOURCE_CHARACTERS, MAX_TITLE_CHARACTERS
from storage.paper_collections import CollectionError, CollectionId, CollectionPage
from storage.paper_screening import ScreeningExport, ScreeningExportItem
from storage.source_context import ContextDocumentIdentity

MAX_QUERY_BYTES = 16_384
_STATUSES = ("unscreened", "include", "exclude", "unsure", "stale")


class _ListParameters(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    limit: _CatalogLimit = DEFAULT_PAGE_SIZE
    cursor: CollectionId | None = None
    catalog_limit: _CatalogLimit = DEFAULT_PAGE_SIZE
    source: _OptionalSource = None
    title: _OptionalTitle = None
    catalog_cursor: ContextDocumentIdentity | None = None

    def parameters(self) -> dict[str, str | int | None]:
        return self.model_dump()


def _list_options(options: _CollectionReturn) -> _ListParameters:
    return _ListParameters(
        limit=options.collections_limit,
        cursor=options.collections_cursor,
        catalog_limit=options.catalog_limit,
        source=options.source,
        title=options.title,
        catalog_cursor=options.catalog_cursor,
    )


def _navigation(options: _ListParameters | _CollectionReturn | None, root_path: str) -> str:
    listed = _list_options(options) if isinstance(options, _CollectionReturn) else options
    listed = listed or _ListParameters()
    return (
        '<nav aria-label="Collection navigation">'
        + _link(
            "back-to-collections",
            _url(root_path + "/explore/collections", listed.parameters()),
            "Back to saved collections",
        )
        + _link(
            "back-to-catalog",
            _url(
                root_path + "/explore",
                {
                    "limit": listed.catalog_limit,
                    "source": listed.source,
                    "title": listed.title,
                    "cursor": listed.catalog_cursor,
                },
            ),
            "Back to filtered catalog",
        )
        + "</nav>"
    )


_ERRORS = {
    "collection_not_found": "The exact collection no longer exists. No replacement was guessed.",
    "collection_revision_conflict": (
        "The collection revision changed, even if only its name changed. Return to the "
        "collection list and open its current revision. No retry or scope change occurred."
    ),
    "collection_documents_missing": (
        "A saved member is missing from the corpus. No partial membership or screening is shown."
    ),
    "invalid_collection_record": "Saved collection metadata is invalid; nothing was repaired.",
    "invalid_screening_record": "Saved human screening is invalid or inconsistent.",
    "invalid_document_record": "A current member's stored title or source is invalid.",
    "screening_cursor_invalid": "The member cursor is not in this collection revision.",
    "screening_export_too_large": (
        "The complete screening JSON snapshot exceeds 524288 UTF-8 bytes. No partial result "
        "is shown, even with a smaller visible page."
    ),
    "collection_storage_error": (
        "Collection storage is unavailable; this is not an empty selection."
    ),
    "screening_storage_unavailable": "Screening storage is unavailable, not an unscreened result.",
}


class _CollectionExplorerRoute(APIRoute):
    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None:
        if self.methods is not None and scope["method"] not in self.methods:
            response = _error(
                405,
                "collection_browser_method_not_allowed",
                "Collection browsing requires GET",
                "No operation was executed. This browser has no editing or screening actions.",
                navigation=_navigation(None, scope.get("root_path", "").rstrip("/")),
                collections=True,
            )
            response.headers["Allow"] = "GET"
            await response(scope, receive, send)
            return
        await super().handle(scope, receive, send)

    def get_route_handler(self) -> Callable[[Request], Coroutine[None, None, Response]]:
        handler = super().get_route_handler()

        async def read(request: Request) -> Response:
            root_path = request.scope.get("root_path", "").rstrip("/")

            def error(status: int, code: str, title: str, message: str) -> HTMLResponse:
                return _error(
                    status,
                    code,
                    title,
                    message,
                    navigation=_navigation(
                        getattr(request.state, "collection_browser_options", None), root_path
                    ),
                    collections=True,
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
                    "invalid_collection_browser_request",
                    "Invalid collection browser request",
                    "Use exact collection IDs, a required positive integer collection_revision "
                    "for detail, page sizes 1-100, and returned cursors. Status is unscreened, "
                    "include, exclude, unsure, stale, or blank for all. Unknown or repeated "
                    "fields, invalid UTF-8, empty cursors and queries over 16384 bytes fail. "
                    "No filter or scope was silently discarded.",
                )
            except CollectionError as exc:
                return error(
                    exc.status_code,
                    exc.code,
                    "Collection inspection unavailable",
                    _ERRORS.get(exc.code, "No complete collection inspection could be read."),
                )
            except sqlite3.Error:
                return error(
                    503,
                    "collection_storage_error",
                    "Storage unavailable",
                    _ERRORS["collection_storage_error"],
                )
            except _PageTooLargeError:
                return error(
                    413,
                    "html_page_too_large",
                    "Collection page exceeds its size limit",
                    "The complete escaped HTML exceeds 1 MiB. No partial page is shown. "
                    "Request a smaller visible page.",
                )
            except (_UnrepresentableTextError, UnicodeError):
                return error(
                    409,
                    "html_text_not_representable",
                    "Text cannot be displayed faithfully",
                    "A label, reason or return filter cannot be represented as UTF-8 HTML. "
                    "No partial page is shown. Use the existing JSON/Python readers.",
                )

        return read


router = APIRouter(route_class=_CollectionExplorerRoute, tags=["Saved collection browser"])
_RESPONSES: dict[int | str, dict[str, str]] = {
    404: {"description": "Exact collection missing"},
    405: {"description": "Browser accepts only GET"},
    409: {"description": "Stale revision, missing member, invalid stored data or HTML text"},
    413: {"description": "Complete screening snapshot or escaped HTML exceeds its byte bound"},
    422: {"description": "Invalid query, status, cursor, revision or return selection"},
    503: {"description": "Existing storage unavailable"},
}


def _filters(options: _ListParameters | _CollectionReturn, root_path: str) -> Iterator[str]:
    detail = isinstance(options, _CollectionReturn)
    params = options.parameters()
    hidden = {
        key: value for key, value in params.items() if key not in {"limit", "cursor", "status"}
    }
    if any(
        isinstance(value, str) and any(character in value for character in "\0\r\n")
        for value in hidden.values()
    ):
        yield (
            '<p class="notice">Return filters contain control characters that native inputs '
            "cannot preserve. Use the exact pagination and status links instead.</p>"
        )
        return
    identifier = "member-filters" if detail else "collection-filters"
    endpoint = "/explore/collection" if detail else "/explore/collections"
    yield (
        f'<form id="{identifier}" method="get" action="{_escaped(root_path + endpoint)}" '
        'autocomplete="off"><div><label for="limit">Items per page</label>'
        f'<input id="limit" name="limit" type="number" min="1" max="100" '
        f'value="{options.limit}" required inputmode="numeric"></div>'
    )
    if isinstance(options, _CollectionReturn):
        yield (
            '<div><label for="status">Human screening status</label>'
            '<select id="status" name="status">'
        )
        for status in ("", *_STATUSES):
            selected = " selected" if (options.status or "") == status else ""
            yield f'<option value="{status}"{selected}>{status or "All members"}</option>'
        yield "</select></div>"
    for key, value in hidden.items():
        if value is not None:
            yield f'<input type="hidden" name="{key}" value="{_escaped(str(value))}">'
    yield (
        '<div class="actions"><button type="submit">Apply / start at first page</button></div>'
        "</form>"
    )


def _list(page: CollectionPage, options: _ListParameters, root_path: str) -> Iterator[str]:
    yield _navigation(options, root_path)
    yield (
        '<p class="notice">Current saved selections, not screened-in papers. '
        "Membership counts are not screening decisions. Open a collection to inspect "
        "current human statuses. Catalog filters are return state, not collection filters.</p>"
    )
    yield from _filters(options, root_path)
    yield (
        f"<p>{len(page.collections)} collections on this page. Ascending SQLite BINARY collection "
        "ID order, not creation time. Membership and labels can change between requests.</p>"
    )
    if not page.collections:
        yield (
            "<p>No saved collections after this cursor. Start again to see current selections.</p>"
            if options.cursor
            else "<p>No saved collections. Save a selection through the existing API or Python "
            "interface; this browser does not create collections.</p>"
        )
    for index, collection in enumerate(page.collections, 1):
        detail = _CollectionReturn(
            collection_id=collection.collection_id,
            collection_revision=collection.revision,
            collections_limit=options.limit,
            collections_cursor=options.cursor,
            catalog_limit=options.catalog_limit,
            source=options.source,
            title=options.title,
            catalog_cursor=options.catalog_cursor,
        )
        yield f"<article><h2>Saved collection {index}</h2>"
        yield _literal(collection.name, "collection-name")
        yield _identity(collection.collection_id, "collection-id")
        yield (
            f"<p>Collection revision: {collection.revision}. "
            f'Members: <span class="member-count">{collection.document_count}</span>.</p>'
        )
        yield _link(
            f"inspect-collection-{index}",
            _url(root_path + "/explore/collection", detail.parameters()),
            "Inspect membership and human screening",
        )
        yield "</article>"
    params = options.parameters()
    yield _pagination(
        _url(root_path + "/explore/collections", {**params, "cursor": None}),
        _url(root_path + "/explore/collections", {**params, "cursor": page.next_cursor})
        if page.next_cursor
        else None,
    )


def _member(
    item: ScreeningExportItem, index: int, options: _CollectionReturn, root_path: str
) -> Iterator[str]:
    yield f"<article><h3>Member {index} on this page</h3>"
    yield _identity(item.document_id, "document-id")
    yield _literal(item.title, "member-title")
    yield _truncation("Title", item.title_truncated, MAX_TITLE_CHARACTERS)
    yield _literal(item.source, "member-source")
    yield _truncation("Source", item.source_truncated, MAX_SOURCE_CHARACTERS)
    yield f'<p>Human screening status: <strong class="member-status">{item.status}</strong>.</p>'
    if item.review is None:
        yield "<p>No human decision is recorded for this member.</p>"
    else:
        review = item.review
        yield (
            "<h3>"
            + ("Earlier decision (stale)" if item.status == "stale" else "Recorded decision")
            + "</h3>"
        )
        yield (
            f'<p>Decision: <span class="review-decision">{review.decision}</span>; '
            f"decision revision: {review.revision}; reviewed collection revision: "
            f"{review.collection_revision}; updated at "
            f'<time datetime="{review.updated_at.isoformat()}">'
            f"{review.updated_at.isoformat()}</time>.</p>"
        )
        yield _literal(review.reason, "review-reason")
    yield _link(
        f"inspect-member-{index}",
        _url(
            root_path + "/explore/document",
            {
                "document_id": item.document_id,
                "limit": options.catalog_limit,
                "source": options.source,
                "title": options.title,
                "catalog_cursor": options.catalog_cursor,
                "collection_return": options.token(),
            },
        ),
        "Read this exact paper / choose source context",
    )
    yield "</article>"


def _detail(snapshot: ScreeningExport, options: _CollectionReturn, root_path: str) -> Iterator[str]:
    identifiers = [item.document_id for item in snapshot.items]
    if options.cursor is not None and options.cursor not in identifiers:
        raise CollectionError(
            "screening_cursor_invalid", "Cursor must identify a collection member.", 422
        )
    start = 0 if options.cursor is None else identifiers.index(options.cursor) + 1
    selected = [
        item
        for item in snapshot.items[start:]
        if options.status is None or item.status == options.status
    ]
    yield _navigation(options, root_path)
    yield _literal(snapshot.collection_name, "collection-name")
    yield _identity(snapshot.collection_id, "collection-id")
    yield (
        f"<p>Collection revision: {snapshot.collection_revision}; "
        f"current members: {snapshot.total_documents}. "
        "These are members, not screened-in papers.</p>"
        '<p class="notice">Membership, labels, decisions and counts on this page come from one '
        "complete screening export read. Separate requests are not one snapshot. A collection "
        "revision binds membership, not later decision edits or corpus contents.</p>"
        "<h2>Human screening counts (all members)</h2><dl>"
    )
    for status, count in snapshot.counts.model_dump().items():
        yield f'<dt>{status}</dt><dd id="count-{status}">{count}</dd>'
    yield "</dl><details><summary>Current human include IDs only (JSON)</summary>"
    if not snapshot.included_document_ids:
        yield "<p>No current includes. An empty selection never means the whole corpus.</p>"
    for identifier in snapshot.included_document_ids:
        yield _identity(identifier, "included-document-id")
    yield "</details><h2>Current-corpus tools</h2>"
    yield (
        "<p>Search and worksheet use all current members, not this status filter or the include "
        "IDs. They resolve again when used. Worksheet accepts at most 10 members and fails "
        "rather than silently dropping extras. Open a paper to choose an exact stored passage "
        "for source context; missing chunks are never guessed.</p>"
        '<nav aria-label="Collection tools">'
    )
    navigation = {
        "catalog_limit": options.catalog_limit,
        "source": options.source,
        "title": options.title,
        "catalog_cursor": options.catalog_cursor,
        "collection_return": options.token(),
    }
    yield _link(
        "search-collection",
        _url(
            root_path + "/explore/search",
            {"scope": "collection", "scope_id": snapshot.collection_id, **navigation},
        ),
        "Search all current collection members",
    )
    yield _link(
        "worksheet-collection",
        _url(
            root_path + "/explore/worksheet",
            {"collection_id": snapshot.collection_id, **navigation},
        ),
        "Open collection worksheet",
    )
    yield "</nav><h2>Revision-bound downloads</h2>"
    yield (
        "<p>Downloads reread all members, regardless of visible filters. Revision changes fail "
        "explicitly. Decision or corpus edits at the same revision can change the output. "
        "Duplicate reports contain candidates, not automatic removals or screening decisions. "
        "CSV uses reversible apostrophe-prefix-v1 text encoding.</p>"
        '<nav aria-label="Collection downloads">'
    )
    for identifier, endpoint, format, label in (
        ("screening-json", "screening/export", "json", "Screening JSON"),
        ("screening-csv", "screening/export", "csv", "Screening CSV"),
        ("duplicates-json", "duplicates", "json", "Duplicate candidates JSON"),
        ("duplicates-markdown", "duplicates", "markdown", "Duplicate candidates Markdown"),
    ):
        yield _link(
            f"download-{identifier}",
            _url(
                f"{root_path}/collections/{snapshot.collection_id}/{endpoint}",
                {"collection_revision": snapshot.collection_revision, "format": format},
            ),
            label,
        )
    yield "</nav><h2>Current membership</h2>"
    yield from _filters(options, root_path)
    yield '<nav aria-label="Status shortcuts">'
    for filter_status in (None, *_STATUSES):
        yield _link(
            f"status-{filter_status or 'all'}",
            _url(
                root_path + "/explore/collection",
                {**options.parameters(), "status": filter_status, "cursor": None},
            ),
            filter_status or "All members",
        )
    yield "</nav>"
    yield (
        f"<p>{min(len(selected), options.limit)} members on this page, in saved membership order. "
        "Counts and include IDs above cover the whole collection, not just this page. "
        "Human statuses may change between pages; start again for a fresh traversal.</p>"
    )
    if not selected:
        yield "<p>No members match this status after this cursor. No selection was widened.</p>"
    for index, item in enumerate(selected[: options.limit], 1):
        yield from _member(item, index, options, root_path)
    yield _pagination(
        _url(root_path + "/explore/collection", {**options.parameters(), "cursor": None}),
        _url(
            root_path + "/explore/collection",
            {**options.parameters(), "cursor": selected[options.limit - 1].document_id},
        )
        if len(selected) > options.limit
        else None,
    )


@router.get("/explore/collections", response_class=HTMLResponse, responses=_RESPONSES)
def explore_collections(
    request: Request, options: Annotated[_ListParameters, Query()]
) -> HTMLResponse:
    request.state.collection_browser_options = options
    container: AppContainer = request.app.state.container
    page = container.paper_collections.list_collections(limit=options.limit, cursor=options.cursor)
    return _page(
        "Saved paper collections",
        _list(page, options, request.scope.get("root_path", "").rstrip("/")),
        collections=True,
    )


@router.get("/explore/collection", response_class=HTMLResponse, responses=_RESPONSES)
def explore_collection(
    request: Request, options: Annotated[_CollectionReturn, Query()]
) -> HTMLResponse:
    request.state.collection_browser_options = options
    container: AppContainer = request.app.state.container
    download = container.paper_screening.export_results(
        options.collection_id, collection_revision=options.collection_revision, format="json"
    )
    snapshot = ScreeningExport.model_validate_json(download.content)
    return _page(
        "Collection membership and screening",
        _detail(snapshot, options, request.scope.get("root_path", "").rstrip("/")),
        collections=True,
    )
