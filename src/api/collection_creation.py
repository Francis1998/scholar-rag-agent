"""Explicit native creation of a named selection through the existing atomic store."""

import json
import sqlite3
from collections.abc import Iterator
from typing import Annotated

from fastapi import APIRouter, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field
from starlette.concurrency import run_in_threadpool
from starlette.types import Receive, Scope, Send

from api.collection_explorer import _ListParameters, _navigation
from api.dependencies import AppContainer
from api.explorer import (
    _DOCUMENT_ID,
    _WRITE_HEADERS,
    _error,
    _escaped,
    _filters,
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
from api.native_forms import (
    FormTooLargeError,
    parse_fields,
    read_body,
    require_same_origin,
    require_urlencoded,
    unsigned_decimal,
)
from retrieval.scope import MAX_DOCUMENT_ID_LENGTH, MAX_DOCUMENT_IDS
from storage.document_catalog import (
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    MAX_SOURCE_CHARACTERS,
    MAX_TITLE_CHARACTERS,
    DocumentCatalogError,
    DocumentCatalogPage,
)
from storage.paper_collections import MAX_NAME_LENGTH, CollectionError, CollectionSelection
from storage.source_context import ContextDocumentIdentity

PATH = "/explore/collections/new"
MAX_QUERY_BYTES = 16_384
MAX_FORM_BYTES = 262_144
# JSON can encode one supplementary Unicode code point as two six-character escapes.
MAX_FIELD_CHARACTERS = 12 * MAX_DOCUMENT_ID_LENGTH + 2
_DOCUMENT_FIELDS = tuple(f"document_{index}" for index in range(1, MAX_DOCUMENT_IDS + 1))


class _Open(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    limit: Annotated[
        int, Field(strict=True, ge=1, le=MAX_PAGE_SIZE), BeforeValidator(unsigned_decimal)
    ] = DEFAULT_PAGE_SIZE
    source: _OptionalSource = None
    title: _OptionalTitle = None
    cursor: ContextDocumentIdentity | None = None

    def parameters(self) -> dict[str, str | int | None]:
        return self.model_dump()


_FAILURES = {
    "invalid_collection_form": (
        "Choose a name of 1-120 characters after trimming and 1-100 distinct exact document "
        "IDs. Names cannot contain control characters; IDs must not have surrounding "
        "whitespace. Unknown/repeated fields, invalid JSON-string IDs, malformed percent "
        "encoding or UTF-8, GET bodies and POST query fields are rejected. "
        "GET filters accept limit 1-100, source, title and an exact cursor. "
        "No submitted input is echoed and nothing was saved."
    ),
    "collection_form_origin": (
        "Open the local form again: one exact same-origin Origin (scheme, host and port) is "
        "required. Fetch Metadata, when present, must also be exactly same-origin. "
        "Ambiguous or cross-origin submissions do not save anything."
    ),
    "collection_form_encoding": (
        "Use a UTF-8 URL-encoded form without content encoding. Nothing was saved."
    ),
    "collection_form_too_large": (
        "The query exceeds 16384 bytes or the form body exceeds 262144 bytes. Nothing was saved."
    ),
    "collection_form_method": (
        "Use GET to choose papers and an explicit POST to create a collection."
    ),
    "unknown_documents": (
        "At least one selected exact ID is unknown or was deleted after the form opened. "
        "No partial collection was saved. Reopen the current catalog and select again."
    ),
    "collection_name_conflict": (
        "A collection with that exact, trimmed name already exists, possibly from another "
        "submission. Nothing was overwritten. Inspect saved collections before retrying."
    ),
    "collection_storage_error": (
        "Collection storage is unavailable or busy. No successful save is being reported. "
        "Inspect saved collections before retrying after an interrupted request."
    ),
    "invalid_document_record": (
        "A stored document summary is invalid or not selectable. No partial form is shown."
    ),
    "html_page_too_large": (
        "The complete escaped page exceeds 1 MiB. No partial form is shown; use a smaller page."
    ),
    "html_text_not_representable": (
        "A stored label cannot be displayed faithfully as UTF-8 HTML. "
        "No partial form is shown; use the existing JSON/Python readers."
    ),
}


def _failure(status: int, code: str, root_path: str) -> HTMLResponse:
    return _error(
        status,
        code,
        "Collection creation unavailable",
        _FAILURES.get(code, "The collection service failed. No success is being reported."),
        navigation=_navigation(None, root_path)
        + _link("reopen-creation", root_path + PATH, "Open a fresh creation form"),
        creation=True,
    )


class _CreationRoute(APIRoute):
    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None:
        if self.methods is not None and scope["method"] not in self.methods:
            response = _failure(
                405, "collection_form_method", scope.get("root_path", "").rstrip("/")
            )
            response.headers["Allow"] = "GET, POST"
            await response(scope, receive, send)
            return
        await super().handle(scope, receive, send)


def _fragments(page: DocumentCatalogPage, options: _Open, root_path: str) -> Iterator[str]:
    yield _navigation(
        _ListParameters(
            catalog_limit=options.limit,
            source=options.source,
            title=options.title,
            catalog_cursor=options.cursor,
        ),
        root_path,
    )
    yield (
        '<p class="notice" id="selection-help">Only checked papers on this displayed page '
        "are submitted. Changing filters or pages opens a fresh form: name and choices are "
        "not carried forward. No paper is preselected, no hidden pages are selected, and "
        "there is no multi-page cart. Inspecting a paper also leaves the unsaved form.</p>"
    )
    yield from _filters(options.limit, options.source, options.title, root_path, endpoint=PATH)
    yield (
        f'<p><span id="catalog-count">{len(page.documents)}</span> papers on this page '
        f"(limit {options.limit}). Ascending document ID in SQLite BINARY order, not creation "
        "time. Separate pages are not a frozen corpus snapshot.</p>"
    )
    if page.documents:
        yield (
            '<form id="collection-create" method="post" '
            f'action="{_escaped(root_path + PATH)}" '
            'enctype="application/x-www-form-urlencoded" accept-charset="UTF-8" '
            'autocomplete="off" aria-describedby="selection-help">'
            '<div class="worksheet-input"><label for="collection-name">Collection name</label>'
            f'<input id="collection-name" name="name" type="text" required '
            f'maxlength="{2 * MAX_NAME_LENGTH}" aria-describedby="name-help">'
            '<p id="name-help">1-120 Unicode characters after trimming surrounding whitespace; '
            "no control characters. Exact case-sensitive names must be unique. "
            "The server enforces the character limit; browsers count UTF-16 units.</p></div>"
            '<fieldset class="worksheet-input collection-choices">'
            "<legend>Explicitly select one or more exact papers</legend>"
        )
        for index, document in enumerate(page.documents, 1):
            field = f"document_{index}"
            value = _escaped(json.dumps(document.document_id, ensure_ascii=True))
            yield (
                f'<article><label class="collection-choice" for="{field}">'
                f'<input id="{field}" type="checkbox" name="{field}" value="{value}">'
                f"Select paper {index}</label>"
            )
            yield _literal(document.title or "Untitled paper", "document-title")
            yield _truncation("Title", document.title_truncated, MAX_TITLE_CHARACTERS)
            yield "<dl><dt>Exact document ID (JSON)</dt><dd>"
            yield _identity(document.document_id, "document-id")
            yield "</dd><dt>Source</dt><dd>" + _literal(document.source, "document-source")
            yield f"</dd><dt>Stored chunks</dt><dd>{document.chunk_count}</dd></dl>"
            yield _truncation("Source", document.source_truncated, MAX_SOURCE_CHARACTERS)
            yield _link(
                f"inspect-document-{index}",
                _url(
                    root_path + "/explore/document",
                    {
                        "document_id": document.document_id,
                        "limit": options.limit,
                        "source": options.source,
                        "title": options.title,
                        "catalog_cursor": options.cursor,
                    },
                ),
                "Inspect paper (leaves unsaved form)",
            )
            yield "</article>"
        yield (
            '</fieldset><div class="actions"><button type="submit">Create collection</button>'
            "</div></form>"
        )
    else:
        yield (
            '<p class="notice">No papers on this page. Adjust filters, start again, or ingest '
            "permitted material through the existing ingestion tools. An empty collection "
            "cannot be created.</p>"
        )
    yield _pagination(
        _url(root_path + PATH, {**options.parameters(), "cursor": None}),
        _url(root_path + PATH, {**options.parameters(), "cursor": page.next_cursor})
        if page.next_cursor is not None
        else None,
    )


def _open(container: AppContainer, options: _Open, root_path: str) -> HTMLResponse:
    page = container.document_catalog.list_documents(
        limit=options.limit, source=options.source, title=options.title, cursor=options.cursor
    )
    return _page("Create a paper collection", _fragments(page, options, root_path), creation=True)


def _selection(fields: dict[str, str]) -> CollectionSelection:
    if "name" not in fields or set(fields) - {"name", *_DOCUMENT_FIELDS}:
        raise ValueError("Expected name and bounded indexed document choices.")
    identifiers = [
        _DOCUMENT_ID.validate_json(fields[field]) for field in _DOCUMENT_FIELDS if field in fields
    ]
    # Validate exact identities before the shared store's normalization/deduplication.
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Repeated document selections are ambiguous.")
    return CollectionSelection(name=fields["name"], document_ids=tuple(identifiers))


router = APIRouter(route_class=_CreationRoute, tags=["Native collection creation"])


@router.get(PATH, response_class=HTMLResponse)
@router.post(
    PATH,
    response_class=HTMLResponse,
    responses={
        303: {"description": "Atomic save; open the persisted collection and actual revision"},
        403: {"description": "Missing, ambiguous or cross-origin browser submission"},
        405: {"description": "Only GET and explicit POST are supported"},
        409: {"description": "Duplicate name or invalid/unrepresentable corpus labels"},
        413: {"description": "Query, body or complete escaped HTML exceeds its byte bound"},
        415: {"description": "Unsupported form encoding"},
        422: {"description": "Invalid form/filter or unknown/deleted exact document ID"},
        503: {"description": "Corpus/collection storage unavailable"},
    },
)
async def create_collection(request: Request) -> Response:
    root_path = request.scope.get("root_path", "").rstrip("/")
    try:
        if request.method == "POST":
            try:
                require_same_origin(request)
            except ValueError as exc:
                raise CollectionError("collection_form_origin", "", 403) from exc
            try:
                require_urlencoded(request)
            except ValueError as exc:
                raise CollectionError("collection_form_encoding", "", 415) from exc
        query = request.scope.get("query_string", b"")
        if len(query) > MAX_QUERY_BYTES:
            raise FormTooLargeError()
        if request.method == "POST" and query:
            raise ValueError("POST fields belong only in the body.")
        body = await read_body(request, max_bytes=MAX_FORM_BYTES)
        container: AppContainer = request.app.state.container
        if request.method == "GET":
            if body:
                raise ValueError("GET cannot accept a body.")
            options = _Open.model_validate(
                parse_fields(query, max_fields=4, max_field_characters=MAX_SOURCE_CHARACTERS)
            )
            return await run_in_threadpool(_open, container, options, root_path)
        selection = _selection(
            parse_fields(
                body, max_fields=MAX_DOCUMENT_IDS + 1, max_field_characters=MAX_FIELD_CHARACTERS
            )
        )
        collection = await run_in_threadpool(
            container.paper_collections.create,
            name=selection.name,
            document_ids=selection.document_ids,
        )
        return RedirectResponse(
            _url(
                root_path + "/explore/collection",
                {
                    "collection_id": collection.collection_id,
                    "collection_revision": collection.revision,
                },
            ),
            status_code=303,
            headers=_WRITE_HEADERS,
        )
    except CollectionError as exc:
        return _failure(exc.status_code, exc.code, root_path)
    except sqlite3.Error:
        return _failure(503, "collection_storage_error", root_path)
    except DocumentCatalogError:
        return _failure(409, "invalid_document_record", root_path)
    except FormTooLargeError:
        return _failure(413, "collection_form_too_large", root_path)
    except _PageTooLargeError:
        return _failure(413, "html_page_too_large", root_path)
    except _UnrepresentableTextError:
        return _failure(409, "html_text_not_representable", root_path)
    except (ValueError, RecursionError):
        return _failure(422, "invalid_collection_form", root_path)
