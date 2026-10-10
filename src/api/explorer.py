"""Script-free, bounded HTML views of the existing read-only corpus readers."""

import base64
import hashlib
import json
import logging
import sqlite3
from collections.abc import Callable, Coroutine, Iterator, Mapping
from html import escape
from itertools import chain
from typing import Annotated, Literal, Self
from urllib.parse import unquote_to_bytes, urlencode

from fastapi import APIRouter, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse
from fastapi.routing import APIRoute
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from api.dependencies import AppContainer
from retrieval.scope import MAX_DOCUMENT_ID_LENGTH
from storage.document_catalog import (
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    MAX_SOURCE_CHARACTERS,
    MAX_TITLE_CHARACTERS,
    DocumentCatalogError,
    DocumentCatalogPage,
    Identity,
    SourceFilter,
    TitleFilter,
)
from storage.document_chunks import (
    MAX_TEXT_CHARACTERS,
    ChunkCursor,
    ChunkCursorError,
    DocumentChunksError,
    DocumentChunksPage,
    DocumentNotFoundError,
    StoredChunk,
)
from storage.literal_search import (
    DEFAULT_LIMIT as DEFAULT_SEARCH_LIMIT,
)
from storage.literal_search import (
    MAX_LIMIT as MAX_SEARCH_LIMIT,
)
from storage.literal_search import (
    LiteralMatch,
    LiteralSearchError,
    LiteralSearchPage,
    LiteralSearchRequest,
    SearchCursor,
    SearchQuery,
)
from storage.paper_collections import CollectionError, CollectionId
from storage.source_context import (
    DEFAULT_CONTEXT_NEIGHBORS,
    MAX_CONTEXT_NEIGHBORS,
    ContextChunkIdentity,
    ContextDocumentIdentity,
    SourceContext,
    SourceContextError,
)

logger = logging.getLogger(__name__)
MAX_HTML_BYTES = 1_048_576
_STYLE = """
:root { color-scheme: light; font-family: system-ui, sans-serif; line-height: 1.55; }
* { box-sizing: border-box; }
body { margin: 0; color: #182b3a; background: #f4f7fa; }
header, main, footer { max-width: 72rem; margin: auto; padding: 1.5rem; }
header { padding-bottom: .5rem; }
h1, h2, h3 { line-height: 1.25; overflow-wrap: anywhere; text-wrap: balance; }
main, h2, h3 { scroll-margin-top: 1rem; }
h1 { margin: .4rem 0; font-size: clamp(1.7rem, 4vw, 2.5rem); }
h2 { font-size: 1.35rem; }
h3 { font-size: 1.15rem; margin-top: 0; }
a { color: #164e82; text-decoration: underline; text-underline-offset: .18em; }
a:hover { color: #0c355a; text-decoration-thickness: .15em; }
a, button, input, select { touch-action: manipulation; -webkit-tap-highlight-color: #c7e1f4; }
:focus-visible { outline: 3px solid #125d9c; outline-offset: 4px; }
.skip-link { position: absolute; top: -5rem; left: 1rem; background: white; padding: .6rem; }
.skip-link:focus { top: .5rem; z-index: 1; }
.eyebrow { color: #28516b; font-weight: 700; letter-spacing: .06em; font-size: .85rem; }
.muted { color: #465e70; }
.notice { padding: .8rem 1rem; background: #fff5db; border-left: 4px solid #9c6a12; }
form, article { background: #fff; border: 1px solid #c4d3df; border-radius: .6rem; padding: 1rem; }
form { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) 9rem; gap: 1rem; }
form > div { min-width: 0; }
label { display: block; font-weight: 650; margin-bottom: .3rem; }
input, select { width: 100%; min-height: 2.75rem; font: inherit; border: 1px solid #647d91;
  border-radius: .3rem; padding: .5rem; background: #fff; color: #182b3a; }
form p { font-size: .9rem; margin: .4rem 0; }
button { font: inherit; font-weight: 650; min-height: 2.75rem; padding: .5rem 1rem;
  border: 1px solid #164e82; border-radius: .3rem; background: #164e82;
  color: #fff; cursor: pointer; }
button:hover { background: #0c355a; border-color: #0c355a; }
.actions { grid-column: 1 / -1; display: flex; align-items: center; flex-wrap: wrap; gap: 1rem; }
nav { display: flex; flex-wrap: wrap; gap: .7rem 1.5rem; margin: 1rem 0; }
nav a { display: inline-block; padding: .3rem 0; }
article { margin: 1rem 0; content-visibility: auto; contain-intrinsic-size: auto 22rem; }
article.selected-anchor { border: 2px solid #125d9c; content-visibility: visible; }
dl { display: grid; grid-template-columns: 11rem minmax(0, 1fr); gap: .4rem 1rem; }
dt { font-weight: 650; }
dd { margin: 0; min-width: 0; }
pre { margin: 0; font: inherit; }
code { font: inherit; }
.literal { white-space: pre-wrap; overflow-wrap: anywhere; unicode-bidi: plaintext; }
.document-id, .chunk-id { font-family: ui-monospace, monospace; font-size: .92rem; }
.passage-text { padding: 1rem; border-left: 4px solid #285e80; background: #f1f6fa; }
mark { background: #ffe29a; color: #182b3a; }
footer { border-top: 1px solid #c4d3df; font-size: .9rem; }
@media (max-width: 42rem) {
  header, main, footer { padding: 1rem; } form { grid-template-columns: minmax(0, 1fr); }
  dl { display: block; } dd { margin: .25rem 0 .75rem; }
}
@media print {
  body { background: white; } form, nav, .skip-link { display: none; }
  article { break-inside: avoid; content-visibility: visible; } a { color: inherit; }
}
"""
_STYLE_HASH = base64.b64encode(hashlib.sha256(_STYLE.encode()).digest()).decode("ascii")
_HEADERS = {
    "Content-Security-Policy": (
        f"default-src 'none'; script-src 'none'; style-src 'sha256-{_STYLE_HASH}'; "
        "base-uri 'none'; object-src 'none'; frame-src 'none'; frame-ancestors 'none'; "
        "form-action 'self'"
    ),
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
}
_FOOTER = (
    "</main><footer><h2>Read-only, not private by default</h2>"
    "<p>These pages inspect the current local corpus, not frozen run evidence or verified "
    "scientific findings. Separate pages are not one snapshot: edits can change passages, "
    "and new IDs behind a cursor need a fresh traversal.</p>"
    "<p>No generation, retrieval, reranking, ingestion, collection edits, or event writes "
    "occur during page reads. Normal app startup still initializes stores and loads retrieval "
    "indexes. No scripts, external resources, analytics, or browser storage are used.</p>"
    "<p>Keep the service on loopback or behind your own access controls. There is no built-in "
    "authentication. Queries, IDs, filters, titles and passages can be sensitive; "
    "URLs may remain in "
    "browser history and server logs, and no-store cannot prevent saving or screenshots. "
    "Source labels are untrusted text, never automatic external links.</p>"
    "<p>IDs are shown as JSON strings for lossless copying into a document_ids array. "
    "Quotes and escapes are part of that representation. Unicode visual order alone "
    "is not an identity check.</p></footer></body></html>"
)


class _PageTooLargeError(ValueError):
    pass


class _UnrepresentableTextError(ValueError):
    pass


def _escaped(text: str) -> str:
    if "\0" in text:
        raise _UnrepresentableTextError()
    return escape(text).replace("\r", "&#13;")


def _literal(text: str, css_class: str) -> str:
    # The code child preserves an initial newline; character references preserve CR.
    return (
        f'<pre class="literal {css_class}" dir="auto">'
        f'<code translate="no">{_escaped(text)}</code></pre>'
    )


def _identity(value: str, css_class: str) -> str:
    return _literal(json.dumps(value, ensure_ascii=True), css_class)


def _url(path: str, params: Mapping[str, str | int | None]) -> str:
    query = urlencode({key: value for key, value in params.items() if value is not None})
    return path + ("?" + query if query else "")


def _link(identifier: str, url: str, label: str) -> str:
    return f'<a id="{identifier}" href="{_escaped(url)}">{label}</a>'


def _page(title: str, fragments: Iterator[str], *, status: int = 200) -> HTMLResponse:
    prefix = (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta name="theme-color" content="#f4f7fa">'
        f"<title>{title} | Scholar RAG</title><style>{_STYLE}</style></head><body>"
        '<a class="skip-link" href="#main">Skip to main content</a>'
        '<header><p class="eyebrow" translate="no">SCHOLAR RAG / READ-ONLY / CURRENT CORPUS</p>'
        f"<h1>{title}</h1>"
        '<p class="muted">Discover stored papers. Inspect passages before choosing evidence.</p>'
        '</header><main id="main" tabindex="-1">'
    )
    content = bytearray()
    for fragment in chain((prefix,), fragments, (_FOOTER,)):
        encoded = fragment.encode("utf-8")
        if len(content) + len(encoded) > MAX_HTML_BYTES:
            raise _PageTooLargeError()
        content.extend(encoded)
    return HTMLResponse(bytes(content), status_code=status, headers=_HEADERS)


def _error(
    status: int,
    code: str,
    title: str,
    message: str,
    *,
    root_path: str = "",
    navigation: str | None = None,
) -> HTMLResponse:
    logger.warning("Local corpus explorer failed: %s", code)
    return _page(
        title,
        iter(
            (
                f'<p class="notice">HTTP {status}. {message}</p>',
                navigation
                or f'<p><a href="{_escaped(root_path + "/explore")}">'
                "Back to the catalog / reset filters</a></p>",
            )
        ),
        status=status,
    )


class _ExplorerRoute(APIRoute):
    def get_route_handler(self) -> Callable[[Request], Coroutine[None, None, Response]]:
        handler = super().get_route_handler()

        async def read(request: Request) -> Response:
            root_path = request.scope.get("root_path", "").rstrip("/")
            search_view = self.path == "/explore/search" or (
                self.path == "/explore/context"
                and any(key.startswith("search_") for key in request.query_params)
            )

            def error(status: int, code: str, title: str, message: str) -> HTMLResponse:
                navigation = None
                options = getattr(request.state, "explorer_search", None)
                if isinstance(options, _SearchParameters):
                    navigation = _search_navigation(
                        options,
                        root_path,
                        back_to_results=self.path == "/explore/context",
                        restart=True,
                    )
                elif search_view:
                    navigation = _search_navigation(_SearchParameters(), root_path)
                return _error(
                    status, code, title, message, root_path=root_path, navigation=navigation
                )

            try:
                if search_view:
                    try:
                        unquote_to_bytes(request.scope.get("query_string", b"")).decode("utf-8")
                    except UnicodeDecodeError as exc:
                        raise RequestValidationError([]) from exc
                    if len(request.query_params.multi_items()) != len(request.query_params):
                        raise RequestValidationError([])
                return await handler(request)
            except RequestValidationError:
                if search_view:
                    return error(
                        422,
                        "invalid_browser_search_request",
                        "Invalid passage search request",
                        "Enter a nonblank Unicode phrase (1-200 characters), limit 1-50, "
                        "and one scope: all, document, or collection. Only all permits a blank "
                        "scope ID; document IDs must be exact (1-128 characters, no surrounding "
                        "whitespace). Use a saved collection ID or a returned cursor. "
                        "Empty queries/cursors, duplicate or unknown fields and incomplete "
                        "search return parameters are invalid. No scope was widened.",
                    )
                if self.path == "/explore/context":
                    return error(
                        422,
                        "invalid_source_context_request",
                        "Invalid source context request",
                        "Use exact Unicode document and chunk IDs, before/after counts from "
                        "0 to 5, and unchanged catalog/passage navigation parameters. "
                        "Document IDs are 1-128 characters without surrounding whitespace; "
                        "chunk IDs are 1-256 characters.",
                    )
                return error(
                    422,
                    "invalid_explorer_request",
                    "Invalid explorer request",
                    "Use a limit from 1 to 100, source up to 512 characters, title up to 300, "
                    "and the returned cursor. Document IDs must be 1-128 characters without "
                    "surrounding whitespace. Only empty source/title fields are omitted.",
                )
            except ChunkCursorError:
                return error(
                    422,
                    "invalid_chunk_cursor",
                    "Invalid passage cursor",
                    "Use a continuation returned for this exact document, or start again "
                    "from the catalog. Cursors are not shared between documents.",
                )
            except DocumentNotFoundError:
                return error(
                    404,
                    "document_not_found",
                    "Document not found",
                    "This document is not in the current stored corpus. It may have been "
                    "removed since browsing. Return to the catalog to discover current IDs.",
                )
            except SourceContextError as exc:
                if exc.status_code == 404:
                    return error(
                        404,
                        exc.code,
                        "Passage not found",
                        "This exact chunk is not in this document in the current corpus. "
                        "It may have been replaced or removed. Rediscover stored passages; "
                        "the reader will not guess a replacement.",
                    )
                if exc.status_code == 503:
                    return error(
                        503,
                        exc.code,
                        "Corpus storage unavailable",
                        "The existing database could not be read. Check its availability "
                        "and permissions locally; this is not an empty context window.",
                    )
                return error(
                    409,
                    exc.code,
                    "Source context unavailable",
                    "Source order must be valid and unambiguous for every stored chunk in "
                    "this document. The reader allows at most 2048 chunks and 8192 stored "
                    "metadata bytes per chunk. No partial context or guessed order is shown. "
                    "Review the import with trusted local tools; browsing cannot repair it.",
                )
            except LiteralSearchError as exc:
                title, message = _SEARCH_ERRORS.get(
                    exc.code,
                    ("Passage search unavailable", "No complete search page could be read."),
                )
                return error(exc.status_code, exc.code, title, message)
            except CollectionError as exc:
                return error(
                    exc.status_code,
                    exc.code,
                    "Collection not found" if exc.status_code == 404 else "Collection unavailable",
                    "The saved collection is missing, invalid, or references missing documents. "
                    "No results or substitute scope are shown. Inspect its metadata through "
                    "the existing collections API before searching again.",
                )
            except (DocumentCatalogError, DocumentChunksError):
                return error(
                    409,
                    "invalid_corpus_record",
                    "Stored data cannot be displayed",
                    "A projected record or its lookahead is invalid. No partial page is shown. "
                    "Review the import using trusted local storage tools; "
                    "browsing cannot repair it.",
                )
            except sqlite3.Error:
                return error(
                    503,
                    "storage_unavailable",
                    "Corpus storage unavailable",
                    "The existing database could not be read. Check availability, permissions "
                    "and the configured database locally, then retry. This is not an empty corpus.",
                )
            except _PageTooLargeError:
                return error(
                    413,
                    "html_page_too_large",
                    "Page exceeds the HTML size limit",
                    "The complete page exceeds 1 MiB after HTML escaping. No partial page is "
                    "shown. Request a smaller limit or inspect the bounded JSON/Python readers.",
                )
            except (_UnrepresentableTextError, UnicodeEncodeError):
                return error(
                    409,
                    "html_text_not_representable",
                    "Text cannot be displayed faithfully",
                    "A stored label or passage cannot be represented faithfully as UTF-8 HTML "
                    "(for example, embedded NUL). Use the JSON/Python readers to inspect it.",
                )

        return read


def _empty_filter(value: object) -> object:
    return None if value == "" else value


_OptionalSource = Annotated[SourceFilter | None, BeforeValidator(_empty_filter)]
_OptionalTitle = Annotated[TitleFilter | None, BeforeValidator(_empty_filter)]
_Limit = Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)]
_NeighborCount = Annotated[int, Query(ge=0, le=MAX_CONTEXT_NEIGHBORS)]


def _query_integer(value: object) -> object:
    # GET forms send strings; do not accept float/boolean spellings as integer limits.
    if isinstance(value, str) and value.isascii() and value.isdecimal():
        return int(value)
    return value


_SearchLimit = Annotated[
    int, Field(strict=True, ge=1, le=MAX_SEARCH_LIMIT), BeforeValidator(_query_integer)
]
_CatalogLimit = Annotated[
    int, Field(strict=True, ge=1, le=MAX_PAGE_SIZE), BeforeValidator(_query_integer)
]
_SearchScope = Literal["all", "document", "collection"]
_ScopeId = Annotated[str, Field(strict=True, max_length=MAX_DOCUMENT_ID_LENGTH)]
_DOCUMENT_ID = TypeAdapter(ContextDocumentIdentity)
_COLLECTION_ID = TypeAdapter(CollectionId)


class _SearchParameters(BaseModel):
    """Native GET form state; only an omitted query opens the form without searching."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query: SearchQuery | None = None
    scope: _SearchScope = "all"
    scope_id: _ScopeId = ""
    limit: _SearchLimit = DEFAULT_SEARCH_LIMIT
    cursor: SearchCursor | None = None
    catalog_limit: _CatalogLimit = DEFAULT_PAGE_SIZE
    source: _OptionalSource = None
    title: _OptionalTitle = None
    catalog_cursor: ContextDocumentIdentity | None = None

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        if self.scope == "document":
            _DOCUMENT_ID.validate_python(self.scope_id)
        elif self.scope == "collection":
            _COLLECTION_ID.validate_python(self.scope_id)
        elif self.scope_id:
            raise ValueError("Whole-corpus search cannot discard a supplied scope ID.")
        if self.query is not None:
            self.search_request()
        elif self.cursor is not None:
            raise ValueError("A search cursor requires its original query.")
        return self

    def search_request(self) -> LiteralSearchRequest:
        payload = self.model_dump(include={"query", "limit", "cursor"}, exclude_none=True)
        if self.scope == "document":
            payload["document_ids"] = [self.scope_id]
        elif self.scope == "collection":
            payload["collection_id"] = self.scope_id
        return LiteralSearchRequest.model_validate(payload)

    def parameters(self) -> dict[str, str | int | None]:
        return {
            "query": self.query,
            "scope": self.scope,
            "scope_id": self.scope_id if self.scope != "all" else None,
            "limit": self.limit,
            "cursor": self.cursor,
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

    def context_parameters(self) -> dict[str, str | int | None]:
        return {
            "limit": self.catalog_limit,
            "source": self.source,
            "title": self.title,
            "catalog_cursor": self.catalog_cursor,
            "search_query": self.query,
            "search_scope": self.scope,
            "search_scope_id": self.scope_id if self.scope != "all" else None,
            "search_limit": self.limit,
            "search_cursor": self.cursor,
        }


_SEARCH_ERRORS = {
    "invalid_search_cursor": (
        "Invalid or stale search cursor",
        "Use the returned cursor with the same exact phrase and scope. A collection revision "
        "change, even a rename, requires starting again with the same selection. "
        "No scope was widened.",
    ),
    "search_text_read_limit": (
        "Passage exceeds the search read limit",
        "An encountered stored passage exceeds 4 MiB. No partial page is shown. Narrow the "
        "scope or inspect the import with trusted local tools; browsing cannot repair it.",
    ),
    "search_response_too_large": (
        "Search exceeds the response size limit",
        "The shared search result exceeds 262144 UTF-8 bytes. No partial page is shown. "
        "Request a smaller limit.",
    ),
    "search_storage_unavailable": (
        "Corpus storage unavailable",
        "The corpus or collection database could not be read. Check local availability and "
        "permissions before retrying. This is not an empty result.",
    ),
    "search_timeout": (
        "Passage search timed out",
        "The literal scan exceeded its five-second read deadline. Narrow the scope and "
        "try again; no partial results are shown.",
    ),
}
_RESPONSES: dict[int | str, dict[str, str]] = {
    404: {"description": "Document no longer exists"},
    409: {"description": "Invalid projected data or text not representable as HTML"},
    413: {"description": "Complete HTML page exceeds 1 MiB; no partial page"},
    422: {"description": "Invalid parameters or document-bound cursor"},
    503: {"description": "Current corpus storage unavailable"},
}
router = APIRouter(route_class=_ExplorerRoute, tags=["Local corpus explorer"])


def _filters(limit: int, source: str | None, title: str | None, root_path: str) -> Iterator[str]:
    catalog_path = _escaped(root_path + "/explore")
    yield '<section aria-labelledby="filters"><h2 id="filters">Filter stored papers</h2>'
    if any(character in (source or "") + (title or "") for character in "\0\r\n"):
        yield (
            "<p>These filters contain control characters that native text inputs cannot preserve. "
            "Page links keep them exactly; reset filters to edit them.</p>"
            "<dl><dt>Exact source (JSON)</dt><dd>"
            + _literal(json.dumps(source), "filter-value")
            + "</dd><dt>Literal title (JSON)</dt><dd>"
            + _literal(json.dumps(title), "filter-value")
            + f'</dd></dl><a id="reset-filters" href="{catalog_path}">Reset filters</a></section>'
        )
        return
    yield f'<form method="get" action="{catalog_path}" autocomplete="off">'
    # Native maxlength counts UTF-16 units, unlike the readers' Unicode codepoint bounds.
    for name, label, value, help_text in (
        (
            "source",
            "Exact source",
            source,
            "Case-sensitive exact match, not a URL to open. Blank means any source.",
        ),
        (
            "title",
            "Title contains",
            title,
            "Literal substring; ASCII case-insensitive. %, _ and spaces stay literal.",
        ),
    ):
        yield (
            f'<div><label for="{name}">{label}</label>'
            f'<input id="{name}" name="{name}" type="text" '
            f'value="{_escaped(value or "")}" aria-describedby="{name}-help" '
            'autocomplete="off" autocapitalize="none" spellcheck="false" translate="no">'
            f'<p id="{name}-help">{help_text}</p></div>'
        )
    yield (
        '<div><label for="limit">Page size</label>'
        f'<input id="limit" name="limit" type="number" min="1" max="{MAX_PAGE_SIZE}" '
        f'value="{limit}" required inputmode="numeric" autocomplete="off" '
        'aria-describedby="limit-help">'
        '<p id="limit-help">1-100 papers or passages per page.</p></div>'
        '<div class="actions"><button type="submit">Apply filters</button>'
        f'<a id="reset-filters" href="{catalog_path}">Reset filters</a></div></form></section>'
    )


def _truncation(label: str, truncated: bool, limit: int) -> str:
    return (
        f'<p class="notice">{label} truncated: only the first {limit} characters are shown.</p>'
        if truncated
        else ""
    )


def _pagination(first: str, next_url: str | None) -> str:
    links = _link("first-page", first, "Start again with the same selection")
    if next_url is not None:
        links += (
            f'<a class="next-page" id="next-page" rel="next" href="{_escaped(next_url)}">'
            "Next page</a>"
        )
    else:
        links += "<span>End of this result set.</span>"
    return f'<nav aria-label="Pagination">{links}</nav>'


def _search_navigation(
    options: _SearchParameters,
    root_path: str,
    *,
    back_to_results: bool = False,
    restart: bool = False,
) -> str:
    search_path = root_path + "/explore/search"
    params = options.parameters()
    links = _link(
        "back-to-catalog",
        _url(root_path + "/explore", options.catalog_parameters()),
        "Back to filtered catalog",
    )
    if back_to_results and options.query is not None:
        links += _link("back-to-search", _url(search_path, params), "Back to search results")
    if restart and options.query is not None:
        links += _link(
            "first-page",
            _url(search_path, {**params, "cursor": None}),
            "Start again with the same selection",
        )
    links += _link(
        "reset-search",
        _url(search_path, {**params, "query": None, "cursor": None}),
        "Clear phrase / keep scope",
    )
    if options.scope != "all":
        links += _link(
            "reset-scope",
            _url(
                search_path,
                {**params, "query": None, "cursor": None, "scope": "all", "scope_id": None},
            ),
            "New whole-corpus search",
        )
    return f'<nav aria-label="Search navigation">{links}</nav>'


def _search_form(options: _SearchParameters, root_path: str) -> Iterator[str]:
    yield '<section aria-labelledby="phrase-form"><h2 id="phrase-form">Find exact wording</h2>'
    if any(
        isinstance(value, str) and any(character in value for character in "\0\r\n")
        for value in options.parameters().values()
    ):
        yield (
            '<p class="notice">This selection contains control characters that native text '
            "inputs cannot preserve. Exact pagination, source-context and return links still "
            "work. Clear the phrase or start a new whole-corpus search to edit safely; "
            "control-bearing catalog filters must be reset in the catalog.</p></section>"
        )
        return
    yield (
        '<form id="passage-search" method="get" '
        f'action="{_escaped(root_path + "/explore/search")}" autocomplete="off">'
        '<div><label for="query">Literal phrase</label>'
        f'<input id="query" name="query" type="text" value="{_escaped(options.query or "")}" '
        'required autocomplete="off" autocapitalize="none" spellcheck="false" translate="no" '
        'aria-describedby="query-help">'
        '<p id="query-help">1-200 Unicode code points; case, accents, punctuation and '
        "nonblank surrounding spaces stay exact. No wildcards or semantic search.</p></div>"
        '<div><label for="scope">Search within</label>'
        '<select id="scope" name="scope" aria-describedby="scope-help">'
    )
    for value, label in (
        ("all", "Whole current corpus"),
        ("document", "One exact paper"),
        ("collection", "One saved collection"),
    ):
        selected = " selected" if options.scope == value else ""
        yield f'<option value="{value}"{selected}>{label}</option>'
    yield (
        '</select><p id="scope-help">Choose only one scope. Catalog filters are return '
        "navigation, not search filters.</p></div>"
        '<div><label for="limit">Matches per page</label>'
        f'<input id="limit" name="limit" type="number" min="1" max="{MAX_SEARCH_LIMIT}" '
        f'value="{options.limit}" required inputmode="numeric" aria-describedby="limit-help">'
        '<p id="limit-help">1-50 matches; ordered by IDs, not relevance.</p></div>'
        '<div><label for="scope_id">Exact paper or collection ID</label>'
        f'<input id="scope_id" name="scope_id" type="text" value="{_escaped(options.scope_id)}" '
        'autocomplete="off" autocapitalize="none" spellcheck="false" translate="no" '
        'aria-describedby="scope-id-help">'
        '<p id="scope-id-help">Required for a paper or collection; leave blank only for '
        "whole corpus. Paste the raw ID, not its JSON quotes. Empty or invalid selections "
        "never become whole-corpus searches.</p></div>"
    )
    for name in ("catalog_limit", "source", "title", "catalog_cursor"):
        navigation_value = options.parameters()[name]
        if navigation_value is not None:
            yield f'<input type="hidden" name="{name}" value="{_escaped(str(navigation_value))}">'
    yield (
        '<div class="actions"><button type="submit">Search passages</button></div></form>'
        "<p>Submitting starts at the first page, without a previous cursor.</p></section>"
    )


def _highlight(match: LiteralMatch) -> str:
    start = match.match_start - match.excerpt_start
    end = match.match_end - match.excerpt_start
    return (
        '<pre class="literal passage-text search-excerpt" dir="auto"><code translate="no">'
        + _escaped(match.excerpt[:start])
        + "<mark>"
        + _escaped(match.excerpt[start:end])
        + "</mark>"
        + _escaped(match.excerpt[end:])
        + "</code></pre>"
    )


def _search(
    page: LiteralSearchPage | None, options: _SearchParameters, root_path: str
) -> Iterator[str]:
    yield _search_navigation(options, root_path)
    yield (
        '<p class="notice">Read-only literal search of the CURRENT CORPUS, not frozen evidence '
        "or a scientific-support judgment. GET phrases, scopes and cursors appear in URLs, "
        "browser history and potentially access logs. Do not enter secrets.</p>"
    )
    yield from _search_form(options, root_path)
    yield "<dl><dt>Search scope</dt><dd>"
    yield {
        "all": "Whole current corpus",
        "document": "One exact paper",
        "collection": "One saved collection",
    }[options.scope]
    yield "</dd>"
    if options.scope != "all":
        yield "<dt>Scope ID (JSON)</dt><dd>" + _identity(options.scope_id, "scope-id") + "</dd>"
    if options.query is not None:
        yield "<dt>Phrase (JSON)</dt><dd>" + _identity(options.query, "search-query") + "</dd>"
    yield "</dl>"
    if page is None:
        yield "<p>Enter a phrase to search. No corpus scan has run on this form-only page.</p>"
        return
    if page.collection_id is not None:
        yield (
            f'<p class="collection-revision">Resolved collection revision: '
            f"{page.collection_revision}. Membership and passages share one read snapshot.</p>"
            "<details><summary>Exact resolved paper IDs (JSON)</summary>"
        )
        for identifier in page.document_ids or ():
            yield _identity(identifier, "scope-document-id")
        yield "</details>"
    yield (
        '<section aria-labelledby="search-results"><h2 id="search-results">Matching passages</h2>'
        f"<p>{len(page.matches)} matching chunks on this page (limit {options.limit}). "
        "Ascending SQLite BINARY document/chunk ID order, not relevance. "
        "Only the first occurrence in each chunk is highlighted. Match and excerpt offsets "
        "are zero-based, half-open Unicode code-point positions in the full stored chunk; "
        "not bytes, UTF-16 units, grapheme clusters or document-wide positions.</p>"
    )
    if not page.matches:
        message = (
            "No matching passages remain after this cursor. Start again to inspect current results."
            if options.cursor is not None
            else "No matching passages in this exact scope. A paper may be absent or have no "
            "chunks; an empty corpus also has no matches. No scope was widened. "
            "This does not mean the literature lacks evidence."
        )
        yield f'<p class="notice">{message}</p>'
    for number, match in enumerate(page.matches, 1):
        yield (
            f'<article aria-labelledby="match-{number}"><h3 id="match-{number}">Match {number}</h3>'
        )
        yield "<dl><dt>Document ID (JSON)</dt><dd>"
        yield _identity(match.document_id, "document-id")
        yield "</dd><dt>Chunk ID (JSON)</dt><dd>" + _identity(match.chunk_id, "chunk-id") + "</dd>"
        yield "</dl><dl><dt>Title</dt><dd>" + _literal(match.title, "chunk-title") + "</dd>"
        yield "<dt>Source</dt><dd>" + _literal(match.source, "chunk-source") + "</dd></dl>"
        yield _truncation("Title", match.title_truncated, MAX_TITLE_CHARACTERS)
        yield _truncation("Source", match.source_truncated, MAX_SOURCE_CHARACTERS)
        yield (
            f'<p class="match-offsets" data-start="{match.match_start}" '
            f'data-end="{match.match_end}">First match '
            f"[{match.match_start}, {match.match_end}).</p>"
            f'<p class="excerpt-offsets" data-start="{match.excerpt_start}" '
            f'data-end="{match.excerpt_end}">Excerpt [{match.excerpt_start}, {match.excerpt_end}); '
            f"full chunk: {match.text_characters} Unicode characters.</p>"
        )
        yield _highlight(match)
        if match.excerpt_truncated_before:
            yield '<p class="notice">Earlier chunk text is omitted from this excerpt.</p>'
        if match.excerpt_truncated_after:
            yield '<p class="notice">Later chunk text is omitted from this excerpt.</p>'
        yield (
            "<p>"
            + _link(
                f"source-context-{number}",
                _url(
                    root_path + "/explore/context",
                    {
                        "document_id": match.document_id,
                        "chunk_id": match.chunk_id,
                        **options.context_parameters(),
                    },
                ),
                "Read surrounding source context",
            )
            + "</p></article>"
        )
    yield (
        "<p>Source context reads this exact document/chunk pair from the current corpus again. "
        "It requires valid source ordering and may show only a long passage's prefix, not "
        "this match's location. Corpus edits can change or remove the passage.</p>"
    )
    params = options.parameters()
    yield _pagination(
        _url(root_path + "/explore/search", {**params, "cursor": None}),
        _url(root_path + "/explore/search", {**params, "cursor": page.next_cursor})
        if page.next_cursor
        else None,
    )
    yield "</section>"


def _passage(chunk: StoredChunk, *, continuation: str) -> Iterator[str]:
    yield "<dl><dt>Chunk ID (JSON)</dt><dd>"
    yield _identity(chunk.chunk_id, "chunk-id")
    yield '</dd><dt>Stored chunk_index</dt><dd class="chunk-index">'
    yield str(chunk.chunk_index) if chunk.chunk_index is not None else "Not recorded"
    yield "</dd><dt>Title</dt><dd>" + _literal(chunk.title, "chunk-title") + "</dd>"
    yield "</dl>" + _truncation("Title", chunk.title_truncated, MAX_TITLE_CHARACTERS)
    yield "<dl><dt>Source</dt><dd>" + _literal(chunk.source, "chunk-source") + "</dd></dl>"
    yield _truncation("Source", chunk.source_truncated, MAX_SOURCE_CHARACTERS)
    yield _literal(chunk.text, "passage-text")
    if not chunk.text:
        yield "<p>This stored passage is empty.</p>"
    if chunk.text_truncated:
        yield (
            f'<p class="notice">Passage truncated: only the first {MAX_TEXT_CHARACTERS} '
            f"characters are shown. {continuation}, not the rest of this "
            "passage. Consult the original trusted source for the omitted text.</p>"
        )


def _catalog(
    page: DocumentCatalogPage,
    *,
    limit: int,
    cursor: str | None,
    source: str | None,
    title: str | None,
    root_path: str,
) -> Iterator[str]:
    search_params: dict[str, str | int | None] = {
        "catalog_limit": limit,
        "source": source,
        "title": title,
        "catalog_cursor": cursor,
    }
    yield '<nav aria-label="Corpus tools">'
    yield _link(
        "search-passages",
        _url(root_path + "/explore/search", search_params),
        "Search passages",
    )
    yield "</nav>"
    yield from _filters(limit, source, title, root_path)
    yield (
        f'<section aria-labelledby="papers"><h2 id="papers">Stored papers</h2>'
        f"<p>{len(page.documents)} papers on this page (limit {limit}). "
        "Ordered by ascending document ID using SQLite BINARY order, not creation time. "
        "Choose a title to inspect its stored passages.</p>"
    )
    if not page.documents:
        message = "The local corpus is empty. Ingest permitted material through the existing API."
        if cursor is not None:
            message = "No papers after this cursor. Start again to see current results."
        elif source is not None or title is not None:
            message = "No papers match these filters. Adjust them or reset the selection."
        yield f'<p class="notice">{message}</p>'
    for number, document in enumerate(page.documents, 1):
        url = _url(
            root_path + "/explore/document",
            {
                "document_id": document.document_id,
                "limit": limit,
                "source": source,
                "title": title,
                "catalog_cursor": cursor,
            },
        )
        yield f'<article><h3><a id="inspect-document-{number}" href="{_escaped(url)}">'
        yield (
            '<span class="literal document-title" dir="auto">'
            + _escaped(document.title)
            + "</span>"
            if document.title.strip()
            else "Untitled paper"
        )
        yield "</a></h3>"
        yield _truncation("Title", document.title_truncated, MAX_TITLE_CHARACTERS)
        yield "<dl><dt>Document ID (JSON)</dt><dd>"
        yield _identity(document.document_id, "document-id")
        yield "</dd><dt>Source</dt><dd>" + _literal(document.source, "document-source") + "</dd>"
        yield f"<dt>Stored chunks</dt><dd>{document.chunk_count}</dd></dl>"
        yield _truncation("Source", document.source_truncated, MAX_SOURCE_CHARACTERS)
        yield _link(
            f"search-document-{number}",
            _url(
                root_path + "/explore/search",
                {**search_params, "scope": "document", "scope_id": document.document_id},
            ),
            "Search this paper",
        )
        yield "</article>"
    params: dict[str, str | int | None] = {"limit": limit, "source": source, "title": title}
    yield _pagination(
        _url(root_path + "/explore", params),
        _url(root_path + "/explore", {**params, "cursor": page.next_cursor})
        if page.next_cursor
        else None,
    )
    yield "</section>"


def _document(
    page: DocumentChunksPage,
    *,
    limit: int,
    cursor: str | None,
    source: str | None,
    title: str | None,
    catalog_cursor: str | None,
    root_path: str,
) -> Iterator[str]:
    back = _url(
        root_path + "/explore",
        {
            "limit": limit,
            "source": source,
            "title": title,
            "cursor": catalog_cursor,
        },
    )
    yield '<nav aria-label="Corpus navigation">'
    yield _link("back-to-catalog", back, "Back to filtered catalog")
    yield f'<a href="{_escaped(root_path + "/explore")}">Reset to all papers</a></nav>'
    yield "<dl><dt>Document ID (JSON)</dt><dd>"
    yield _identity(page.document_id, "document-id")
    yield "</dd></dl>"
    yield (
        f'<section aria-labelledby="passages"><h2 id="passages">Current stored passages</h2>'
        f"<p>{len(page.chunks)} passages on this page (limit {limit}). "
        "Ordered by ascending chunk ID using SQLite BINARY order, not chunk_index, "
        "paragraph position or relevance. Labels below belong to each stored chunk.</p>"
    )
    if not page.chunks:
        message = (
            "No passages remain after this cursor. Start again to inspect current chunks."
            if cursor is not None
            else "This document has no stored passages."
        )
        yield f'<p class="notice">{message}</p>'
    params: dict[str, str | int | None] = {
        "document_id": page.document_id,
        "limit": limit,
        "source": source,
        "title": title,
        "catalog_cursor": catalog_cursor,
    }
    for number, chunk in enumerate(page.chunks, 1):
        yield f"<article><h3>Passage {number} on this page</h3>"
        yield from _passage(chunk, continuation="Next page moves to later chunk IDs")
        yield (
            "<p>"
            + _link(
                f"source-context-{number}",
                _url(
                    root_path + "/explore/context",
                    {**params, "cursor": cursor, "chunk_id": chunk.chunk_id},
                ),
                "Read surrounding source context",
            )
            + "</p></article>"
        )
    yield _pagination(
        _url(root_path + "/explore/document", params),
        _url(root_path + "/explore/document", {**params, "cursor": page.next_cursor})
        if page.next_cursor
        else None,
    )
    yield "</section>"


def _context_window(
    page: SourceContext, params: Mapping[str, str | int | None], root_path: str
) -> Iterator[str]:
    yield '<section aria-labelledby="window"><h2 id="window">Choose surrounding passages</h2>'
    controls = {**params, "chunk_id": page.anchor_chunk_id}
    if any(
        isinstance(value, str) and any(character in value for character in "\0\r\n")
        for value in controls.values()
    ):
        yield (
            "<p>This selection contains control characters that a native form cannot "
            "preserve. Use the exact window-size and passage links instead.</p>"
        )
    else:
        yield (
            '<form id="context-window" method="get" '
            f'action="{_escaped(root_path + "/explore/context")}" autocomplete="off">'
        )
        for name, value in controls.items():
            if value is not None:
                yield f'<input type="hidden" name="{name}" value="{_escaped(str(value))}">'
        for name, count in (("before", page.before), ("after", page.after)):
            yield (
                f'<div><label for="{name}">Passages {name} the selection</label>'
                f'<input id="{name}" name="{name}" type="number" min="0" '
                f'max="{MAX_CONTEXT_NEIGHBORS}" value="{count}" required inputmode="numeric" '
                'aria-describedby="window-help"></div>'
            )
        yield (
            '<p id="window-help">0-5 stored neighbors on each side; the selected passage '
            'is always included.</p><div class="actions">'
            '<button type="submit">Read window</button></div></form>'
        )
    yield '<nav aria-label="Window shortcuts">'
    for identifier, count, label in (
        ("anchor-only", 0, "Selected passage only"),
        ("default-window", DEFAULT_CONTEXT_NEIGHBORS, "Two on each side"),
        ("maximum-window", MAX_CONTEXT_NEIGHBORS, "Five on each side"),
    ):
        yield _link(
            identifier,
            _url(root_path + "/explore/context", {**controls, "before": count, "after": count}),
            label,
        )
    yield "</nav></section>"


def _context(
    page: SourceContext,
    *,
    limit: int,
    cursor: str | None,
    source: str | None,
    title: str | None,
    catalog_cursor: str | None,
    root_path: str,
    search: _SearchParameters | None = None,
) -> Iterator[str]:
    params: dict[str, str | int | None] = {
        "document_id": page.document_id,
        "limit": limit,
        "source": source,
        "title": title,
        "catalog_cursor": catalog_cursor,
        "cursor": cursor,
    }
    if search is not None:
        params.update(search.context_parameters())
        yield _search_navigation(search, root_path, back_to_results=True)
    else:
        yield '<nav aria-label="Corpus navigation">'
        yield _link(
            "back-to-passages",
            _url(root_path + "/explore/document", params),
            "Back to passage page",
        )
        yield _link(
            "back-to-catalog",
            _url(
                root_path + "/explore",
                {"limit": limit, "source": source, "title": title, "cursor": catalog_cursor},
            ),
            "Back to filtered catalog",
        )
        yield "</nav>"
    yield "<dl><dt>Document ID (JSON)</dt><dd>"
    yield _identity(page.document_id, "document-id")
    yield "</dd></dl>"
    yield (
        '<p class="notice">CURRENT CORPUS: this is not frozen saved-run evidence. '
        "Only unique, validated ascending chunk_index values define source order. "
        "Gaps are allowed; neighbors are stored chunks, not sentences or missing paragraphs. "
        "Overlaps are preserved, not merged. Each window is one read snapshot.</p>"
    )
    yield from _context_window(page, params, root_path)
    yield (
        '<section aria-labelledby="context"><h2 id="context">Surrounding source context</h2>'
        f"<p>Returned {page.returned_before} before + selected passage + "
        f"{page.returned_after} after (requested {page.before} / {page.after}).</p>"
    )
    yield (
        "<p>More stored passages exist before this window.</p>"
        if page.has_more_before
        else "<p>Start of the stored source order.</p>"
    )
    for number, chunk in enumerate(page.chunks, 1):
        offset = number - page.returned_before - 1
        label = (
            "Selected passage"
            if chunk.is_anchor
            else f"{abs(offset)} passage(s) {'before' if offset < 0 else 'after'} the selection"
        )
        anchor = ' id="context-anchor" class="selected-anchor"' if chunk.is_anchor else ""
        yield f'<article{anchor} aria-labelledby="context-passage-{number}">'
        yield f'<h3 id="context-passage-{number}">{label}</h3>'
        yield from _passage(chunk, continuation="Changing the window moves between stored chunks")
        yield (
            "<p>"
            + _link(
                f"center-context-{number}",
                _url(
                    root_path + "/explore/context",
                    {
                        **params,
                        "chunk_id": chunk.chunk_id,
                        "before": page.before,
                        "after": page.after,
                    },
                ),
                "Read around this passage",
            )
            + "</p></article>"
        )
    yield (
        "<p>More stored passages exist after this window.</p>"
        if page.has_more_after
        else "<p>End of the stored source order.</p>"
    )
    yield "</section>"


@router.get("/explore", response_class=HTMLResponse, responses=_RESPONSES)
def explore(
    request: Request,
    limit: _Limit = DEFAULT_PAGE_SIZE,
    cursor: Annotated[Identity | None, Query()] = None,
    source: Annotated[_OptionalSource, Query()] = None,
    title: Annotated[_OptionalTitle, Query()] = None,
) -> HTMLResponse:
    """Browse current paper summaries; only blank source/title form fields mean omission."""
    container: AppContainer = request.app.state.container
    page = container.document_catalog.list_documents(
        limit=limit,
        cursor=cursor,
        source=source,
        title=title,
    )
    return _page(
        "Local corpus explorer",
        _catalog(
            page,
            limit=limit,
            cursor=cursor,
            source=source,
            title=title,
            root_path=request.scope.get("root_path", "").rstrip("/"),
        ),
    )


@router.get("/explore/document", response_class=HTMLResponse, responses=_RESPONSES)
def explore_document(
    request: Request,
    document_id: Annotated[Identity, Query()],
    limit: _Limit = DEFAULT_PAGE_SIZE,
    cursor: Annotated[ChunkCursor | None, Query()] = None,
    source: Annotated[_OptionalSource, Query()] = None,
    title: Annotated[_OptionalTitle, Query()] = None,
    catalog_cursor: Annotated[Identity | None, Query()] = None,
) -> HTMLResponse:
    """Inspect exactly one document; query encoding keeps slashes and dot segments intact."""
    container: AppContainer = request.app.state.container
    page = container.document_chunks.list_chunks(document_id, limit=limit, cursor=cursor)
    return _page(
        "Stored passages",
        _document(
            page,
            limit=limit,
            cursor=cursor,
            source=source,
            title=title,
            catalog_cursor=catalog_cursor,
            root_path=request.scope.get("root_path", "").rstrip("/"),
        ),
    )


@router.get(
    "/explore/search",
    response_class=HTMLResponse,
    responses={
        **_RESPONSES,
        404: {"description": "Saved collection not found"},
        409: {"description": "Invalid stored passage, collection or HTML text"},
        413: {"description": "Shared search read/response limit or escaped HTML limit exceeded"},
        422: {"description": "Invalid search form or query/scope/revision-bound cursor"},
        504: {"description": "Literal scan deadline exceeded"},
    },
)
def explore_search(
    request: Request, options: Annotated[_SearchParameters, Query()]
) -> HTMLResponse:
    """Search current chunks using the API/Python literal reader; GET queries are not private."""
    request.state.explorer_search = options
    container: AppContainer = request.app.state.container
    page = (
        container.literal_search.search(options.search_request())
        if options.query is not None
        else None
    )
    return _page(
        "Literal passage search",
        _search(page, options, request.scope.get("root_path", "").rstrip("/")),
    )


@router.get("/explore/context", response_class=HTMLResponse, responses=_RESPONSES)
def explore_context(
    request: Request,
    document_id: Annotated[ContextDocumentIdentity, Query()],
    chunk_id: Annotated[ContextChunkIdentity, Query()],
    before: _NeighborCount = DEFAULT_CONTEXT_NEIGHBORS,
    after: _NeighborCount = DEFAULT_CONTEXT_NEIGHBORS,
    limit: _Limit = DEFAULT_PAGE_SIZE,
    cursor: Annotated[ChunkCursor | None, Query()] = None,
    source: Annotated[_OptionalSource, Query()] = None,
    title: Annotated[_OptionalTitle, Query()] = None,
    catalog_cursor: Annotated[Identity | None, Query()] = None,
    search_query: Annotated[SearchQuery | None, Query()] = None,
    search_scope: Annotated[_SearchScope | None, Query()] = None,
    search_scope_id: Annotated[_ScopeId | None, Query()] = None,
    search_limit: Annotated[_SearchLimit | None, Query()] = None,
    search_cursor: Annotated[SearchCursor | None, Query()] = None,
) -> HTMLResponse:
    """Read exact source context, preserving native navigation back to the original page."""
    search = None
    if any(key.startswith("search_") for key in request.query_params):
        allowed = {
            "document_id",
            "chunk_id",
            "before",
            "after",
            "limit",
            "cursor",
            "source",
            "title",
            "catalog_cursor",
            "search_query",
            "search_scope",
            "search_scope_id",
            "search_limit",
            "search_cursor",
        }
        if (
            set(request.query_params) - allowed
            or search_query is None
            or search_scope is None
            or search_limit is None
        ):
            raise RequestValidationError([])
        try:
            search = _SearchParameters(
                query=search_query,
                scope=search_scope,
                scope_id=search_scope_id if search_scope_id is not None else "",
                limit=search_limit,
                cursor=search_cursor,
                catalog_limit=limit,
                source=source,
                title=title,
                catalog_cursor=catalog_cursor,
            )
        except ValidationError as exc:
            raise RequestValidationError(exc.errors()) from exc
        request.state.explorer_search = search
    container: AppContainer = request.app.state.container
    page = container.source_context.read(document_id, chunk_id, before=before, after=after)
    return _page(
        "Source context reader",
        _context(
            page,
            limit=limit,
            cursor=cursor,
            source=source,
            title=title,
            catalog_cursor=catalog_cursor,
            root_path=request.scope.get("root_path", "").rstrip("/"),
            search=search,
        ),
    )
