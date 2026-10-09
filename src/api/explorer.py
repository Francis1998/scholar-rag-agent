"""Script-free, bounded HTML views of the existing read-only corpus readers."""

import base64
import hashlib
import json
import logging
import sqlite3
from collections.abc import Callable, Coroutine, Iterator, Mapping
from html import escape
from itertools import chain
from typing import Annotated
from urllib.parse import urlencode

from fastapi import APIRouter, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse
from fastapi.routing import APIRoute
from pydantic import BeforeValidator

from api.dependencies import AppContainer
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
a, button, input { touch-action: manipulation; -webkit-tap-highlight-color: #c7e1f4; }
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
input { width: 100%; min-height: 2.75rem; font: inherit; border: 1px solid #647d91;
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
    "authentication. IDs, filters, titles and passages can be sensitive; URLs may remain in "
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


def _error(status: int, code: str, title: str, message: str) -> HTMLResponse:
    logger.warning("Local corpus explorer failed: %s", code)
    return _page(
        title,
        iter(
            (
                f'<p class="notice">HTTP {status}. {message}</p>',
                '<p><a href="/explore">Back to the catalog / reset filters</a></p>',
            )
        ),
        status=status,
    )


class _ExplorerRoute(APIRoute):
    def get_route_handler(self) -> Callable[[Request], Coroutine[None, None, Response]]:
        handler = super().get_route_handler()

        async def read(request: Request) -> Response:
            try:
                return await handler(request)
            except RequestValidationError:
                if request.url.path == "/explore/context":
                    return _error(
                        422,
                        "invalid_source_context_request",
                        "Invalid source context request",
                        "Use exact Unicode document and chunk IDs, before/after counts from "
                        "0 to 5, and unchanged catalog/passage navigation parameters. "
                        "Document IDs are 1-128 characters without surrounding whitespace; "
                        "chunk IDs are 1-256 characters.",
                    )
                return _error(
                    422,
                    "invalid_explorer_request",
                    "Invalid explorer request",
                    "Use a limit from 1 to 100, source up to 512 characters, title up to 300, "
                    "and the returned cursor. Document IDs must be 1-128 characters without "
                    "surrounding whitespace. Only empty source/title fields are omitted.",
                )
            except ChunkCursorError:
                return _error(
                    422,
                    "invalid_chunk_cursor",
                    "Invalid passage cursor",
                    "Use a continuation returned for this exact document, or start again "
                    "from the catalog. Cursors are not shared between documents.",
                )
            except DocumentNotFoundError:
                return _error(
                    404,
                    "document_not_found",
                    "Document not found",
                    "This document is not in the current stored corpus. It may have been "
                    "removed since browsing. Return to the catalog to discover current IDs.",
                )
            except SourceContextError as exc:
                if exc.status_code == 404:
                    return _error(
                        404,
                        exc.code,
                        "Passage not found",
                        "This exact chunk is not in this document in the current corpus. "
                        "It may have been replaced or removed. Rediscover stored passages; "
                        "the reader will not guess a replacement.",
                    )
                if exc.status_code == 503:
                    return _error(
                        503,
                        exc.code,
                        "Corpus storage unavailable",
                        "The existing database could not be read. Check its availability "
                        "and permissions locally; this is not an empty context window.",
                    )
                return _error(
                    409,
                    exc.code,
                    "Source context unavailable",
                    "Source order must be valid and unambiguous for every stored chunk in "
                    "this document. The reader allows at most 2048 chunks and 8192 stored "
                    "metadata bytes per chunk. No partial context or guessed order is shown. "
                    "Review the import with trusted local tools; browsing cannot repair it.",
                )
            except (DocumentCatalogError, DocumentChunksError):
                return _error(
                    409,
                    "invalid_corpus_record",
                    "Stored data cannot be displayed",
                    "A projected record or its lookahead is invalid. No partial page is shown. "
                    "Review the import using trusted local storage tools; "
                    "browsing cannot repair it.",
                )
            except sqlite3.Error:
                return _error(
                    503,
                    "storage_unavailable",
                    "Corpus storage unavailable",
                    "The existing database could not be read. Check availability, permissions "
                    "and the configured database locally, then retry. This is not an empty corpus.",
                )
            except _PageTooLargeError:
                return _error(
                    413,
                    "html_page_too_large",
                    "Page exceeds the HTML size limit",
                    "The complete page exceeds 1 MiB after HTML escaping. No partial page is "
                    "shown. Request a smaller limit or inspect the bounded JSON/Python readers.",
                )
            except (_UnrepresentableTextError, UnicodeEncodeError):
                return _error(
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
_RESPONSES: dict[int | str, dict[str, str]] = {
    404: {"description": "Document no longer exists"},
    409: {"description": "Invalid projected data or text not representable as HTML"},
    413: {"description": "Complete HTML page exceeds 1 MiB; no partial page"},
    422: {"description": "Invalid parameters or document-bound cursor"},
    503: {"description": "Current corpus storage unavailable"},
}
router = APIRouter(route_class=_ExplorerRoute, tags=["Local corpus explorer"])


def _filters(limit: int, source: str | None, title: str | None) -> Iterator[str]:
    yield '<section aria-labelledby="filters"><h2 id="filters">Filter stored papers</h2>'
    if any(character in (source or "") + (title or "") for character in "\0\r\n"):
        yield (
            "<p>These filters contain control characters that native text inputs cannot preserve. "
            "Page links keep them exactly; reset filters to edit them.</p>"
            "<dl><dt>Exact source (JSON)</dt><dd>"
            + _literal(json.dumps(source), "filter-value")
            + "</dd><dt>Literal title (JSON)</dt><dd>"
            + _literal(json.dumps(title), "filter-value")
            + '</dd></dl><a id="reset-filters" href="/explore">Reset filters</a></section>'
        )
        return
    yield '<form method="get" action="/explore" autocomplete="off">'
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
        '<a id="reset-filters" href="/explore">Reset filters</a></div></form></section>'
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
) -> Iterator[str]:
    yield from _filters(limit, source, title)
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
            "/explore/document",
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
        yield "</article>"
    params: dict[str, str | int | None] = {"limit": limit, "source": source, "title": title}
    yield _pagination(
        _url("/explore", params),
        _url("/explore", {**params, "cursor": page.next_cursor}) if page.next_cursor else None,
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
) -> Iterator[str]:
    back = _url(
        "/explore",
        {
            "limit": limit,
            "source": source,
            "title": title,
            "cursor": catalog_cursor,
        },
    )
    yield '<nav aria-label="Corpus navigation">'
    yield _link("back-to-catalog", back, "Back to filtered catalog")
    yield '<a href="/explore">Reset to all papers</a></nav>'
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
                    "/explore/context",
                    {**params, "cursor": cursor, "chunk_id": chunk.chunk_id},
                ),
                "Read surrounding source context",
            )
            + "</p></article>"
        )
    yield _pagination(
        _url("/explore/document", params),
        _url("/explore/document", {**params, "cursor": page.next_cursor})
        if page.next_cursor
        else None,
    )
    yield "</section>"


def _context_window(page: SourceContext, params: Mapping[str, str | int | None]) -> Iterator[str]:
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
        yield '<form id="context-window" method="get" action="/explore/context" autocomplete="off">'
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
            _url("/explore/context", {**controls, "before": count, "after": count}),
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
) -> Iterator[str]:
    params: dict[str, str | int | None] = {
        "document_id": page.document_id,
        "limit": limit,
        "source": source,
        "title": title,
        "catalog_cursor": catalog_cursor,
        "cursor": cursor,
    }
    yield '<nav aria-label="Corpus navigation">'
    yield _link("back-to-passages", _url("/explore/document", params), "Back to passage page")
    yield _link(
        "back-to-catalog",
        _url(
            "/explore",
            {"limit": limit, "source": source, "title": title, "cursor": catalog_cursor},
        ),
        "Back to filtered catalog",
    )
    yield "</nav><dl><dt>Document ID (JSON)</dt><dd>"
    yield _identity(page.document_id, "document-id")
    yield "</dd></dl>"
    yield (
        '<p class="notice">CURRENT CORPUS: this is not frozen saved-run evidence. '
        "Only unique, validated ascending chunk_index values define source order. "
        "Gaps are allowed; neighbors are stored chunks, not sentences or missing paragraphs. "
        "Overlaps are preserved, not merged. Each window is one read snapshot.</p>"
    )
    yield from _context_window(page, params)
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
                    "/explore/context",
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
        ),
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
) -> HTMLResponse:
    """Read exact source context, preserving native navigation back to the original page."""
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
        ),
    )
