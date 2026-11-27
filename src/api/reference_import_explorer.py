"""Native browser upload, escaped preview and explicit commit of reference-library imports."""

import sqlite3
from collections.abc import Iterator

from fastapi import APIRouter, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.routing import APIRoute
from pydantic import TypeAdapter, ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.types import Receive, Scope, Send

from api.collection_explorer import _navigation
from api.dependencies import AppContainer
from api.explorer import (
    _WRITE_HEADERS,
    _error,
    _escaped,
    _identity,
    _link,
    _literal,
    _page,
    _PageTooLargeError,
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
from api.reference_import import read_preview_request
from ingestion.reference_files import (
    MAX_FILE_BYTES,
    MAX_RECORDS_PER_FILE,
    REFERENCE_FORMATS,
)
from ingestion.reference_import import (
    MAX_FILES,
    ImportCandidate,
    ImportFileSummary,
    ReferenceImportPreview,
)
from retrieval.scope import MAX_DOCUMENT_IDS
from storage.paper_collections import CollectionError, CollectionId, PaperCollection, Revision

PATH = "/explore/collections/import"
COMMIT_PATH = PATH + "/commit"
MAX_QUERY_BYTES = 4_096
MAX_COMMIT_BYTES = 1_024
MAX_LISTED = 100
_COLLECTION_ID = TypeAdapter(CollectionId)
_REVISION = TypeAdapter(Revision)
_FORMAT_LABELS = {
    "ris": "RIS (.ris: PubMed, Embase, Scopus, Web of Science, Zotero, Mendeley, EndNote)",
    "medline": "PubMed MEDLINE / NBIB (.nbib, .txt)",
    "csl_json": "CSL-JSON (.json: Zotero, Mendeley)",
    "csv": "CSV (ASReview-style columns; Excel UTF-8)",
    "tsv": "TSV / tab-delimited (Web of Science, Excel Unicode text)",
}
_STATUS_LABELS = {
    "duplicate_in_import": "Duplicate within this import",
    "duplicate_in_collection": "Already in the collection",
}
_FAILURES = {
    "reference_import_origin": (
        "Open the import form again: one exact same-origin Origin is required. "
        "Nothing was previewed or imported."
    ),
    "reference_import_method": "Use GET to open the form and an explicit POST to submit it.",
    "invalid_reference_import_form": (
        "The form fields are invalid or repeated. Nothing was previewed or imported."
    ),
    "reference_import_form_too_large": "The request exceeds its byte bound; nothing was stored.",
    "reference_import_storage_error": (
        "Import, collection or corpus storage is unavailable. No success is being reported; "
        "inspect the collection before retrying."
    ),
    "html_page_too_large": (
        "The complete escaped preview exceeds 1 MiB. Nothing was imported; use the JSON API "
        "(POST /collections/imports/preview) for this many records."
    ),
    "html_text_not_representable": (
        "A value cannot be displayed faithfully as UTF-8 HTML. Nothing was imported; use the "
        "JSON API instead."
    ),
}


def _failure(status: int, code: str, root_path: str, message: str | None = None) -> HTMLResponse:
    return _error(
        status,
        code,
        "Reference import unavailable",
        _escaped(message or _FAILURES.get(code, "The import failed. No success is reported.")),
        navigation=_navigation(None, root_path)
        + _link("reopen-import", root_path + PATH, "Open a fresh import form"),
        importing=True,
    )


class _ImportRoute(APIRoute):
    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None:
        if self.methods is not None and scope["method"] not in self.methods:
            response = _failure(
                405, "reference_import_method", scope.get("root_path", "").rstrip("/")
            )
            response.headers["Allow"] = ", ".join(sorted(self.methods))
            await response(scope, receive, send)
            return
        await super().handle(scope, receive, send)


def _target_from_query(query: bytes) -> tuple[str, int] | None:
    fields = parse_fields(query, max_fields=2, max_field_characters=64)
    if not fields:
        return None
    if set(fields) != {"collection_id", "collection_revision"}:
        raise ValueError("Use collection_id with collection_revision, or no parameters.")
    revision = _REVISION.validate_python(unsigned_decimal(fields["collection_revision"]))
    return _COLLECTION_ID.validate_python(fields["collection_id"]), revision


def _form(
    collection: PaperCollection | None, revision: int | None, root_path: str
) -> Iterator[str]:
    yield _navigation(None, root_path)
    yield (
        '<p class="notice" id="import-help">Export your search results or library, choose '
        f"1-{MAX_FILES} files and preview. Each file may have at most {MAX_FILE_BYTES} bytes "
        f"and {MAX_RECORDS_PER_FILE} records; a collection holds at most {MAX_DOCUMENT_IDS} "
        "papers. Preview shows every skipped record, duplicate and new paper before you "
        "decide; include/exclude labels in the files are ignored.</p>"
    )
    yield (
        f'<form id="reference-import" method="post" action="{_escaped(root_path + PATH)}" '
        'enctype="multipart/form-data" accept-charset="UTF-8" autocomplete="off" '
        'aria-describedby="import-help">'
    )
    if collection is not None and revision is not None:
        yield "<h2>Import into this saved collection</h2>"
        yield _literal(collection.name, "collection-name")
        yield _identity(collection.collection_id, "collection-id")
        yield (
            f"<p>Collection revision: {revision}; current members: "
            f'<span id="current-members">{len(collection.document_ids)}</span>. Adding papers '
            "increments the revision, so earlier screening decisions become stale.</p>"
            f'<input type="hidden" name="collection_id" '
            f'value="{_escaped(collection.collection_id)}">'
            f'<input type="hidden" name="collection_revision" value="{revision}">'
        )
    else:
        yield (
            '<div class="worksheet-input"><label for="new-collection-name">New collection '
            'name</label><input id="new-collection-name" name="new_collection_name" '
            'type="text" required maxlength="240"><p>1-120 characters after trimming; the '
            "exact name must be unused.</p></div>"
        )
    yield (
        '<div class="worksheet-input"><label for="reference-files">Reference files</label>'
        '<input id="reference-files" name="files" type="file" multiple required '
        'accept=".ris,.nbib,.medline,.txt,.json,.csv,.tsv,.tab"></div>'
        '<div class="worksheet-input"><label for="reference-format">Format</label>'
        '<select id="reference-format" name="format">'
        '<option value="auto" selected>Detect from content (recommended)</option>'
    )
    for name in REFERENCE_FORMATS:
        yield f'<option value="{name}">{_escaped(_FORMAT_LABELS[name])}</option>'
    yield (
        "</select></div>"
        '<div class="actions"><button type="submit">Preview import</button></div></form>'
    )


def _position(item: ImportCandidate) -> str:
    line = f", line {item.line}" if item.line is not None else ""
    return f"file {item.file_index}, record {item.record_index}{line}"


def _candidate(item: ImportCandidate) -> Iterator[str]:
    yield f'<article class="import-candidate" data-status="{item.status}">'
    yield _literal(item.title, "candidate-title")
    yield f"<dl><dt>From</dt><dd>{_position(item)}</dd><dt>Year</dt><dd>"
    yield _literal(item.year or "none", "candidate-year")
    yield "</dd><dt>DOI</dt><dd>" + _literal(item.doi or "none", "candidate-doi")
    yield "</dd><dt>PMID</dt><dd>" + _literal(item.pmid or "none", "candidate-pmid")
    yield "</dd><dt>Exact document ID (JSON)</dt><dd>"
    yield _identity(item.document_id, "candidate-document-id") + "</dd>"
    if item.status != "new":
        yield (
            f'<dt>Match</dt><dd><span class="match-reason">{item.match_reason}</span>: '
            f"{_STATUS_LABELS[item.status]}"
        )
        if item.matched_document_id is not None:
            yield _identity(item.matched_document_id, "matched-document-id")
        if item.matched_record is not None:
            yield f" (file {item.matched_record[0]}, record {item.matched_record[1]})"
        yield "</dd>"
    elif item.existing_document:
        yield "<dt>Corpus</dt><dd>Identical paper already stored: reused, not overwritten.</dd>"
    yield "</dl></article>"


def _file(summary: ImportFileSummary) -> Iterator[str]:
    labels = summary.labels
    how = "detected" if summary.format_detected else "chosen"
    yield f'<article class="import-file"><h3>File {summary.file_index}</h3>'
    yield _literal(summary.file_name, "file-name")
    yield (
        f"<dl><dt>Format</dt><dd>{summary.format} ({how})</dd>"
        f"<dt>Bytes / SHA-256</dt><dd>{summary.byte_count} / <code>{summary.sha256}</code></dd>"
        f'<dt>Records read</dt><dd class="records-read">{summary.records_read}</dd>'
        f'<dt>Skipped</dt><dd class="records-skipped">{summary.records_skipped}</dd>'
        f"<dt>Duplicates within this import</dt><dd>{summary.duplicates_in_import}</dd>"
        f"<dt>Already in the collection</dt><dd>{summary.duplicates_in_collection}</dd>"
        f'<dt>New papers</dt><dd class="file-papers-to-add">{summary.papers_to_add}</dd>'
        f"<dt>Labels in the file (ignored)</dt><dd>{labels.included} included, "
        f"{labels.excluded} excluded, {labels.unlabelled} unlabelled</dd>"
    )
    if summary.previously_imported:
        yield "<dt>Same file already imported here</dt><dd>"
        yield "".join(_identity(value, "previous-import") for value in summary.previously_imported)
        yield "</dd>"
    yield "</dl></article>"


def _listed(title: str, identifier: str, total: int) -> str:
    shown = min(total, MAX_LISTED)
    more = f" Showing the first {shown}; the JSON API lists all." if total > shown else ""
    return f'<h2 id="{identifier}">{title} ({total})</h2><p class="muted">{total} total.{more}</p>'


def _preview(preview: ReferenceImportPreview, root_path: str) -> Iterator[str]:
    totals = preview.totals
    yield _navigation(None, root_path)
    yield "<h2>Target</h2>" + _literal(preview.collection_name, "collection-name")
    if preview.target.collection_id is not None:
        yield _identity(preview.target.collection_id, "collection-id")
        yield f"<p>Revision {preview.target.collection_revision}.</p>"
    else:
        yield "<p>A new collection will be created.</p>"
    yield (
        f'<p id="import-summary">Records read: <span id="records-read">{totals.records_read}'
        f'</span>. Skipped: <span id="records-skipped">{totals.records_skipped}</span>. '
        'Duplicates within this import: <span id="duplicates-in-import">'
        f"{totals.duplicates_in_import}</span>. Already in the collection: "
        f'<span id="duplicates-in-collection">{totals.duplicates_in_collection}</span>. '
        f'New papers: <span id="papers-to-add">{totals.papers_to_add}</span> '
        f"(current members {preview.current_members}, capacity {preview.capacity}). "
        f"Labels in the files (ignored): {totals.labels.included} included, "
        f"{totals.labels.excluded} excluded.</p>"
    )
    yield '<ul id="import-notes">'
    yield "".join(f"<li>{_escaped(note)}</li>" for note in preview.notes)
    yield "</ul>"
    if preview.can_commit:
        yield (
            '<form id="reference-import-commit" method="post" '
            f'action="{_escaped(root_path + COMMIT_PATH)}" '
            'enctype="application/x-www-form-urlencoded" accept-charset="UTF-8">'
            f'<input type="hidden" name="draft_id" value="{preview.draft_id}">'
            '<div class="actions"><button type="submit">'
            f"Import {totals.papers_to_add} new papers</button></div></form>"
            f'<p class="muted">The draft expires at {preview.expires_at.isoformat()}.</p>'
        )
    elif preview.blocking_reason == "nothing_new":
        yield (
            '<p class="notice" id="blocking-reason">Nothing new to import: every record is a '
            "duplicate or was skipped. Nothing will be written.</p>"
        )
    else:
        yield (
            '<p class="notice" id="blocking-reason">These new papers would exceed the '
            f"{preview.capacity}-paper collection limit. Split the files or import into a new "
            "collection. Nothing will be written.</p>"
        )
    yield _link("restart-import", root_path + PATH, "Start a different import")
    yield "<h2>Files</h2>"
    for summary in preview.files:
        yield from _file(summary)
    for status, title, identifier in (
        ("new", "Papers that would be added", "new-papers"),
        ("duplicate_in_collection", "Already in the collection", "collection-duplicates"),
        ("duplicate_in_import", "Duplicates within this import", "import-duplicates"),
    ):
        items = [item for item in preview.candidates if item.status == status]
        yield _listed(title, identifier, len(items))
        for item in items[:MAX_LISTED]:
            yield from _candidate(item)
    skipped = [(file.file_index, item) for file in preview.files for item in file.skipped]
    yield _listed("Skipped records", "skipped-records", len(skipped))
    for file_index, skip in skipped[:MAX_LISTED]:
        line = f", line {skip.line}" if skip.line is not None else ""
        yield (
            f'<article class="skipped-record"><p>File {file_index}, record '
            f'{skip.record_index}{line}: <span class="skip-reason">{_escaped(skip.reason)}'
            f"</span>. {_escaped(skip.message)}</p></article>"
        )


router = APIRouter(route_class=_ImportRoute, tags=["Native reference import"])
_RESPONSES: dict[int | str, dict[str, object]] = {
    403: {"description": "Missing, ambiguous or cross-origin browser submission"},
    404: {"description": "Unknown collection or unknown/expired preview draft"},
    405: {"description": "Unsupported method"},
    409: {"description": "Stale revision, name conflict, nothing new or already imported"},
    413: {"description": "Request, file, record count or escaped page exceeds its bound"},
    415: {"description": "Unsupported form encoding"},
    422: {"description": "Invalid form, target, file, format or collection capacity"},
    503: {"description": "Import, collection or corpus storage unavailable"},
}


async def _guarded(request: Request, action: str) -> Response:
    root_path = request.scope.get("root_path", "").rstrip("/")
    container: AppContainer = request.app.state.container
    try:
        query = request.scope.get("query_string", b"")
        if request.method == "GET":
            if len(query) > MAX_QUERY_BYTES:
                raise FormTooLargeError()
            if await read_body(request, max_bytes=MAX_QUERY_BYTES):
                raise ValueError("GET cannot accept a body.")
            target = _target_from_query(query)
            collection = None
            if target is not None:
                collection = await run_in_threadpool(container.paper_collections.get, target[0])
                if collection.revision != target[1]:
                    raise CollectionError(
                        "collection_revision_conflict",
                        "The collection changed. Open its current revision before importing.",
                        409,
                    )
            return _page(
                "Import references",
                _form(collection, None if target is None else target[1], root_path),
                importing=True,
            )
        try:
            require_same_origin(request)
        except ValueError as exc:
            raise CollectionError(
                "reference_import_origin", _FAILURES["reference_import_origin"], 403
            ) from exc
        if query:
            raise ValueError("POST fields belong only in the body.")
        if action == "preview":
            if (
                not request.headers.get("content-type", "")
                .lower()
                .startswith("multipart/form-data")
            ):
                raise CollectionError(
                    "reference_import_media_type", "Submit the multipart upload form.", 415
                )
            files, chosen = await read_preview_request(request)
            preview = await run_in_threadpool(container.reference_imports.preview, files, chosen)
            return _page("Preview reference import", _preview(preview, root_path), importing=True)
        try:
            require_urlencoded(request)
        except ValueError as exc:
            raise CollectionError(
                "reference_import_media_type", "Use the URL-encoded import button.", 415
            ) from exc
        fields = parse_fields(
            await read_body(request, max_bytes=MAX_COMMIT_BYTES),
            max_fields=1,
            max_field_characters=64,
        )
        if set(fields) != {"draft_id"}:
            raise ValueError("Expected exactly one draft_id.")
        result = await run_in_threadpool(container.reference_imports.commit, fields["draft_id"])
        return RedirectResponse(
            _url(
                root_path + "/explore/collection",
                {
                    "collection_id": result.collection.collection_id,
                    "collection_revision": result.collection.revision,
                },
            ),
            status_code=303,
            headers=_WRITE_HEADERS,
        )
    except CollectionError as exc:
        return _failure(exc.status_code, exc.code, root_path, str(exc))
    except sqlite3.Error:
        return _failure(503, "reference_import_storage_error", root_path)
    except FormTooLargeError:
        return _failure(413, "reference_import_form_too_large", root_path)
    except _PageTooLargeError:
        return _failure(413, "html_page_too_large", root_path)
    except _UnrepresentableTextError:
        return _failure(409, "html_text_not_representable", root_path)
    except (ValueError, ValidationError, RecursionError):
        return _failure(422, "invalid_reference_import_form", root_path)


@router.get(PATH, response_class=HTMLResponse)
@router.post(PATH, response_class=HTMLResponse, responses=_RESPONSES)
async def reference_import_page(request: Request) -> Response:
    """GET opens the upload form; an explicit POST previews and stores only a draft."""
    return await _guarded(request, "preview")


@router.post(
    COMMIT_PATH,
    response_class=HTMLResponse,
    responses={303: {"description": "Imported; open the saved collection"}, **_RESPONSES},
)
async def commit_reference_import(request: Request) -> Response:
    """Explicitly import a previewed draft, then open the saved collection."""
    return await _guarded(request, "commit")
