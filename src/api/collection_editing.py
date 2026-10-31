"""Stateless, explicit collection deltas; filtered candidates never replace membership."""

import json
import sqlite3
from collections.abc import Iterator
from typing import Annotated, Literal

from fastapi import APIRouter, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, BeforeValidator, ConfigDict
from starlette.concurrency import run_in_threadpool
from starlette.types import Receive, Scope, Send

from api.collection_creation import _FAILURES as _CREATION_FAILURES
from api.collection_creation import MAX_FIELD_CHARACTERS, MAX_QUERY_BYTES, _Open
from api.collection_explorer import _ERRORS as _COLLECTION_ERRORS
from api.collection_explorer import _navigation
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
from retrieval.scope import MAX_DOCUMENT_IDS
from storage.document_catalog import (
    MAX_SOURCE_CHARACTERS,
    MAX_TITLE_CHARACTERS,
    DocumentCatalogError,
    DocumentCatalogPage,
)
from storage.paper_collections import (
    MAX_NAME_LENGTH,
    CollectionEdit,
    CollectionError,
    CollectionId,
    CollectionReplacement,
    PaperCollection,
    Revision,
)

PATH = "/explore/collections/edit"
MAX_FORM_BYTES = 524_288
_ADDITIONS = tuple(f"add_{index}" for index in range(1, MAX_DOCUMENT_IDS + 1))
_REMOVALS = tuple(f"remove_{index}" for index in range(1, MAX_DOCUMENT_IDS + 1))


class _OpenEdit(_Open):
    collection_id: CollectionId
    expected_revision: Annotated[Revision, BeforeValidator(unsigned_decimal)]


class _Submission(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    collection_id: CollectionId
    action: Literal["preview", "save"]
    changes: CollectionEdit


_FAILURES = {
    **_CREATION_FAILURES,
    **_COLLECTION_ERRORS,
    "invalid_collection_edit": (
        "Provide the exact collection ID, a positive expected_revision, a name of 1-120 "
        "characters and distinct JSON-string add_1..100 / remove_1..100 choices. Add only "
        "nonmembers; remove only current members. The result must contain 1-100 papers. "
        "Unknown/repeated fields, overlapping choices and malformed encoding fail. "
        "GET accepts only collection_id, expected_revision, limit, source, title and cursor; "
        "POST accepts only body fields and action preview or save. Nothing was saved."
    ),
    "collection_edit_unchanged": "No name or membership change was selected. Nothing was saved.",
    "collection_revision_exhausted": "The collection revision cannot increase. Nothing was saved.",
    "collection_form_method": "Use GET to choose changes, then explicit POST preview and save.",
    "collection_form_too_large": (
        "The query exceeds 16384 bytes or the form body exceeds 524288 bytes. Nothing was saved."
    ),
}


def _failure(status: int, code: str, root_path: str) -> HTMLResponse:
    return _error(
        status,
        code,
        "Collection editing unavailable",
        _FAILURES.get(code, "The collection service failed. No success is being reported."),
        navigation=_navigation(None, root_path),
        editing=True,
    )


class _EditingRoute(APIRoute):
    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None:
        if self.methods is not None and scope["method"] not in self.methods:
            response = _failure(
                405, "collection_form_method", scope.get("root_path", "").rstrip("/")
            )
            response.headers["Allow"] = "GET, POST"
            await response(scope, receive, send)
            return
        await super().handle(scope, receive, send)


def _hidden(name: str, value: str | int) -> str:
    return f'<input type="hidden" name="{name}" value="{_escaped(str(value))}">'


def _form(identifier: str, root_path: str) -> str:
    return (
        f'<form id="{identifier}" method="post" action="{_escaped(root_path + PATH)}" '
        'enctype="application/x-www-form-urlencoded" accept-charset="UTF-8" autocomplete="off">'
    )


def _choice(field: str, identifier: str, label: str) -> str:
    return (
        f'<label class="collection-choice" for="{field}">'
        f'<input id="{field}" type="checkbox" name="{field}" '
        f'value="{_escaped(json.dumps(identifier, ensure_ascii=True))}">{label}</label>'
        + _identity(identifier, "document-id")
    )


def _selection(collection: PaperCollection) -> dict[str, str | int]:
    return {
        "collection_id": collection.collection_id,
        "expected_revision": collection.revision,
    }


def _navigation_to(collection: PaperCollection, root_path: str) -> str:
    return _navigation(None, root_path) + _link(
        "inspect-collection",
        _url(
            root_path + "/explore/collection",
            {
                "collection_id": collection.collection_id,
                "collection_revision": collection.revision,
            },
        ),
        "Inspect this collection (leaves unsaved changes)",
    )


def _editor(
    collection: PaperCollection, page: DocumentCatalogPage, options: _OpenEdit, root_path: str
) -> Iterator[str]:
    yield _navigation_to(collection, root_path)
    yield _literal(collection.name, "collection-name")
    yield _identity(collection.collection_id, "collection-id")
    yield (
        f"<p>Editing revision {collection.revision}; "
        f"{len(collection.document_ids)} saved members.</p>"
        '<p class="notice" id="edit-help">All unchecked members are retained, including hidden '
        "or off-page members. Only checked removals and additions change membership. "
        "Filters affect addition candidates only. Changing pages or filters opens a fresh form: "
        "unsaved name and choices are not carried forward. Preview before saving each batch. "
        "Removing membership does not delete a paper. A rename also makes earlier screening "
        "decisions stale.</p>"
    )
    yield from _filters(
        options.limit,
        options.source,
        options.title,
        root_path,
        endpoint=PATH,
        hidden=_selection(collection),
        form_id="collection-candidates",
    )
    yield _form("collection-edit", root_path)
    for key, value in {**_selection(collection), "action": "preview"}.items():
        yield _hidden(key, value)
    yield (
        '<div class="worksheet-input"><label for="collection-name">Collection name</label>'
        '<input id="collection-name" name="name" type="text" required '
        f'maxlength="{2 * MAX_NAME_LENGTH}" value="{_escaped(collection.name)}" '
        'aria-describedby="name-help"><p id="name-help">1-120 Unicode characters after trimming; '
        "no control characters. Exact names must be unique. Server limits count Unicode "
        "characters, browsers count UTF-16 units.</p></div>"
        '<fieldset class="worksheet-input"><legend>Remove only checked current members '
        "(complete membership)</legend>"
    )
    for index, identifier in enumerate(collection.document_ids, 1):
        yield "<article>" + _choice(f"remove_{index}", identifier, f"Remove member {index}")
        yield _link(
            f"inspect-existing-{index}",
            _url(root_path + "/explore/document", {"document_id": identifier}),
            "Inspect exact paper (leaves unsaved changes)",
        )
        yield "</article>"
    yield (
        '</fieldset><fieldset class="worksheet-input"><legend>Add only checked candidate papers'
        "</legend>"
        f"<p>{len(page.documents)} catalog rows on this page (limit {options.limit}). "
        "Current members are marked, never preselected. Ascending SQLite BINARY document-ID "
        "order; catalog pages and collection metadata are not a frozen corpus snapshot.</p>"
    )
    for index, document in enumerate(page.documents, 1):
        yield "<article>"
        if document.document_id in collection.document_ids:
            yield "<p>Already a member; retained unless explicitly removed above.</p>"
            yield _identity(document.document_id, "document-id")
        else:
            yield _choice(f"add_{index}", document.document_id, f"Add candidate {index}")
        yield _literal(document.title or "Untitled paper", "document-title")
        yield _truncation("Title", document.title_truncated, MAX_TITLE_CHARACTERS)
        yield _literal(document.source, "document-source")
        yield _truncation("Source", document.source_truncated, MAX_SOURCE_CHARACTERS)
        yield _link(
            f"inspect-candidate-{index}",
            _url(root_path + "/explore/document", {"document_id": document.document_id}),
            "Inspect candidate (leaves unsaved changes)",
        )
        yield "</article>"
    if not page.documents:
        yield "<p>No matching candidates. Existing membership is unaffected.</p>"
    yield (
        '</fieldset><div class="actions"><button type="submit">Preview changes</button>'
        "</div></form>"
    )
    yield _pagination(
        _url(root_path + PATH, {**options.parameters(), "cursor": None}),
        _url(root_path + PATH, {**options.parameters(), "cursor": page.next_cursor})
        if page.next_cursor is not None
        else None,
    )


def _open(container: AppContainer, options: _OpenEdit, root_path: str) -> HTMLResponse:
    with container.paper_collections.resolved_collection(
        options.collection_id, expected_revision=options.expected_revision
    ) as (_, collection):
        pass
    page = container.document_catalog.list_documents(
        limit=options.limit, source=options.source, title=options.title, cursor=options.cursor
    )
    return _page(
        "Edit a paper collection", _editor(collection, page, options, root_path), editing=True
    )


def _submission(fields: dict[str, str]) -> _Submission:
    base = {"collection_id", "expected_revision", "name", "action"}
    if not base.issubset(fields) or set(fields) - base - set(_ADDITIONS) - set(_REMOVALS):
        raise ValueError("Expected bounded indexed deltas and exact save metadata.")
    return _Submission.model_validate(
        {
            "collection_id": fields["collection_id"],
            "action": fields["action"],
            "changes": {
                "name": fields["name"],
                "expected_revision": unsigned_decimal(fields["expected_revision"]),
                "add_document_ids": tuple(
                    _DOCUMENT_ID.validate_json(fields[key]) for key in _ADDITIONS if key in fields
                ),
                "remove_document_ids": tuple(
                    _DOCUMENT_ID.validate_json(fields[key]) for key in _REMOVALS if key in fields
                ),
            },
        }
    )


def _preview(
    current: PaperCollection,
    replacement: CollectionReplacement,
    changes: CollectionEdit,
    root_path: str,
) -> Iterator[str]:
    yield _navigation_to(current, root_path)
    yield (
        '<p class="notice">Preview only: nothing saved. Confirm the exact changes below. '
        "Save rechecks the revision and all old/new members atomically. No automatic retry "
        "or merge occurs. No-op edits are rejected.</p>"
    )
    yield "<h2>Name before</h2>" + _literal(current.name, "before-name")
    yield "<h2>Name after</h2>" + _literal(replacement.name, "after-name")
    yield (
        f'<p>Before: <span id="before-count">{len(current.document_ids)}</span>; '
        f'retained: <span id="retained-count">'
        f"{len(current.document_ids) - len(changes.remove_document_ids)}</span>; "
        f'after: <span id="after-count">{len(replacement.document_ids)}</span>. '
        f"Expected revision {current.revision}; successful save advances it by one.</p>"
    )
    for label, identifiers, css in (
        ("Remove memberships only", changes.remove_document_ids, "removed-id"),
        ("Add memberships", changes.add_document_ids, "added-id"),
        ("Complete resulting membership (saved order)", replacement.document_ids, "result-id"),
    ):
        yield f"<h2>{label}</h2>"
        if not identifiers:
            yield "<p>None.</p>"
        for identifier in identifiers:
            yield _identity(identifier, css)
    yield _form("collection-save", root_path)
    for key, value in {**_selection(current), "name": replacement.name, "action": "save"}.items():
        yield _hidden(key, value)
    for prefix, identifiers in (
        ("add", changes.add_document_ids),
        ("remove", changes.remove_document_ids),
    ):
        for index, identifier in enumerate(identifiers, 1):
            yield _hidden(f"{prefix}_{index}", json.dumps(identifier, ensure_ascii=True))
    yield '<div class="actions"><button type="submit">Save changes</button></div></form>'
    yield _link(
        "discard-changes",
        _url(root_path + PATH, _selection(current)),
        "Discard preview / start again",
    )


def _preview_response(
    container: AppContainer, submission: _Submission, root_path: str
) -> HTMLResponse:
    current, replacement = container.paper_collections.preview_edit(
        submission.collection_id, changes=submission.changes
    )
    return _page(
        "Review collection changes",
        _preview(current, replacement, submission.changes, root_path),
        editing=True,
    )


router = APIRouter(route_class=_EditingRoute, tags=["Native collection editing"])


@router.get(PATH, response_class=HTMLResponse)
@router.post(
    PATH,
    response_class=HTMLResponse,
    responses={
        303: {"description": "Atomic save; open the actual persisted revision"},
        403: {"description": "Missing, ambiguous or cross-origin submission"},
        404: {"description": "Collection missing"},
        405: {"description": "Only GET and explicit POST supported"},
        409: {"description": "Stale revision, missing members, name conflict or invalid storage"},
        413: {"description": "Query, body or complete escaped HTML exceeds its byte bound"},
        415: {"description": "Unsupported form encoding"},
        422: {"description": "Invalid, unknown, repeated, empty or unchanged selection"},
        503: {"description": "Storage unavailable"},
    },
)
async def edit_collection(request: Request) -> Response:
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
            options = _OpenEdit.model_validate(
                parse_fields(query, max_fields=6, max_field_characters=MAX_SOURCE_CHARACTERS)
            )
            return await run_in_threadpool(_open, container, options, root_path)
        submission = _submission(
            parse_fields(
                body,
                max_fields=2 * MAX_DOCUMENT_IDS + 4,
                max_field_characters=MAX_FIELD_CHARACTERS,
            )
        )
        if submission.action == "preview":
            return await run_in_threadpool(_preview_response, container, submission, root_path)
        collection = await run_in_threadpool(
            container.paper_collections.edit,
            submission.collection_id,
            changes=submission.changes,
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
        return _failure(422, "invalid_collection_edit", root_path)
