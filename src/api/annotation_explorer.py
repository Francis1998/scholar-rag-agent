"""Script-free preview and explicit human notes on exact, immutable saved-run passages."""

import json
import sqlite3
from collections.abc import Iterator
from typing import Annotated, Self
from urllib.parse import urlencode
from uuid import UUID, uuid4

from fastapi import APIRouter, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator
from starlette.concurrency import run_in_threadpool
from starlette.types import Receive, Scope, Send

from agent.evidence import SHA256, EvidenceSource
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
    _return_pairs,
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
from api.review_notebook import _NotebookParameters
from api.run_explorer import _EVIDENCE_ERRORS, _navigation
from api.run_explorer import MAX_QUERY_BYTES as MAX_RETURN_QUERY_BYTES
from storage.evidence_annotations import (
    MAX_NOTE_CHARACTERS,
    MAX_QUOTE_CHARACTERS,
    AnnotationError,
    AnnotationIdentity,
    AnnotationSubmission,
    SpanOffset,
    _anchor,
)
from storage.evidence_export import EvidenceExportError

PATH = "/explore/run/annotate"
MAX_FORM_BYTES = 65_536
MAX_QUERY_BYTES = 65_536
_Offset = Annotated[SpanOffset, BeforeValidator(unsigned_decimal)]


class _Selection(_NotebookParameters):
    run_id: AnnotationIdentity
    document_id: AnnotationIdentity
    chunk_id: AnnotationIdentity
    source_text_sha256: Annotated[SHA256, Field(strict=True)]

    @model_validator(mode="after")
    def valid_unicode(self) -> Self:
        json.dumps(self.model_dump(), ensure_ascii=False).encode("utf-8")
        return self

    def token(self) -> str:
        return json.dumps(self.model_dump(), ensure_ascii=True, separators=(",", ":"))


class _Open(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    selection: str = Field(strict=True, max_length=16_384)

    def selected(self) -> _Selection:
        return _Selection.model_validate(
            json.loads(self.selection, object_pairs_hook=_return_pairs)
        )


class _Draft(_Open):
    annotation_id: UUID
    start: _Offset
    end: _Offset
    note: str = Field(strict=True, max_length=MAX_NOTE_CHARACTERS)


class _Save(_Open):
    submission: str = Field(strict=True, max_length=32_768)


_FAILURES = {
    **_EVIDENCE_ERRORS,
    "invalid_annotation_form": (
        "Use the opened exact frozen selection, UUID, explicit start/end integer offsets and "
        "a nonblank note of 1-1000 Unicode code points without NUL. The span must select 1-1000 "
        "code points. Unknown/duplicate fields, malformed JSON/encoding, GET bodies and POST "
        "queries are rejected. No input is echoed and nothing was written."
    ),
    "annotation_form_origin": (
        "Origin must be exactly this scheme, host and port; Fetch Metadata, when present, must "
        "be one same-origin value. Missing, null, ambiguous or different origins fail closed. "
        "Reopen the local form. Nothing was written."
    ),
    "annotation_form_encoding": "Use a UTF-8 URL-encoded form without content encoding.",
    "annotation_form_too_large": "The URL or form body exceeds 65536 bytes. Nothing was written.",
    "annotation_return_too_large": (
        "Return navigation exceeds the existing saved-reader URL bound. Reopen with shorter "
        "return filters. Nothing was written."
    ),
    "annotation_form_method": (
        "Use GET to open the composer, POST to preview, then an explicit POST to save. "
        "The save endpoint accepts only POST."
    ),
    "annotation_id_conflict": (
        "This UUID already has different content. The original annotation was not overwritten. "
        "Inspect the notebook; reopen the composer only to create a separate opinion."
    ),
    "annotation_source_not_found": "This exact chunk is not in this run's frozen evidence.",
    "invalid_annotation_selector": (
        "Document, chunk, digest, half-open code-point offsets and quote must match the exact "
        "frozen source. No normalization, fuzzy match or replacement occurrence is used."
    ),
    "html_page_too_large": "The complete escaped page exceeds 1 MiB. No partial form is shown.",
}


def _failure(status: int, code: str, root_path: str) -> HTMLResponse:
    return _error(
        status,
        code,
        "Frozen annotation unavailable",
        _FAILURES.get(code, "Annotation authoring failed. No success is being reported."),
        navigation=_link(
            "back-to-runs", root_path + "/explore/runs", "Reopen the exact saved run / notebook"
        ),
        annotation=True,
    )


class _AnnotationExplorerRoute(APIRoute):
    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None:
        if self.methods is not None and scope["method"] not in self.methods:
            response = _failure(
                405, "annotation_form_method", scope.get("root_path", "").rstrip("/")
            )
            response.headers["Allow"] = (
                "GET, POST" if self.path == PATH else ", ".join(sorted(self.methods))
            )
            await response(scope, receive, send)
            return
        await super().handle(scope, receive, send)


def _read_source(container: AppContainer, selection: _Selection) -> EvidenceSource:
    bundle = container.saved_run_evidence.export(selection.run_id)
    source = next(
        (item for item in bundle.snapshot.sources if item.chunk.chunk_id == selection.chunk_id),
        None,
    )
    if source is None:
        raise AnnotationError("annotation_source_not_found", "", 404)
    if (
        source.chunk.document_id != selection.document_id
        or source.text_sha256 != selection.source_text_sha256
    ):
        raise AnnotationError("invalid_annotation_selector", "", 422)
    return source


def _return_url(path: str, selection: _Selection, **extra: str) -> str:
    parameters = {**selection.notebook_parameters(), **extra}
    if len(urlencode({key: value for key, value in parameters.items() if value is not None})) > (
        MAX_RETURN_QUERY_BYTES
    ):
        raise AnnotationError("annotation_return_too_large", "", 413)
    return _url(path, parameters)


def _draft(
    selection: _Selection,
    source: EvidenceSource,
    identifier: UUID,
    root_path: str,
    payload: AnnotationSubmission | None,
) -> Iterator[str]:
    start, end = ("", "") if payload is None else (str(payload.start), str(payload.end))
    note = "" if payload is None else payload.note
    # HTML strips one leading LF in a textarea; preserve the submitted note on explicit editing.
    textarea = ("\n" if note.startswith("\n") else "") + _escaped(note)
    yield (
        '<form id="annotation-draft" method="post" enctype="application/x-www-form-urlencoded" '
        f'action="{_escaped(root_path + PATH)}" accept-charset="UTF-8" autocomplete="off">'
        f'<input type="hidden" name="selection" value="{_escaped(selection.token())}">'
        f'<input type="hidden" name="annotation_id" value="{identifier}">'
        '<div><label for="start">Start (0-based, inclusive)</label>'
        f'<input id="start" name="start" type="number" min="0" max="{len(source.chunk.text)}" '
        f'step="1" value="{start}" required inputmode="numeric" aria-describedby="span-help"></div>'
        '<div><label for="end">End (exclusive)</label>'
        f'<input id="end" name="end" type="number" min="1" max="{len(source.chunk.text)}" '
        f'step="1" value="{end}" required inputmode="numeric" aria-describedby="span-help"></div>'
        '<div class="worksheet-input"><p id="span-help">Offsets count Unicode code points '
        "(Python string indices), not UTF-8 bytes or UTF-16 units. An emoji counts as one; "
        "combining marks and CR/LF each count separately. Start must be less than end. "
        f"Select at most {MAX_QUOTE_CHARACTERS} code points; repeated text needs the offsets "
        "of the occurrence you intend. Preview checks and highlights exactly that span.</p></div>"
        '<div class="worksheet-input"><label for="note">Your human note</label>'
        f'<textarea id="note" name="note" required aria-describedby="note-help" '
        f'spellcheck="false">{textarea}</textarea>'
        '<p id="note-help">1-1000 Unicode code points, nonblank, no NUL. The server counts '
        "code points; browsers serialize textarea line breaks as CRLF. No trimming or "
        "normalization is applied by the server. No credentials or personal data.</p></div>"
        '<div class="actions"><button type="submit">Preview exact annotation</button>'
        "</div></form>"
    )


def _fragments(
    selection: _Selection,
    source: EvidenceSource,
    identifier: UUID,
    root_path: str,
    payload: AnnotationSubmission | None,
) -> Iterator[str]:
    yield _navigation(selection, root_path)
    yield '<nav aria-label="Annotation return navigation">'
    yield _link(
        "back-to-notebook",
        _return_url(root_path + "/explore/run/notebook", selection),
        "Back to this notebook page",
    )
    yield _link(
        "full-frozen-source",
        selection.evidence_url(root_path) + f"#source-{source.rank}",
        "Read this passage in the complete saved evidence",
    )
    yield "</nav>"
    yield '<p class="notice">Nothing has been saved by opening or previewing this form. '
    yield (
        "Human notes are opinions, not authenticated reviews or scientific validation. "
        "This source is frozen, not today's corpus.</p><dl>"
    )
    for label, value, css_class in (
        ("Run ID (JSON)", selection.run_id, "run-id"),
        ("Document ID (JSON)", selection.document_id, "document-id"),
        ("Chunk ID (JSON)", selection.chunk_id, "chunk-id"),
    ):
        yield f"<dt>{label}</dt><dd>{_identity(value, css_class)}</dd>"
    yield (
        "<dt>Frozen text SHA-256</dt><dd>"
        + _literal(source.text_sha256, "source-digest")
        + f"</dd></dl><h2>Frozen evidence {source.rank}</h2>"
        + f"<p>Complete captured passage: {len(source.chunk.text)} Unicode code points. "
        "Not necessarily a complete paper. Text and source labels are never executed.</p>"
    )
    text = source.chunk.text
    if payload is None:
        yield _literal(text, "passage-text")
    else:
        yield (
            '<pre class="literal passage-text" dir="auto"><code translate="no">'
            + _escaped(text[: payload.start])
            + '<mark id="annotation-preview-quote" tabindex="-1">'
            + _escaped(payload.quote)
            + "</mark>"
            + _escaped(text[payload.end :])
            + "</code></pre>"
        )
        yield "<h2>Preview your exact annotation</h2>"
        yield _literal(f"[{payload.start}, {payload.end})", "quote-span")
        yield "<h3>Exact quote</h3>" + _literal(payload.quote, "annotation-quote")
        yield "<h3>Your note</h3>" + _literal(payload.note, "annotation-note")
        yield "<p>Retry UUID (not an author identity):</p>"
        yield _identity(str(identifier), "annotation-id")
        submission = json.dumps(payload.model_dump(mode="json"), ensure_ascii=True)
        fields = {"selection": selection.token(), "submission": submission}
        if len(urlencode(fields).encode("ascii")) > MAX_FORM_BYTES:
            raise FormTooLargeError()
        yield (
            '<form id="annotation-save" method="post" '
            'enctype="application/x-www-form-urlencoded" '
            f'action="{_escaped(root_path + PATH + "/save")}" accept-charset="UTF-8" '
            'autocomplete="off">'
        )
        for name, value in fields.items():
            yield f'<input type="hidden" name="{name}" value="{_escaped(value)}">'
        yield (
            '<div class="actions"><button type="submit">Save annotation</button></div>'
            '<div class="worksheet-input"><p>This explicit POST appends the previewed note. '
            "An unchanged retry reuses this UUID and its original timestamp; changed content "
            "under an already saved UUID conflicts.</p></div></form>"
            "<details><summary>Edit and preview again (keep this retry UUID)</summary>"
        )
    yield from _draft(selection, source, identifier, root_path, payload)
    if payload is not None:
        yield "</details>"


def _render(
    selection: _Selection,
    source: EvidenceSource,
    identifier: UUID,
    root_path: str,
    payload: AnnotationSubmission | None = None,
) -> HTMLResponse:
    if payload is not None:
        _anchor(payload, {source.chunk.chunk_id: source}, saved=False)
    return _page(
        "Annotate frozen evidence",
        _fragments(selection, source, identifier, root_path, payload),
        annotation=True,
    )


def _open(container: AppContainer, form: _Open, root_path: str) -> HTMLResponse:
    selection = form.selected()
    return _render(selection, _read_source(container, selection), uuid4(), root_path)


def _preview(container: AppContainer, form: _Draft, root_path: str) -> HTMLResponse:
    selection = form.selected()
    source = _read_source(container, selection)
    payload = AnnotationSubmission(
        annotation_id=form.annotation_id,
        document_id=selection.document_id,
        chunk_id=selection.chunk_id,
        source_text_sha256=selection.source_text_sha256,
        start=form.start,
        end=form.end,
        quote=source.chunk.text[form.start : form.end],
        note=form.note,
    )
    return _render(selection, source, form.annotation_id, root_path, payload)


def _save(container: AppContainer, form: _Save, root_path: str) -> RedirectResponse:
    selection = form.selected()
    payload = AnnotationSubmission.model_validate(
        json.loads(form.submission, object_pairs_hook=_return_pairs)
    )
    # Fail unrenderable/oversized previews before the sole write, including direct submissions.
    _render(
        selection, _read_source(container, selection), payload.annotation_id, root_path, payload
    )
    redirect = RedirectResponse(
        _return_url(
            root_path + "/explore/run/annotation",
            selection,
            annotation_id=str(payload.annotation_id),
        )
        + "#annotation-quote",
        status_code=303,
        headers=_WRITE_HEADERS,
    )
    container.evidence_annotations.create(selection.run_id, payload)
    return redirect


router = APIRouter(route_class=_AnnotationExplorerRoute, tags=["Frozen annotation authoring"])


@router.get(PATH, response_class=HTMLResponse)
@router.post(PATH, response_class=HTMLResponse)
@router.post(
    PATH + "/save",
    response_class=HTMLResponse,
    responses={
        303: {"description": "Annotation persisted or identical UUID replayed; exact-quote GET"},
        403: {"description": "Missing, ambiguous or different browser origin"},
        409: {"description": "UUID conflict or unavailable/invalid frozen evidence"},
        413: {"description": "Form, return URL or complete HTML exceeds a fixed bound"},
        415: {"description": "Unsupported form encoding"},
        422: {"description": "Invalid exact selection, offsets, note or ambiguous fields"},
        503: {"description": "Saved evidence or annotation storage unavailable"},
    },
)
async def explore_annotation_authoring(request: Request) -> Response:
    root_path = request.scope.get("root_path", "").rstrip("/")
    try:
        if request.method == "POST":
            try:
                require_same_origin(request)
            except ValueError as exc:
                raise AnnotationError("annotation_form_origin", "", 403) from exc
            try:
                require_urlencoded(request)
            except ValueError as exc:
                raise AnnotationError("annotation_form_encoding", "", 415) from exc
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
            opened = _Open.model_validate(parse_fields(query, max_fields=1))
            return await run_in_threadpool(_open, container, opened, root_path)
        if request.scope["route"].path == PATH + "/save":
            save = _Save.model_validate(
                parse_fields(body, max_fields=2, max_field_characters=32_768)
            )
            return await run_in_threadpool(_save, container, save, root_path)
        draft = _Draft.model_validate(parse_fields(body, max_fields=5))
        return await run_in_threadpool(_preview, container, draft, root_path)
    except EvidenceExportError as exc:
        return _failure(exc.status_code, exc.code, root_path)
    except sqlite3.Error:
        return _failure(503, "annotation_storage_unavailable", root_path)
    except FormTooLargeError:
        return _failure(413, "annotation_form_too_large", root_path)
    except _PageTooLargeError:
        return _failure(413, "html_page_too_large", root_path)
    except _UnrepresentableTextError:
        return _failure(409, "html_text_not_representable", root_path)
    except (ValueError, RecursionError):
        return _failure(422, "invalid_annotation_form", root_path)
