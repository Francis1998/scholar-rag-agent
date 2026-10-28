"""Script-free preview, explicit append and exact-UUID reading of human answer opinions."""

import json
import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Self
from urllib.parse import urlencode, urlsplit
from uuid import UUID, uuid4

from fastapi import APIRouter, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, model_validator
from starlette.concurrency import run_in_threadpool
from starlette.types import Receive, Scope, Send

from agent.evidence import EvidenceBundle
from agent.review_models import (
    MAX_COMMENT_CHARACTERS,
    MAX_REVIEW_SEQUENCE,
    AnswerReviewError,
    ReviewComment,
    ReviewDecision,
    ReviewIdentity,
    ReviewSubmission,
    SavedAnswerReview,
)
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
)
from api.review_notebook import _NotebookParameters, _review_record
from api.run_explorer import _EVIDENCE_ERRORS, _navigation, _review_url
from api.run_explorer import MAX_QUERY_BYTES as MAX_RETURN_QUERY_BYTES
from storage.evidence_export import EvidenceExportError

PATH = "/explore/run/review"
MAX_FORM_BYTES = 524_288
MAX_QUERY_BYTES = 65_536
MAX_SUBMISSION_CHARACTERS = 262_144
_IDENTITY = TypeAdapter(ReviewIdentity)


class _Selection(_NotebookParameters):
    run_id: ReviewIdentity

    @model_validator(mode="after")
    def valid_unicode(self) -> Self:
        json.dumps(self.model_dump(), ensure_ascii=False).encode("utf-8")
        return self

    def token(self) -> str:
        return json.dumps(self.notebook_parameters(), ensure_ascii=True, separators=(",", ":"))


class _Open(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    selection: str = Field(strict=True, max_length=16_384)

    def selected(self) -> _Selection:
        return _Selection.model_validate(
            json.loads(self.selection, object_pairs_hook=_return_pairs)
        )


class _Draft(_Open):
    review_id: UUID
    decision: ReviewDecision
    comment: ReviewComment


class _Save(_Open):
    submission: str = Field(strict=True, max_length=MAX_SUBMISSION_CHARACTERS)


class _Saved(_Open):
    review_id: UUID


_FAILURES = {
    **_EVIDENCE_ERRORS,
    "invalid_review_form": (
        "Use the exact opened run, a UUID, an explicit accepted / needs_revision / rejected "
        "decision, a nonblank comment of 1-4000 Unicode code points without NUL and optional "
        "actual frozen chunk IDs. Review IDs retain the existing 1-256-code-point, no-control "
        "or outer-whitespace contract. Unknown/duplicate fields, malformed JSON/encoding, "
        "POST queries and GET bodies fail. No input is echoed and nothing was written."
    ),
    "review_form_origin": (
        "Origin must be exactly this scheme, host and port; Fetch Metadata, when present, "
        "must be one same-origin value. Missing, null, ambiguous or different origins fail "
        "closed. Reopen the local form. Nothing was written."
    ),
    "review_form_encoding": "Use a UTF-8 URL-encoded form without content encoding.",
    "review_form_too_large": (
        "The query exceeds 65536 bytes, the form exceeds 524288 bytes, or the complete "
        "preview cannot fit its save form. Nothing was written."
    ),
    "review_return_too_large": (
        "Return navigation exceeds the existing 16384-byte saved-reader URL bound. "
        "Reopen with shorter return filters. Nothing was written."
    ),
    "review_form_method": (
        "Use GET to open or reread, POST to preview, then an explicit POST to save. "
        "The save endpoint accepts only POST; the saved-opinion reader accepts only GET."
    ),
    "review_id_conflict": (
        "This UUID already has different content. The original opinion was not overwritten. "
        "Inspect the notebook; reopen the composer only to create a separate opinion."
    ),
    "invalid_review_references": "Every cited chunk ID must belong to this exact frozen bundle.",
    "review_not_found": (
        "No opinion with this exact UUID exists in this run. No replacement was guessed."
    ),
    "review_read_limit_exceeded": (
        "The exact review exceeds 262144 UTF-8 text bytes; no partial record."
    ),
    "html_page_too_large": "The complete escaped page exceeds 1 MiB. No partial form is shown.",
}


def _failure(status: int, code: str, root_path: str) -> HTMLResponse:
    return _error(
        status,
        code,
        "Saved-answer review unavailable",
        _FAILURES.get(code, "Review authoring failed. No success is being reported."),
        navigation=_link(
            "back-to-runs", root_path + "/explore/runs", "Reopen the exact saved run / notebook"
        ),
        review=True,
    )


class _ReviewExplorerRoute(APIRoute):
    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None:
        if self.methods is not None and scope["method"] not in self.methods:
            response = _failure(405, "review_form_method", scope.get("root_path", "").rstrip("/"))
            response.headers["Allow"] = (
                "GET, POST" if self.path == PATH else ", ".join(sorted(self.methods))
            )
            await response(scope, receive, send)
            return
        await super().handle(scope, receive, send)


def _bounded_url(url: str) -> str:
    if len(urlsplit(url).query.encode("ascii")) > MAX_RETURN_QUERY_BYTES:
        raise AnswerReviewError("review_return_too_large", "", 413)
    return url


def _navigation_links(selection: _Selection, root_path: str) -> Iterator[str]:
    yield _navigation(selection, root_path)
    yield '<nav aria-label="Answer review return navigation">'
    yield _link(
        "back-to-notebook",
        _bounded_url(_url(root_path + "/explore/run/notebook", selection.notebook_parameters())),
        "Back to this notebook page",
    )
    yield _link(
        "back-to-evidence", selection.evidence_url(root_path), "Read complete saved evidence"
    )
    yield "</nav>"


def _answer(bundle: EvidenceBundle) -> Iterator[str]:
    yield (
        '<section id="saved-answer"><h2>Completed saved answer</h2>'
        "<p>Human opinions are not scientific validation. No authenticated reviewer identity "
        "is recorded. Frozen evidence is not today's corpus; model grounding is unchanged.</p>"
        "<dl><dt>Run ID (JSON)</dt><dd>"
    )
    yield _identity(bundle.run_id, "run-id") + "</dd><dt>Recorded ungrounded flag</dt><dd>"
    yield _literal(str(bundle.answer.ungrounded).lower(), "ungrounded") + "</dd></dl>"
    yield "<h3>Original question</h3>" + _literal(bundle.query, "query-text")
    yield _literal(bundle.answer.answer, "answer-text")
    for label, warnings, css_class in (
        ("Saved answer warnings", bundle.answer.warnings, "answer-warning"),
        ("Frozen bundle warnings", bundle.warnings, "bundle-warning"),
    ):
        yield f"<h3>{label}</h3>"
        for warning in warnings:
            yield _literal(warning, css_class)
        if not warnings:
            yield "<p>No warnings recorded in this category; this is not a quality verdict.</p>"
    yield (
        "<details><summary>Recorded claims and citations (not human approval)</summary>"
        + _literal(
            json.dumps(
                bundle.answer.model_dump(mode="json", exclude={"answer", "warnings", "ungrounded"}),
                ensure_ascii=True,
                indent=2,
            ),
            "answer-metadata",
        )
        + "</details></section>"
    )


def _sources(bundle: EvidenceBundle) -> Iterator[str]:
    yield '<section id="frozen-sources"><h2>Complete captured passages</h2>'
    yield (
        "<p>Full frozen chunks, not necessarily complete papers. Source labels are never links.</p>"
    )
    for source in bundle.snapshot.sources:
        yield f'<article id="source-{source.rank}"><h3>Frozen evidence {source.rank}</h3>'
        yield _identity(source.chunk.title, "source-title")
        yield "<dl><dt>Document ID (JSON)</dt><dd>"
        yield _identity(source.chunk.document_id, "document-id") + "</dd>"
        yield (
            "<dt>Chunk ID (JSON)</dt><dd>" + _identity(source.chunk.chunk_id, "chunk-id") + "</dd>"
        )
        yield (
            "<dt>Text SHA-256</dt><dd>"
            + _literal(source.text_sha256, "source-digest")
            + "</dd></dl>"
        )
        yield _literal(source.chunk.text, "passage-text")
        metadata = source.model_dump(mode="json")
        metadata["chunk"].pop("text")
        yield (
            "<details><summary>Frozen identity, rank, score, path and metadata</summary>"
            + _literal(json.dumps(metadata, ensure_ascii=True, indent=2), "source-metadata")
            + "</details></article>"
        )
    if not bundle.snapshot.sources:
        yield (
            "<p>No frozen passages were captured. You may explicitly review without citations.</p>"
        )
    yield "</section>"


def _draft(
    selection: _Selection,
    bundle: EvidenceBundle,
    identifier: UUID,
    root_path: str,
    payload: ReviewSubmission | None,
) -> Iterator[str]:
    comment = "" if payload is None else payload.comment
    # HTML removes one leading LF in textarea contents.
    textarea = ("\n" if comment.startswith("\n") else "") + _escaped(comment)
    yield (
        '<form id="review-draft" method="post" enctype="application/x-www-form-urlencoded" '
        f'action="{_escaped(root_path + PATH)}" accept-charset="UTF-8" autocomplete="off">'
        f'<input type="hidden" name="selection" value="{_escaped(selection.token())}">'
        f'<input type="hidden" name="review_id" value="{identifier}">'
        '<div class="worksheet-input"><label for="decision">Your explicit human verdict</label>'
        '<select id="decision" name="decision" required><option value="">Choose a verdict</option>'
    )
    for decision in ReviewDecision:
        selected = " selected" if payload is not None and payload.decision == decision else ""
        yield f'<option value="{decision.value}"{selected}>{decision.value}</option>'
    yield (
        '</select></div><div class="worksheet-input">'
        '<label for="comment">Your exact comment</label>'
        f'<textarea id="comment" name="comment" required aria-describedby="comment-help" '
        f'spellcheck="false">{textarea}</textarea><p id="comment-help">'
        f"1-{MAX_COMMENT_CHARACTERS} Unicode code points, nonblank, no NUL. No trimming or "
        "normalization by the server. Browsers serialize textarea line breaks as CRLF; "
        "the preview shows the submitted text. No credentials or personal data.</p></div>"
        '<div class="worksheet-input"><fieldset><legend>Optional frozen citations</legend>'
        "<p>Nothing is selected automatically. Choose only passages you intend to cite; "
        "these references do not establish support or entailment.</p>"
    )
    for source in bundle.snapshot.sources:
        chunk_id = source.chunk.chunk_id
        try:
            _IDENTITY.validate_python(chunk_id)
            disabled = ""
        except ValidationError:
            disabled = " disabled"
        checked = " checked" if payload is not None and chunk_id in payload.cited_chunk_ids else ""
        yield (
            f'<label class="review-source-choice" for="cite-{source.rank}">'
            f'<input id="cite-{source.rank}" name="cite_{source.rank}" type="checkbox" '
            f'value="{_escaped(json.dumps(chunk_id, ensure_ascii=True))}"{disabled}{checked}>'
            f"Frozen evidence {source.rank}</label>"
        )
        yield _identity(chunk_id, "citation-choice-id")
        yield _link(
            f"inspect-source-{source.rank}", f"#source-{source.rank}", "Inspect full passage"
        )
        if disabled:
            yield (
                "<p>This exact chunk ID cannot be cited under the existing review ID contract "
                "(1-256 code points; no controls or outer whitespace). It remains visible and "
                "is not normalized or replaced. Reviewing without this citation is allowed.</p>"
            )
    yield (
        '</fieldset></div><div class="actions"><button type="submit">Preview review</button>'
        "</div></form>"
    )


def _validate_references(payload: ReviewSubmission, bundle: EvidenceBundle) -> None:
    if not {source.chunk.chunk_id for source in bundle.snapshot.sources}.issuperset(
        payload.cited_chunk_ids
    ):
        raise AnswerReviewError("invalid_review_references", "", 422)


def _fragments(
    selection: _Selection,
    bundle: EvidenceBundle,
    identifier: UUID,
    root_path: str,
    payload: ReviewSubmission | None,
) -> Iterator[str]:
    yield from _navigation_links(selection, root_path)
    yield '<p class="notice">Nothing has been saved by opening or previewing this form.</p>'
    yield from _answer(bundle)
    if payload is not None:
        yield '<section id="review-preview"><h2>Preview your exact human review</h2>'
        yield _literal(payload.decision.value, "review-decision")
        yield _literal(payload.comment, "review-comment")
        yield "<p>Retry UUID (not a person):</p>" + _literal(str(identifier), "review-id")
        yield "<h3>Exact chosen frozen chunk IDs (JSON)</h3>"
        yield _literal(json.dumps(payload.cited_chunk_ids, ensure_ascii=True), "review-citations")
        fields = {
            "selection": selection.token(),
            "submission": json.dumps(payload.model_dump(mode="json"), ensure_ascii=True),
        }
        if (
            len(fields["submission"]) > MAX_SUBMISSION_CHARACTERS
            or len(urlencode(fields).encode("ascii")) > MAX_FORM_BYTES
        ):
            raise FormTooLargeError()
        yield (
            '<form id="review-save" method="post" enctype="application/x-www-form-urlencoded" '
            f'action="{_escaped(root_path + PATH + "/save")}" '
            'accept-charset="UTF-8" autocomplete="off">'
        )
        for name, value in fields.items():
            yield f'<input type="hidden" name="{name}" value="{_escaped(value)}">'
        yield (
            '<div class="actions"><button type="submit">Save review</button></div>'
            '<div class="worksheet-input"><p>Only this explicit POST appends the previewed '
            "opinion. An unchanged retry preserves its UUID, sequence and timestamp. Changed "
            "content under an already saved UUID conflicts.</p></div></form></section>"
            "<details><summary>Edit and preview again (keep this retry UUID)</summary>"
        )
    yield from _draft(selection, bundle, identifier, root_path, payload)
    if payload is not None:
        yield "</details>"
    yield from _sources(bundle)


def _render(
    selection: _Selection,
    bundle: EvidenceBundle,
    identifier: UUID,
    root_path: str,
    payload: ReviewSubmission | None = None,
) -> HTMLResponse:
    if payload is not None:
        _validate_references(payload, bundle)
    return _page(
        "Review saved answer",
        _fragments(selection, bundle, identifier, root_path, payload),
        review=True,
    )


def _saved_fragments(
    saved: SavedAnswerReview, bundle: EvidenceBundle, selection: _Selection, root_path: str
) -> Iterator[str]:
    yield from _navigation_links(selection, root_path)
    yield (
        '<section id="saved-review"><h2>Exact persisted human opinion</h2>'
        "<p>This GET rereads this run's exact UUID, not the latest review or a moving history "
        "page. Different opinions coexist without consensus; none replaces this record. "
        "Evidence and this opinion are separate reads, not a shared database snapshot.</p>"
    )
    yield from _review_record(saved, bundle, selection, root_path)
    yield _link(
        "review-answer",
        _bounded_url(_review_url(selection.notebook_parameters(), root_path)),
        "Create a separate review of this answer",
    )
    yield "</section>"
    yield from _answer(bundle)
    yield from _sources(bundle)


def _saved_page(
    saved: SavedAnswerReview, bundle: EvidenceBundle, selection: _Selection, root_path: str
) -> HTMLResponse:
    return _page(
        "Saved human review",
        _saved_fragments(saved, bundle, selection, root_path),
        saved_runs=True,
    )


def _open(container: AppContainer, form: _Open, root_path: str) -> HTMLResponse:
    selection = form.selected()
    bundle = container.saved_run_evidence.export(selection.run_id)
    return _render(selection, bundle, uuid4(), root_path)


def _preview(container: AppContainer, fields: dict[str, str], root_path: str) -> HTMLResponse:
    citations = {name: value for name, value in fields.items() if name.startswith("cite_")}
    form = _Draft.model_validate(
        {name: value for name, value in fields.items() if name not in citations}
    )
    selection = form.selected()
    bundle = container.saved_run_evidence.export(selection.run_id)
    chosen: list[str] = []
    for source in bundle.snapshot.sources:
        name = f"cite_{source.rank}"
        if name in citations:
            value = json.loads(citations.pop(name), object_pairs_hook=_return_pairs)
            if value != source.chunk.chunk_id:
                raise ValueError("A checkbox must identify its exact frozen source.")
            chosen.append(source.chunk.chunk_id)
    if citations:
        raise ValueError("Unknown frozen-source selection.")
    payload = ReviewSubmission(
        review_id=form.review_id,
        decision=form.decision,
        comment=form.comment,
        cited_chunk_ids=tuple(chosen),
    )
    return _render(selection, bundle, payload.review_id, root_path, payload)


def _save(container: AppContainer, form: _Save, root_path: str) -> RedirectResponse:
    selection = form.selected()
    payload = ReviewSubmission.model_validate(
        json.loads(form.submission, object_pairs_hook=_return_pairs)
    )
    bundle = container.saved_run_evidence.export(selection.run_id)
    _render(selection, bundle, payload.review_id, root_path, payload)
    # Preflight the complete GET using the maximum valid widths of server-assigned fields.
    prospective = SavedAnswerReview.model_validate(
        {
            **payload.model_dump(),
            "schema_version": "1.0",
            "run_id": selection.run_id,
            "sequence": MAX_REVIEW_SEQUENCE,
            "created_at": datetime.max.replace(tzinfo=UTC).isoformat().replace("+00:00", "Z"),
        }
    )
    _saved_page(prospective, bundle, selection, root_path)
    redirect = RedirectResponse(
        _bounded_url(
            _review_url(selection.notebook_parameters(), root_path, str(payload.review_id))
        )
        + "#saved-review",
        status_code=303,
        headers=_WRITE_HEADERS,
    )
    container.notebook_reviews.create(selection.run_id, payload)
    return redirect


def _read_saved(container: AppContainer, form: _Saved, root_path: str) -> HTMLResponse:
    selection = form.selected()
    bundle = container.saved_run_evidence.export(selection.run_id)
    saved = container.notebook_reviews.get_review(selection.run_id, form.review_id)
    return _saved_page(saved, bundle, selection, root_path)


router = APIRouter(route_class=_ReviewExplorerRoute, tags=["Saved answer review authoring"])


@router.get(PATH, response_class=HTMLResponse)
@router.post(PATH, response_class=HTMLResponse)
@router.get(PATH + "/saved", response_class=HTMLResponse)
@router.post(
    PATH + "/save",
    response_class=HTMLResponse,
    responses={
        303: {"description": "Opinion persisted or identical UUID replayed; exact-opinion GET"},
        403: {"description": "Missing, ambiguous or different browser origin"},
        404: {"description": "No exact saved run or review"},
        409: {"description": "UUID conflict or invalid/unavailable frozen evidence/opinion"},
        413: {
            "description": "Form, return URL, review read or complete HTML exceeds a fixed bound"
        },
        415: {"description": "Unsupported form encoding"},
        422: {"description": "Invalid exact selection, verdict, comment or cited frozen IDs"},
        503: {"description": "Saved evidence or review storage unavailable"},
    },
)
async def explore_review_authoring(request: Request) -> Response:
    root_path = request.scope.get("root_path", "").rstrip("/")
    try:
        if request.method == "POST":
            try:
                require_same_origin(request)
            except ValueError as exc:
                raise AnswerReviewError("review_form_origin", "", 403) from exc
            try:
                require_urlencoded(request)
            except ValueError as exc:
                raise AnswerReviewError("review_form_encoding", "", 415) from exc
        query = request.scope.get("query_string", b"")
        if len(query) > MAX_QUERY_BYTES:
            raise FormTooLargeError()
        if request.method == "POST" and query:
            raise ValueError("POST fields belong only in the body.")
        body = await read_body(request, max_bytes=MAX_FORM_BYTES)
        container: AppContainer = request.app.state.container
        path = request.scope["route"].path
        if request.method == "GET":
            if body:
                raise ValueError("GET cannot accept a body.")
            if path == PATH + "/saved":
                saved = _Saved.model_validate(parse_fields(query, max_fields=2))
                return await run_in_threadpool(_read_saved, container, saved, root_path)
            opened = _Open.model_validate(parse_fields(query, max_fields=1))
            return await run_in_threadpool(_open, container, opened, root_path)
        if path == PATH + "/save":
            save = _Save.model_validate(
                parse_fields(body, max_fields=2, max_field_characters=MAX_SUBMISSION_CHARACTERS)
            )
            return await run_in_threadpool(_save, container, save, root_path)
        fields = parse_fields(body, max_fields=54)
        return await run_in_threadpool(_preview, container, fields, root_path)
    except (EvidenceExportError, AnswerReviewError) as exc:
        return _failure(exc.status_code, exc.code, root_path)
    except sqlite3.Error:
        return _failure(503, "review_storage_unavailable", root_path)
    except FormTooLargeError:
        return _failure(413, "review_form_too_large", root_path)
    except _PageTooLargeError:
        return _failure(413, "html_page_too_large", root_path)
    except (_UnrepresentableTextError, UnicodeEncodeError):
        return _failure(409, "html_text_not_representable", root_path)
    except (ValueError, RecursionError):
        return _failure(422, "invalid_review_form", root_path)
