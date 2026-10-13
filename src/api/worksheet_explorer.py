"""Bounded native forms over the existing model-free research worksheet service."""

import json
import re
from collections.abc import Iterator
from typing import Annotated, Literal, Self
from urllib.parse import parse_qsl

from fastapi import APIRouter, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator
from starlette.types import Receive, Scope, Send

from agent.research_worksheet import ResearchWorksheet, WorksheetError, WorksheetRequest
from api.dependencies import AppContainer
from api.explorer import (
    _HEADERS,
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
    _query_integer,
    _UnrepresentableTextError,
    _url,
)
from storage.document_catalog import DEFAULT_PAGE_SIZE
from storage.paper_collections import CollectionError, CollectionId
from storage.source_context import ContextDocumentIdentity

MAX_FORM_BYTES = 65_536
MAX_QUERY_BYTES = 16_384
MAX_FORM_FIELDS = 6
MAX_FIELD_CHARACTERS = 16_384
MAX_QUESTIONS_TEXT = 2_508
_PassageCount = Annotated[int, Field(strict=True, ge=1, le=3), BeforeValidator(_query_integer)]


class _CatalogReturn(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    catalog_limit: _CatalogLimit = DEFAULT_PAGE_SIZE
    source: _OptionalSource = None
    title: _OptionalTitle = None
    catalog_cursor: ContextDocumentIdentity | None = None

    @model_validator(mode="after")
    def valid_unicode(self) -> Self:
        json.dumps(self.model_dump(), ensure_ascii=False).encode("utf-8")
        return self

    def parameters(self) -> dict[str, str | int | None]:
        return {
            "limit": self.catalog_limit,
            "source": self.source,
            "title": self.title,
            "cursor": self.catalog_cursor,
        }


class _OpenForm(_CatalogReturn):
    document_id: ContextDocumentIdentity | None = None
    collection_id: CollectionId | None = None

    @model_validator(mode="after")
    def unambiguous(self) -> Self:
        if self.document_id is not None and self.collection_id is not None:
            raise ValueError("Prefill one selection only.")
        return self


class _Form(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    questions: str = Field(default="", max_length=MAX_QUESTIONS_TEXT)
    document_ids: str = Field(default="", max_length=MAX_FIELD_CHARACTERS)
    collection_id: str = Field(default="", max_length=36)
    passages_per_cell: _PassageCount = 2
    format: Literal["html", "json", "markdown"] = "html"
    navigation: str = Field(default="{}", max_length=MAX_FIELD_CHARACTERS)

    def worksheet_request(self) -> WorksheetRequest:
        payload: dict[str, object] = {
            "questions": self.questions.replace("\r\n", "\n").split("\n"),
            "passages_per_cell": self.passages_per_cell,
        }
        if self.document_ids != "":
            payload["document_ids"] = json.loads(self.document_ids)
        if self.collection_id != "":
            payload["collection_id"] = self.collection_id
        return WorksheetRequest.model_validate(payload)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Duplicate fields are ambiguous.")
    return result


def _fields(raw: bytes) -> dict[str, str]:
    wire = raw.decode("ascii")
    if re.search(r"%(?![0-9a-fA-F]{2})", wire):
        raise ValueError("Malformed percent encoding.")
    pairs = parse_qsl(
        wire,
        keep_blank_values=True,
        strict_parsing=True,
        encoding="utf-8",
        errors="strict",
        max_num_fields=MAX_FORM_FIELDS,
    )
    if len(dict(pairs)) != len(pairs) or any(
        len(name) > 32 or len(value) > MAX_FIELD_CHARACTERS or "\0" in name + value
        for name, value in pairs
    ):
        raise ValueError("Invalid form fields.")
    return dict(pairs)


async def _body(request: Request) -> bytes:
    lengths = request.headers.getlist("content-length")
    if len(lengths) > 1 or (lengths and (not lengths[0].isascii() or not lengths[0].isdecimal())):
        raise ValueError("Invalid body length.")
    if lengths and int(lengths[0]) > MAX_FORM_BYTES:
        raise WorksheetError("worksheet_form_too_large", "", 413)
    content = bytearray()
    async for chunk in request.stream():
        if len(content) + len(chunk) > MAX_FORM_BYTES:
            raise WorksheetError("worksheet_form_too_large", "", 413)
        content.extend(chunk)
    if lengths and int(lengths[0]) != len(content):
        raise ValueError("Body length mismatch.")
    return bytes(content)


_FAILURES = {
    "worksheet_form_too_large": (413, "Form exceeds 65,536 bytes. No inspection was started."),
    "worksheet_query_too_large": (422, "The form URL exceeds 16,384 bytes."),
    "worksheet_form_encoding": (
        415,
        "Submit application/x-www-form-urlencoded with UTF-8, without content encoding.",
    ),
    "invalid_worksheet_form": (
        422,
        "Use 1-5 nonblank questions (one per line, at most 500 characters each), exactly one "
        "selection (a JSON array of 1-10 document IDs or one collection ID), 1-3 passages, "
        "and html/json/markdown output. Blank lines, unknown/duplicate fields, invalid UTF-8, "
        "malformed percent encoding, invalid return state, and POST query fields are rejected. "
        "No input is echoed and no scope is widened.",
    ),
    "unknown_documents": (422, "Every selected document must exist. No substitute scope was used."),
    "collection_not_found": (404, "The exact saved collection does not exist."),
    "invalid_collection_record": (
        409,
        "The saved collection is invalid. No membership was repaired.",
    ),
    "collection_documents_missing": (409, "The collection references missing corpus documents."),
    "collection_storage_error": (503, "Corpus/collection storage is unavailable, not empty."),
    "worksheet_selection_too_large": (422, "Select at most 10 papers and 50 question/paper cells."),
    "worksheet_documents_changed": (409, "Selected papers disappeared during inspection. Retry."),
    "generative_retrieval": (
        409,
        "Generative retrieval (including LLM-backed HyDE) is configured. It was not invoked "
        "or silently replaced. Use non-generative retrieval for worksheets.",
    ),
    "worksheet_timeout": (504, "The 30-second overall worksheet deadline was exceeded."),
    "retrieval_timeout": (504, "The existing retrieval phase deadline was exceeded."),
    "context_preparation_timeout": (504, "The existing context preparation deadline was exceeded."),
    "worksheet_too_large": (
        413,
        "Complete JSON or Markdown exceeds 262,144 bytes. Reduce questions, papers or passages.",
    ),
    "worksheet_html_too_large": (
        413,
        "Escaped HTML exceeds 1,048,576 bytes. Reduce the selection or choose JSON/Markdown.",
    ),
    "worksheet_html_unrepresentable": (
        409,
        "Text cannot be represented faithfully as UTF-8 HTML, for example embedded NUL. "
        "Use the existing JSON/Python worksheet interface.",
    ),
    "worksheet_failed": (
        500,
        "Inspection or provenance validation failed. No cells or raw diagnostics are shown.",
    ),
    "worksheet_method_not_allowed": (405, "The form accepts only GET and read-only POST."),
}


def _failure(code: str, root_path: str) -> HTMLResponse:
    status, message = _FAILURES.get(code, _FAILURES["worksheet_failed"])
    safe_code = code if code in _FAILURES else "worksheet_failed"
    return _error(
        status,
        safe_code,
        "Worksheet unavailable",
        message + " No partial worksheet was returned.",
        navigation=(
            '<nav aria-label="Worksheet recovery">'
            + _link("reset-worksheet", root_path + "/explore/worksheet", "Start a blank worksheet")
            + _link("back-to-catalog", root_path + "/explore", "Back to current corpus")
            + "</nav>"
        ),
        worksheet=True,
    )


class _WorksheetRoute(APIRoute):
    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None:
        if self.methods is not None and scope["method"] not in self.methods:
            response = _failure(
                "worksheet_method_not_allowed", scope.get("root_path", "").rstrip("/")
            )
            response.headers["Allow"] = "GET, POST"
            await response(scope, receive, send)
            return
        await super().handle(scope, receive, send)


def _form(fields: _Form, navigation: _CatalogReturn, root_path: str) -> Iterator[str]:
    yield (
        '<nav aria-label="Worksheet navigation">'
        + _link(
            "back-to-catalog",
            _url(root_path + "/explore", navigation.parameters()),
            "Back to filtered current corpus",
        )
        + _link("reset-worksheet", root_path + "/explore/worksheet", "Start a blank worksheet")
        + "</nav>"
    )
    yield (
        '<form id="worksheet-form" method="post" enctype="application/x-www-form-urlencoded" '
        f'action="{_escaped(root_path + "/explore/worksheet")}" '
        'accept-charset="UTF-8" autocomplete="off">'
    )
    # JSON escapes keep CR/LF in return-only filters intact through native hidden inputs.
    encoded_navigation = json.dumps(navigation.model_dump(), ensure_ascii=True)
    yield f'<input type="hidden" name="navigation" value="{_escaped(encoded_navigation)}">'
    for name, label, value, limit, help_text in (
        (
            "questions",
            "Literal questions: one per line",
            fields.questions,
            MAX_QUESTIONS_TEXT,
            "1-5 nonblank lines, at most 500 characters each. Spaces are retained; "
            "CRLF separates lines. A question is an input, not a claim.",
        ),
        (
            "document_ids",
            "Selected document IDs (JSON array)",
            fields.document_ids,
            MAX_FIELD_CHARACTERS,
            'Copy 1-10 exact IDs from the catalog as ["first-id", "second-id"]. '
            "The shared API trims outer ID whitespace and keeps first-seen order. "
            "Leave this entirely blank when using a saved collection.",
        ),
    ):
        required = " required" if name == "questions" else ""
        yield (
            f'<div class="worksheet-input"><label for="{name}">{label}</label>'
            f'<textarea id="{name}" name="{name}" maxlength="{limit}"{required} '
            f'aria-describedby="{name}-help" spellcheck="false">'
            + "\n"
            + _escaped(value)
            + f'</textarea><p id="{name}-help">{help_text}</p></div>'
        )
    yield (
        '<div><label for="collection_id">Or one saved collection ID</label>'
        f'<input id="collection_id" name="collection_id" maxlength="36" '
        f'value="{_escaped(fields.collection_id)}" aria-describedby="scope-help">'
        '<p id="scope-help">Choose IDs OR a collection, never both. An empty selection '
        "fails; it never means the whole corpus. No collection editor is added.</p></div>"
        '<div><label for="passages_per_cell">Passages per cell</label>'
        f'<input id="passages_per_cell" name="passages_per_cell" type="number" min="1" max="3" '
        f'value="{fields.passages_per_cell}" required inputmode="numeric"></div>'
        '<div><label for="format">Output for this submission</label>'
        '<select id="format" name="format">'
    )
    for value, label in (
        ("html", "Browser HTML"),
        ("json", "JSON download"),
        ("markdown", "Markdown"),
    ):
        selected = " selected" if fields.format == value else ""
        yield f'<option value="{value}"{selected}>{label}</option>'
    yield (
        '</select></div><div class="actions"><button type="submit">Inspect selected papers</button>'
        '</div><p class="worksheet-input">Each submission builds a new worksheet. '
        "Export submissions rerun retrieval, not a saved/frozen result. "
        "Up to 50 cells, 30 seconds, 800 excerpt characters per passage. "
        "Nothing is saved on the server.</p></form>"
    )


def _results(
    result: ResearchWorksheet, navigation: _CatalogReturn, root_path: str
) -> Iterator[str]:
    yield (
        '<section aria-labelledby="worksheet-result">'
        '<h2 id="worksheet-result">Research worksheet</h2>'
    )
    for warning in result.warnings:
        yield '<p class="notice">' + _escaped(warning) + "</p>"
    if result.collection_id is not None:
        yield "<p>Resolved collection ID (not a frozen corpus):</p>"
        yield _identity(result.collection_id, "collection-id")
    yield "<h3>Selected papers</h3>"
    for number, document in enumerate(result.documents, 1):
        yield f"<p>Paper {number}</p>" + _identity(document.document_id, "document-id")
        yield _link(
            f"inspect-paper-{number}",
            _url(
                root_path + "/explore/document",
                {
                    "document_id": document.document_id,
                    "limit": navigation.catalog_limit,
                    "source": navigation.source,
                    "title": navigation.title,
                    "catalog_cursor": navigation.catalog_cursor,
                },
            ),
            "Inspect this exact paper in the current corpus",
        )
    yield (
        '<div class="worksheet-grid" tabindex="0" role="region" aria-label="Question by paper">'
        "<table><caption>Passage counts, not evidence strength</caption>"
        '<thead><tr><th scope="col">Question</th>'
    )
    for number in range(1, len(result.documents) + 1):
        yield f'<th scope="col">Paper {number}</th>'
    yield "</tr></thead><tbody>"
    for question_number, row in enumerate(result.rows, 1):
        yield f'<tr><th scope="row">Question {question_number}</th>'
        for paper_number, cell in enumerate(row.cells, 1):
            yield (
                f'<td><a href="#cell-{question_number}-{paper_number}">{cell.status}: '
                f"{len(cell.passages)} shown</a>; {cell.passages_omitted} omitted</td>"
            )
        yield "</tr>"
    yield "</tbody></table></div>"
    for question_number, row in enumerate(result.rows, 1):
        yield f"<h2>Question {question_number}</h2>" + _literal(row.question, "worksheet-question")
        for paper_number, cell in enumerate(row.cells, 1):
            cell_id = f"cell-{question_number}-{paper_number}"
            yield f'<article class="worksheet-cell" id="{cell_id}"><h3>Paper {paper_number}</h3>'
            yield _identity(cell.document_id, "cell-document-id")
            yield (
                f'<p class="cell-status">{cell.status}; '
                f"preview returned {cell.preview_source_count}; "
                f"showing {len(cell.passages)}; passages omitted: {cell.passages_omitted}.</p>"
            )
            if not cell.passages:
                yield "<p>No passages returned, not a finding of no scientific evidence.</p>"
            yield "<details><summary>Effective preview configuration</summary>"
            yield _literal(cell.configuration.model_dump_json(indent=2), "preview-configuration")
            yield "</details>"
            for index, passage in enumerate(cell.passages, 1):
                yield f"<h3>Returned passage {index}</h3><dl><dt>Document ID (JSON)</dt><dd>"
                yield _identity(passage.document_id, "passage-document-id")
                yield "</dd><dt>Chunk ID (JSON)</dt><dd>" + _identity(passage.chunk_id, "chunk-id")
                yield (
                    f"</dd><dt>Rank / score</dt><dd>{passage.rank} / "
                    f'<span class="passage-score">{passage.score}</span></dd>'
                    "<dt>Retriever</dt><dd>"
                )
                yield _literal(passage.retriever, "retriever") + "</dd><dt>Path (JSON)</dt><dd>"
                yield _literal(json.dumps(passage.path, ensure_ascii=True), "retrieval-path")
                for label, text, css_class, truncated in (
                    ("Title", passage.title, "passage-title", passage.title_truncated),
                    ("Source", passage.source, "passage-source", passage.source_truncated),
                ):
                    yield f"</dd><dt>{label}</dt><dd>" + _literal(text, css_class)
                    yield f"</dd><dt>{label} truncated</dt><dd>{str(truncated).lower()}"
                yield (
                    "</dd><dt>Excerpt truncated</dt>"
                    f'<dd class="excerpt-truncated">{str(passage.excerpt_truncated).lower()}</dd>'
                    f"<dt>Full text characters</dt><dd>{passage.text_characters}</dd>"
                    "<dt>Full text SHA-256</dt><dd>"
                    + _literal(passage.text_sha256, "text-digest")
                    + "</dd></dl>"
                )
                yield _literal(passage.excerpt, "passage-text")
                yield (
                    "<p>"
                    + _link(
                        f"source-{question_number}-{paper_number}-{index}",
                        _url(
                            root_path + "/explore/context",
                            {
                                "document_id": passage.document_id,
                                "chunk_id": passage.chunk_id,
                                "limit": navigation.catalog_limit,
                                "source": navigation.source,
                                "title": navigation.title,
                                "catalog_cursor": navigation.catalog_cursor,
                            },
                        ),
                        "Read current source context (exact IDs, not frozen text)",
                    )
                    + "</p>"
                )
            yield "</article>"
    yield "</section>"


router = APIRouter(route_class=_WorksheetRoute, tags=["Research worksheet browser"])


@router.get(
    "/explore/worksheet",
    response_class=HTMLResponse,
)
@router.post(
    "/explore/worksheet",
    response_class=HTMLResponse,
    responses={
        200: {"description": "Native form/worksheet, or an existing JSON/Markdown attachment"},
        404: {"description": "Exact collection not found"},
        405: {"description": "Only GET and read-only POST are supported"},
        409: {"description": "Invalid/stale scope, generative retrieval or unrepresentable HTML"},
        413: {"description": "Request body or complete serialized output exceeds its fixed bound"},
        415: {"description": "Only UTF-8 URL-encoded forms without content encoding"},
        422: {"description": "Invalid, empty, unknown or ambiguous input/selection"},
        500: {"description": "Inspection/provenance failure, never partial output"},
        503: {"description": "Corpus/collection storage unavailable"},
        504: {"description": "Worksheet or preview deadline exceeded"},
    },
)
async def explore_worksheet(request: Request) -> Response:
    """GET opens controls only; POST reuses one complete build without saving anything."""
    root_path = request.scope.get("root_path", "").rstrip("/")
    try:
        raw_query = request.scope.get("query_string", b"")
        if len(raw_query) > MAX_QUERY_BYTES:
            raise WorksheetError("worksheet_query_too_large", "", 422)
        body = await _body(request)
        if request.method == "GET":
            if body:
                raise ValueError("GET form opening does not accept a body.")
            options = _OpenForm.model_validate(_fields(raw_query))
            navigation = _CatalogReturn.model_validate(
                options.model_dump(exclude={"document_id", "collection_id"})
            )
            fields = _Form(
                document_ids=json.dumps([options.document_id], ensure_ascii=True)
                if options.document_id is not None
                else "",
                collection_id=options.collection_id or "",
            )
            return _page(
                "Research evidence worksheet", _form(fields, navigation, root_path), worksheet=True
            )
        content_types = request.headers.getlist("content-type")
        if (
            len(content_types) != 1
            or not re.fullmatch(
                r'application/x-www-form-urlencoded(?:\s*;\s*charset=(?:utf-8|"utf-8"))?',
                content_types[0].strip().lower(),
            )
            or "content-encoding" in request.headers
        ):
            raise WorksheetError("worksheet_form_encoding", "", 415)
        if raw_query:
            raise ValueError("POST controls belong only in the body.")
        fields = _Form.model_validate(_fields(body))
        navigation = _CatalogReturn.model_validate(
            json.loads(fields.navigation, object_pairs_hook=_unique_object)
        )
        payload = fields.worksheet_request()
    except (WorksheetError, CollectionError) as exc:
        return _failure(exc.code, root_path)
    except _PageTooLargeError:
        return _failure("worksheet_html_too_large", root_path)
    except (ValueError, RecursionError):
        return _failure("invalid_worksheet_form", root_path)
    try:
        container: AppContainer = request.app.state.container
        result = await container.worksheets.build(payload)
        if fields.format != "html":
            markdown = fields.format == "markdown"
            return Response(
                content=result.to_markdown() if markdown else result.to_json(),
                media_type="text/markdown" if markdown else "application/json",
                headers={
                    **_HEADERS,
                    "Content-Disposition": 'attachment; filename="worksheet.'
                    + ("md" if markdown else "json")
                    + '"',
                },
            )

        def fragments() -> Iterator[str]:
            yield from _form(fields, navigation, root_path)
            yield from _results(result, navigation, root_path)

        return _page("Research evidence worksheet", fragments(), worksheet=True)
    except (WorksheetError, CollectionError) as exc:
        return _failure(exc.code, root_path)
    except _PageTooLargeError:
        return _failure("worksheet_html_too_large", root_path)
    except (_UnrepresentableTextError, UnicodeError):
        return _failure("worksheet_html_unrepresentable", root_path)
