"""Explicit, bounded native forms over the existing generation-free retrieval endpoint."""

import json
import re
from collections.abc import Iterator
from typing import Literal, Self

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.types import Receive, Scope, Send

from agent.retrieval_preview import RetrievalPreview
from api.explorer import (
    _collection_link,
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
from api.retrieval import RetrievalRequest, retrieve
from api.worksheet_explorer import _CatalogReturn
from storage.paper_collections import CollectionId
from storage.source_context import ContextDocumentIdentity

PATH = "/explore/retrieve"
MAX_FORM_BYTES = 65_536
MAX_QUERY_BYTES = 65_536
MAX_QUERY_CHARACTERS = 2_000
MAX_FIELD_CHARACTERS = 16_384


class _OpenForm(_CatalogReturn):
    document_id: ContextDocumentIdentity | None = None
    collection_id: CollectionId | None = None

    @model_validator(mode="after")
    def unambiguous(self) -> Self:
        if self.document_id is not None and self.collection_id is not None:
            raise ValueError("Prefill one exact selection only.")
        return self


class _Form(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str = Field(max_length=MAX_QUERY_CHARACTERS)
    scope: Literal["all", "documents", "collection"]
    document_ids: str = Field(default="", max_length=MAX_FIELD_CHARACTERS)
    collection_id: str = Field(default="", max_length=36)
    max_chunks_per_document: str = Field(default="", max_length=19)
    min_evidence_documents: str = Field(default="", max_length=19)
    near_duplicate_threshold: str = Field(default="", max_length=32)
    navigation: str = Field(default="{}", max_length=MAX_FIELD_CHARACTERS)

    def payload(self) -> RetrievalRequest:
        payload: dict[str, object] = {"query": self.query}
        if self.scope == "documents":
            if self.collection_id or not self.document_ids:
                raise ValueError("Select document IDs only.")
            payload["document_ids"] = json.loads(self.document_ids)
        elif self.scope == "collection":
            if self.document_ids or not self.collection_id:
                raise ValueError("Select a collection only.")
            payload["collection_id"] = self.collection_id
        elif self.document_ids or self.collection_id:
            raise ValueError("Whole corpus requires both selection fields to be blank.")
        for name in ("max_chunks_per_document", "min_evidence_documents"):
            value = getattr(self, name)
            if value != "":
                payload[name] = unsigned_decimal(value)
        if self.near_duplicate_threshold != "":
            # Native number controls allow .9 and 00.9, unlike JSON numeric syntax.
            if not re.fullmatch(
                r"-?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?",
                self.near_duplicate_threshold,
            ):
                raise ValueError("Use a native decimal number for the threshold.")
            payload["near_duplicate_threshold"] = float(self.near_duplicate_threshold)
        # JSON-escaped IDs must be real Unicode before reaching SQLite or retrieval.
        json.dumps(payload, ensure_ascii=False).encode("utf-8")
        return RetrievalRequest.model_validate(payload)


_FAILURES = {
    "preview_origin": (403, "Use one exact same-origin browser submission."),
    "preview_encoding": (415, "Submit a UTF-8 URL-encoded form without content encoding."),
    "preview_form_too_large": (413, "The complete form exceeds 65,536 bytes."),
    "preview_query_too_large": (422, "The form-opening URL exceeds 65,536 bytes."),
    "invalid_preview_form": (
        422,
        "Use a nonblank question of at most 2,000 characters, an explicit scope, and valid "
        "evidence controls. Select 1-100 JSON document IDs OR one collection, or explicitly "
        "choose Whole corpus with both fields blank. Unknown/duplicate fields, malformed "
        "encoding, POST query controls and ambiguous selections are rejected.",
    ),
    "collection_not_found": (404, "The exact saved collection does not exist."),
    "invalid_collection_record": (409, "The saved collection is invalid; it was not repaired."),
    "collection_documents_missing": (409, "The collection references missing corpus documents."),
    "collection_storage_error": (503, "Collection/corpus storage is unavailable, not empty."),
    "generative_retrieval": (
        409,
        "LLM-backed retrieval is configured. It was not invoked or silently replaced.",
    ),
    "planning_failed": (500, "Planning failed; no retrieval result is available."),
    "retrieval_failed": (500, "Retrieval or scope validation failed."),
    "context_preparation_failed": (500, "Context preparation or evidence validation failed."),
    "planning_timeout": (504, "The planner reported a timeout."),
    "retrieval_timeout": (504, "The configured retrieval deadline was exceeded."),
    "context_preparation_timeout": (
        504,
        "The configured context-preparation deadline was exceeded.",
    ),
    "preview_html_too_large": (
        413,
        "Complete escaped HTML exceeds 1,048,576 bytes. Reduce the selection or use /retrieve "
        "JSON. No passages were silently truncated.",
    ),
    "preview_html_unrepresentable": (
        409,
        "Text cannot be represented faithfully as UTF-8 HTML (for example embedded NUL). "
        "Use the existing JSON/Python preview interface.",
    ),
    "preview_failed": (500, "Retrieval inspection failed."),
    "preview_method_not_allowed": (405, "This workbench accepts GET and read-only POST only."),
}


def _failure(code: str, root_path: str) -> HTMLResponse:
    safe_code = code if code in _FAILURES else "preview_failed"
    status, message = _FAILURES[safe_code]
    return _error(
        status,
        safe_code,
        "Retrieval preview unavailable",
        message + " No partial preview was returned. No scope was widened.",
        navigation=(
            '<nav aria-label="Preview recovery">'
            + _link("reset-preview", root_path + PATH, "Start a blank preview")
            + _link("back-to-catalog", root_path + "/explore", "Back to current corpus")
            + "</nav>"
        ),
        retrieval=True,
    )


class _PreviewRoute(APIRoute):
    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None:
        if self.methods is not None and scope["method"] not in self.methods:
            response = _failure(
                "preview_method_not_allowed", scope.get("root_path", "").rstrip("/")
            )
            response.headers["Allow"] = "GET, POST"
            await response(scope, receive, send)
            return
        await super().handle(scope, receive, send)


def _form(fields: _Form, navigation: _CatalogReturn, root_path: str) -> Iterator[str]:
    yield (
        '<nav aria-label="Preview navigation">'
        + _link(
            "back-to-catalog",
            _url(root_path + "/explore", navigation.parameters()),
            "Back to filtered current corpus",
        )
        + _collection_link(navigation.collection_return, root_path)
        + _link("reset-preview", root_path + PATH, "Start a blank preview")
        + "</nav>"
        '<form id="retrieval-form" method="post" enctype="application/x-www-form-urlencoded" '
        f'action="{_escaped(root_path + PATH)}" accept-charset="UTF-8" autocomplete="off">'
        '<input type="hidden" name="navigation" '
        f'value="{_escaped(navigation.model_dump_json())}">'
    )
    for name, label, value, limit, help_text in (
        (
            "query",
            "Research question",
            fields.query,
            MAX_QUERY_CHARACTERS,
            "A nonblank literal question, at most 2,000 characters. Nothing runs until submission.",
        ),
        (
            "document_ids",
            "Exact document IDs (JSON array)",
            fields.document_ids,
            MAX_FIELD_CHARACTERS,
            'Use 1-100 IDs, for example ["paper-a", "paper-b"]. The existing API trims outer ID '
            "whitespace and deduplicates in first-seen order; the resolved scope is shown below. "
            "Leave blank for Whole corpus or Collection.",
        ),
    ):
        required = " required" if name == "query" else ""
        yield (
            f'<div class="worksheet-input"><label for="{name}">{label}</label>'
            f'<textarea id="{name}" name="{name}" maxlength="{limit}"{required} '
            f'aria-describedby="{name}-help" spellcheck="false">\n{_escaped(value)}</textarea>'
            f'<p id="{name}-help">{help_text}</p></div>'
        )
    yield (
        '<div><label for="scope">Explicit retrieval scope</label>'
        '<select id="scope" name="scope" required aria-describedby="scope-help">'
    )
    for value, label in (
        ("all", "Whole corpus"),
        ("documents", "Document IDs"),
        ("collection", "Saved collection (all members)"),
    ):
        selected = " selected" if fields.scope == value else ""
        yield f'<option value="{value}"{selected}>{label}</option>'
    yield (
        '</select><p id="scope-help">Empty document or collection selections fail; they never '
        "mean the whole corpus. Catalog filters are navigation, not retrieval scope.</p></div>"
        '<div><label for="collection_id">Saved collection ID</label>'
        '<input id="collection_id" name="collection_id" maxlength="36" '
        f'value="{_escaped(fields.collection_id)}" aria-describedby="collection-help">'
        '<p id="collection-help">All current members, not only human includes. '
        "Leave document IDs blank. Use the JSON API for revision-bound screened "
        "selections.</p></div>"
    )
    for name, label in (
        ("max_chunks_per_document", "Maximum passages per document"),
        ("min_evidence_documents", "Minimum distinct documents"),
    ):
        yield (
            f'<div><label for="{name}">{label}</label>'
            f'<input id="{name}" name="{name}" type="number" min="1" max="50" step="1" '
            f'value="{_escaped(getattr(fields, name))}" aria-describedby="policy-help"></div>'
        )
    yield (
        '<div><label for="near_duplicate_threshold">Near-duplicate threshold</label>'
        '<input id="near_duplicate_threshold" name="near_duplicate_threshold" type="number" '
        'min="0" max="1" step="any" '
        f'value="{_escaped(fields.near_duplicate_threshold)}" aria-describedby="policy-help"></div>'
        '<p id="policy-help" class="worksheet-input">Blank means unconfigured. Counts are integers '
        "1-50; the lexical overlap threshold is greater than 0 and at most 1. Preparation applies "
        "reranking, optional collapse, quota, then the minimum diagnostic. No candidate refill. "
        "A failed minimum retains passages; a passing count is not scientific support.</p>"
        '<div class="actions"><button type="submit">Preview prepared evidence</button></div>'
        '<p class="worksheet-input">Each submission reruns local retrieval and preparation. '
        "No answer, saved run, collection edit or agent event is created.</p></form>"
    )


def _results(preview: RetrievalPreview, root_path: str) -> Iterator[str]:
    observation = preview.plan.observation
    yield (
        '<section aria-labelledby="preview-results"><h2 id="preview-results">Prepared evidence</h2>'
    )
    yield "<h3>Resolved document scope (JSON; null = whole corpus)</h3>"
    yield _literal(json.dumps(observation.document_ids, ensure_ascii=True), "resolved-scope")
    yield "<h3>Selected evidence policy (JSON; null = unconfigured)</h3>"
    yield _literal(
        observation.evidence_policy.model_dump_json(indent=2)
        if observation.evidence_policy
        else "null",
        "evidence-policy",
    )
    assessment = preview.evidence_assessment
    if assessment is None:
        yield "<p>No minimum-document requirement was requested.</p>"
    else:
        label = "Minimum met" if assessment.passed else "Minimum not met"
        yield (
            f'<p class="notice">{label}: {assessment.observed_documents} distinct documents '
            f"prepared; {assessment.required_documents} required. "
            "This is a count diagnostic, not relevance, independence or answerability.</p>"
        )
        yield _literal(assessment.model_dump_json(indent=2), "evidence-assessment")
    yield f"<p>{len(preview.sources)} complete passages in actual prepared order.</p>"
    if not preview.sources:
        yield (
            '<p class="notice">No passages returned for this explicit scope. Unknown document '
            "IDs match nothing. This is not a finding of no scientific evidence.</p>"
        )
    yield "<details><summary>Actual plan and effective limits</summary>"
    yield _literal(preview.plan.model_dump_json(indent=2), "preview-plan")
    yield _literal(preview.configuration.model_dump_json(indent=2), "preview-configuration")
    yield _literal(preview.capture_limits.model_dump_json(indent=2), "capture-limits")
    yield "</details><h3>Exact prepared context SHA-256</h3>"
    yield _literal(preview.context_sha256, "context-sha256")
    yield "<details><summary>Complete prepared context (not a generated answer)</summary>"
    yield _literal(preview.context, "prepared-context")
    yield "</details>"
    for source in preview.sources:
        yield (
            f'<article class="prepared-source" id="source-{source.rank}">'
            f"<h3>Prepared passage {source.rank}</h3><dl><dt>Document ID (JSON)</dt><dd>"
            + _identity(source.chunk.document_id, "source-document-id")
            + "</dd><dt>Chunk ID (JSON)</dt><dd>"
            + _identity(source.chunk.chunk_id, "chunk-id")
            + f"</dd><dt>Rank / score</dt><dd>{source.rank} / {source.score!r}</dd>"
        )
        for label, value, css_class in (
            ("Title", source.chunk.title, "source-title"),
            ("Source label", source.chunk.source, "source-label"),
            ("Retriever", source.retriever, "source-retriever"),
            ("Retrieval path (JSON)", json.dumps(source.path, ensure_ascii=True), "retrieval-path"),
            ("Passage SHA-256", source.text_sha256, "text-sha256"),
            (
                "Metadata (JSON)",
                json.dumps(source.chunk.metadata, ensure_ascii=True),
                "source-metadata",
            ),
        ):
            yield f"<dt>{label}</dt><dd>" + _literal(value, css_class) + "</dd>"
        yield "</dl>" + _literal(source.chunk.text, "passage-text")
        yield _link(
            f"preview-source-{source.rank}",
            _url(
                root_path + "/explore/context",
                {"document_id": source.chunk.document_id, "chunk_id": source.chunk.chunk_id},
            ),
            "Read current source context (exact IDs; not frozen preview text)",
        )
        yield "</article>"
    yield "</section>"


router = APIRouter(route_class=_PreviewRoute, tags=["Retrieval preview browser"])


@router.get(PATH, response_class=HTMLResponse)
@router.post(
    PATH,
    response_class=HTMLResponse,
    responses={
        403: {"description": "Not an exact same-origin form submission"},
        404: {"description": "Unknown saved collection"},
        405: {"description": "Only GET and read-only POST are supported"},
        409: {"description": "Broken collection, generative retrieval or unrepresentable HTML"},
        413: {"description": "Bounded form or complete escaped HTML is too large"},
        415: {"description": "Unsupported form encoding"},
        422: {"description": "Invalid, empty or ambiguous input"},
        500: {"description": "Planning, retrieval or preparation failed; no partial preview"},
        503: {"description": "Collection storage unavailable"},
        504: {"description": "Existing preview deadline exceeded"},
    },
)
async def explore_retrieval(request: Request) -> HTMLResponse:
    root_path = request.scope.get("root_path", "").rstrip("/")
    raw_query = request.scope.get("query_string", b"")
    if len(raw_query) > MAX_QUERY_BYTES:
        return _failure("preview_query_too_large", root_path)
    if request.method == "POST":
        try:
            require_same_origin(request)
        except ValueError:
            return _failure("preview_origin", root_path)
        try:
            require_urlencoded(request)
        except ValueError:
            return _failure("preview_encoding", root_path)
    try:
        body = await read_body(request, max_bytes=MAX_FORM_BYTES)
        if (request.method == "GET" and body) or (request.method == "POST" and raw_query):
            raise ValueError("GET accepts URL controls only; POST accepts body controls only.")
        parsed = parse_fields(
            raw_query if request.method == "GET" else body,
            max_fields=8,
            max_field_characters=MAX_FIELD_CHARACTERS,
        )
        # GET identities become escaped JSON controls, so embedded NUL can survive losslessly.
        if request.method == "POST" and any("\0" in value for value in parsed.values()):
            raise ValueError("Literal NUL cannot be rendered in HTML.")
        if request.method == "GET":
            options = _OpenForm.model_validate(parsed)
            navigation = _CatalogReturn.model_validate(
                options.model_dump(exclude={"document_id", "collection_id"})
            )
            fields = _Form(
                query="",
                scope="documents"
                if options.document_id is not None
                else "collection"
                if options.collection_id is not None
                else "all",
                document_ids=json.dumps([options.document_id], ensure_ascii=True)
                if options.document_id is not None
                else "",
                collection_id=options.collection_id or "",
            )
            return _page(
                "Retrieval preview workbench", _form(fields, navigation, root_path), retrieval=True
            )
        fields = _Form.model_validate(parsed)
        navigation = _CatalogReturn.model_validate(
            json.loads(fields.navigation, object_pairs_hook=_return_pairs)
        )
        payload = fields.payload()
    except FormTooLargeError:
        return _failure("preview_form_too_large", root_path)
    except _PageTooLargeError:
        return _failure("preview_html_too_large", root_path)
    except (ValueError, RecursionError):
        return _failure("invalid_preview_form", root_path)
    try:
        # The JSON endpoint owns scope resolution, runner arguments and operational errors.
        preview = await retrieve(request, payload, Response())

        def fragments() -> Iterator[str]:
            yield from _form(fields, navigation, root_path)
            yield from _results(preview, root_path)

        return _page("Retrieval preview workbench", fragments(), retrieval=True)
    except HTTPException as exc:
        detail = exc.detail
        code = detail.get("code") if isinstance(detail, dict) else None
        return _failure(code if isinstance(code, str) else "preview_failed", root_path)
    except _PageTooLargeError:
        return _failure("preview_html_too_large", root_path)
    except (_UnrepresentableTextError, UnicodeError):
        return _failure("preview_html_unrepresentable", root_path)
