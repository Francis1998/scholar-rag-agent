"""Read-only native opinions and exact quote navigation over frozen run evidence."""

import json
from collections.abc import Iterator
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse

from agent.evidence import EvidenceBundle, EvidenceSource
from agent.review_models import (
    DEFAULT_REVIEW_LIMIT,
    ReviewHistoryPage,
    invalid_review_record,
)
from api.dependencies import AppContainer
from api.explorer import _escaped, _identity, _link, _literal, _page, _pagination, _url
from api.run_explorer import (
    _RESPONSES,
    _annotation_composer_url,
    _HistoryCursor,
    _HistoryLimit,
    _navigation,
    _RunExplorerRoute,
    _RunParameters,
)
from storage.evidence_annotations import AnnotationPage, SavedEvidenceAnnotation
from storage.run_history import DEFAULT_PAGE_SIZE


class _NotebookParameters(_RunParameters):
    review_limit: _HistoryLimit = DEFAULT_REVIEW_LIMIT
    review_cursor: _HistoryCursor | None = None
    annotation_limit: _HistoryLimit = DEFAULT_PAGE_SIZE
    annotation_cursor: _HistoryCursor | None = None

    def notebook_parameters(self) -> dict[str, str | int | None]:
        return {
            "run_id": self.run_id,
            **self.parameters(),
            "review_limit": self.review_limit,
            "review_cursor": self.review_cursor,
            "annotation_limit": self.annotation_limit,
            "annotation_cursor": self.annotation_cursor,
        }

    def evidence_url(self, root_path: str) -> str:
        return _url(root_path + "/explore/run", {"run_id": self.run_id, **self.parameters()})


class _AnnotationParameters(_NotebookParameters):
    annotation_id: UUID


def _detail_navigation(options: _NotebookParameters, root_path: str) -> str:
    links = _link(
        "back-to-evidence", options.evidence_url(root_path), "Read saved answer / evidence"
    )
    if isinstance(options, _AnnotationParameters):
        links += _link(
            "back-to-notebook",
            _url(root_path + "/explore/run/notebook", options.notebook_parameters()),
            "Back to this notebook page",
        )
    return '<nav aria-label="Human-review navigation">' + links + "</nav>"


def _prepare(request: Request, options: _NotebookParameters) -> str:
    root_path: str = request.scope.get("root_path", "").rstrip("/")
    request.state.saved_run_options = options
    request.state.saved_run_detail_navigation = _detail_navigation(options, root_path)
    return root_path


_NOTICE = (
    '<p class="notice">Human opinions, not scientific validation. Accepted, needs_revision '
    "and rejected are recorded answer-review decisions, separate from model grounding. "
    "Annotations locate exact quoted spans; they do not establish support or entailment. "
    "No authenticated reviewer identity is recorded.</p>"
    "<p>These records are append-only. Different opinions coexist; no consensus, latest verdict "
    "or review count for the whole run is inferred from a page. Nothing can be edited here.</p>"
)


def _field(label: str, value: str, css_class: str) -> str:
    return f"<dt>{label}</dt><dd>{_literal(value, css_class)}</dd>"


def _timestamp(value: str) -> str:
    return (
        "<dt>Recorded at (UTC)</dt><dd>"
        f'<time datetime="{_escaped(value)}">{_escaped(value)}</time></dd>'
    )


def _paging(
    options: _NotebookParameters,
    root_path: str,
    kind: Literal["review", "annotation"],
    next_cursor: int | None,
) -> Iterator[str]:
    params = options.notebook_parameters()
    path = root_path + "/explore/run/notebook"
    limit_key, cursor_key = kind + "_limit", kind + "_cursor"
    label = "Answer reviews" if kind == "review" else "Evidence annotations"
    if any(
        isinstance(value, str) and any(character in value for character in "\0\r\n")
        for value in params.values()
    ):
        yield (
            "<p>Exact IDs or return filters contain controls that native inputs cannot preserve. "
            "Use the exact pagination and return links; page-size forms are unavailable.</p>"
        )
    else:
        yield (
            f'<form method="get" action="{_escaped(path)}" autocomplete="off">'
            f'<div><label for="{limit_key}">{label} per page</label>'
            f'<input id="{limit_key}" name="{limit_key}" type="number" min="1" max="100" '
            f'value="{params[limit_key]}" required inputmode="numeric"></div>'
        )
        for name, value in params.items():
            if value is not None and name not in {limit_key, cursor_key}:
                yield f'<input type="hidden" name="{name}" value="{_escaped(str(value))}">'
        yield (
            '<div class="actions"><button type="submit">Apply page size</button></div></form>'
            "<p>Changing this size restarts only this history; "
            "the other history stays selected.</p>"
        )
    yield _pagination(
        _url(path, {**params, cursor_key: None}),
        _url(path, {**params, cursor_key: next_cursor}) if next_cursor is not None else None,
        prefix=kind + "s-",
        label=label + " pagination",
    )


def _reviews(
    page: ReviewHistoryPage,
    bundle: EvidenceBundle,
    options: _NotebookParameters,
    root_path: str,
) -> Iterator[str]:
    yield '<section id="reviews"><h2>Append-only answer reviews</h2>'
    yield (
        f"<p>{len(page.reviews)} reviews on this page (limit {options.review_limit}). "
        "Newest sequence first; the cursor selects strictly older sequences.</p>"
    )
    sources = {source.chunk.chunk_id: source for source in bundle.snapshot.sources}
    for review in page.reviews:
        yield f'<article id="review-{review.sequence}"><h3>Review sequence {review.sequence}</h3>'
        yield "<dl>" + _field("Decision (human opinion)", review.decision.value, "review-decision")
        yield _field("Review UUID (not a person)", str(review.review_id), "review-id")
        yield _field("Sequence", str(review.sequence), "review-sequence")
        yield _field("Record schema", review.schema_version, "record-schema")
        yield _timestamp(review.created_at.isoformat()) + "</dl><h4>Exact recorded comment</h4>"
        yield _literal(review.comment, "review-comment")
        yield "<h4>Cited frozen passages</h4>"
        if not review.cited_chunk_ids:
            yield "<p>This review cites no frozen passages.</p>"
        for number, chunk_id in enumerate(review.cited_chunk_ids, 1):
            source = sources.get(chunk_id)
            if source is None:
                raise invalid_review_record()
            yield _link(
                f"review-source-{review.sequence}-{number}",
                options.evidence_url(root_path) + f"#source-{source.rank}",
                f"Frozen evidence {source.rank}",
            )
            yield "<dl><dt>Document ID (JSON)</dt><dd>"
            yield _identity(source.chunk.document_id, "document-id") + "</dd>"
            yield "<dt>Chunk ID (JSON)</dt><dd>" + _identity(chunk_id, "chunk-id") + "</dd></dl>"
        yield "</article>"
    if not page.reviews:
        yield (
            "<p>No answer reviews remain after this cursor.</p>"
            if options.review_cursor is not None
            else "<p>No answer reviews recorded for this completed run.</p>"
        )
    yield from _paging(options, root_path, "review", page.next_cursor)
    yield "</section>"


def _annotation_record(saved: SavedEvidenceAnnotation) -> Iterator[str]:
    yield "<dl>" + _field(
        "Annotation UUID (not a person)", str(saved.annotation_id), "annotation-id"
    )
    yield _field("Sequence", str(saved.sequence), "annotation-sequence")
    yield _field("Record schema", saved.schema_version, "record-schema")
    yield _timestamp(saved.created_at.isoformat())
    yield "<dt>Document ID (JSON)</dt><dd>" + _identity(saved.document_id, "document-id") + "</dd>"
    yield "<dt>Chunk ID (JSON)</dt><dd>" + _identity(saved.chunk_id, "chunk-id") + "</dd>"
    yield _field("Frozen text SHA-256", saved.source_text_sha256, "source-digest")
    yield _field("Half-open Unicode code-point span", f"[{saved.start}, {saved.end})", "quote-span")
    yield "</dl><h4>Exact recorded quote</h4>" + _literal(saved.quote, "annotation-quote")
    yield "<h4>Exact recorded note</h4>" + _literal(saved.note, "annotation-note")


def _annotations(
    page: AnnotationPage, options: _NotebookParameters, root_path: str
) -> Iterator[str]:
    yield '<section id="annotations"><h2>Append-only evidence annotations</h2>'
    yield (
        f"<p>{len(page.annotations)} annotations on this page (limit {options.annotation_limit}). "
        "Newest sequence first; the cursor selects strictly older sequences.</p>"
    )
    for saved in page.annotations:
        yield (
            f'<article id="annotation-{saved.sequence}">'
            f"<h3>Annotation sequence {saved.sequence}</h3>"
        )
        yield _link(
            f"annotation-source-{saved.sequence}",
            _url(
                root_path + "/explore/run/annotation",
                {**options.notebook_parameters(), "annotation_id": str(saved.annotation_id)},
            )
            + "#annotation-quote",
            "Follow exact quote in frozen source",
        )
        yield from _annotation_record(saved)
        yield "</article>"
    if not page.annotations:
        yield (
            "<p>No evidence annotations remain after this cursor.</p>"
            if options.annotation_cursor is not None
            else "<p>No evidence annotations recorded for this completed run.</p>"
        )
    yield from _paging(options, root_path, "annotation", page.next_cursor)
    yield "</section>"


def _notebook(
    bundle: EvidenceBundle,
    reviews: ReviewHistoryPage,
    annotations: AnnotationPage,
    options: _NotebookParameters,
    root_path: str,
) -> Iterator[str]:
    yield _navigation(options, root_path) + _detail_navigation(options, root_path)
    yield _NOTICE
    yield '<nav aria-label="Complete notebook downloads">'
    for format, label in (("json", "JSON"), ("markdown", "Markdown")):
        url = _url(
            root_path + "/explore/run/notebook/export",
            {"run_id": bundle.run_id, "format": format},
        )
        yield (
            f'<a id="download-notebook-{format}" href="{_escaped(url)}" download>'
            f"Download complete notebook ({label})</a>"
        )
    yield (
        "</nav><p>Downloads include frozen evidence and both complete histories in one read-only "
        "snapshot, not just the visible pages. Fixed record and byte limits fail explicitly "
        "without partial exports. Review sensitive source text and human notes before sharing.</p>"
    )
    yield (
        "<p>The answer, reviews and annotation history are independently read and paginated, "
        "not one shared snapshot. Newly appended opinions require restarting that history. "
        "Following an annotation reads that record and its frozen source together.</p>"
        '<nav aria-label="Notebook sections"><a href="#saved-answer">Saved answer</a>'
        '<a href="#reviews">Answer reviews</a><a href="#annotations">Evidence annotations</a></nav>'
        '<section id="saved-answer"><h2>Completed saved answer</h2><dl>'
        "<dt>Run ID (JSON)</dt><dd>"
    )
    yield _identity(bundle.run_id, "run-id") + "</dd><dt>Agent ID (JSON)</dt><dd>"
    yield _identity(bundle.agent_id, "agent-id") + "</dd>"
    yield _field("Captured status", bundle.status, "captured-status")
    yield _field("Completed (UTC)", bundle.completed_at, "completed-at")
    yield _field("Recorded ungrounded flag", str(bundle.answer.ungrounded).lower(), "ungrounded")
    yield (
        "</dl><p>Model grounding is recorded separately and is never changed by these opinions.</p>"
    )
    yield _literal(bundle.answer.answer, "answer-text") + "</section>"
    yield '<section id="author-annotation"><h2>Add an exact-source note</h2>'
    yield (
        "<p>Choose a frozen passage, enter explicit code-point offsets and your note, "
        "then preview before saving. Opening a composer does not create an annotation.</p>"
    )
    for source in bundle.snapshot.sources:
        yield _link(
            f"annotate-source-{source.rank}",
            _annotation_composer_url(source, options.notebook_parameters(), root_path),
            f"Annotate frozen evidence {source.rank}",
        )
        yield " " + _identity(source.chunk.title, "annotation-source-title")
    if not bundle.snapshot.sources:
        yield "<p>No frozen passages were captured; there is no source to annotate.</p>"
    yield "</section>"
    yield from _reviews(reviews, bundle, options, root_path)
    yield from _annotations(annotations, options, root_path)


def _source(
    saved: SavedEvidenceAnnotation,
    source: EvidenceSource,
    options: _AnnotationParameters,
    root_path: str,
) -> Iterator[str]:
    yield _navigation(options, root_path) + _detail_navigation(options, root_path)
    yield _NOTICE
    yield "<dl><dt>Run ID (JSON)</dt><dd>" + _identity(saved.run_id, "run-id") + "</dd></dl>"
    yield (
        "<p>This annotation and source were validated in one read-only snapshot of saved events. "
        "Document ID, chunk ID, SHA-256, offsets and exact quote must all match. "
        "No current-corpus text is used, even after corpus deletion or restart.</p>"
    )
    yield f'<article id="source-{source.rank}" class="selected-anchor">'
    yield f"<h2>Frozen evidence {source.rank} / exact annotated occurrence</h2>"
    yield (
        "<p>The mark selects only the recorded half-open Unicode code-point span, not every "
        "occurrence of the quote, UTF-8 byte offsets or normalized text. The complete captured "
        "chunk follows, not necessarily a complete paper.</p>"
        '<pre class="literal passage-text" dir="auto"><code translate="no">'
        + _escaped(source.chunk.text[: saved.start])
        + '<mark id="annotation-quote" tabindex="-1">'
        + _escaped(source.chunk.text[saved.start : saved.end])
        + "</mark>"
        + _escaped(source.chunk.text[saved.end :])
        + "</code></pre>"
    )
    yield "<h3>Recorded annotation</h3>"
    yield from _annotation_record(saved)
    metadata = source.model_dump(mode="json")
    metadata["chunk"].pop("text")
    yield "<h3>Frozen source identity, rank, path and metadata (literal, not active links)</h3>"
    yield _literal(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True), "source-metadata"
    )
    yield _link(
        "full-frozen-source",
        options.evidence_url(root_path) + f"#source-{source.rank}",
        "Open this source in the complete saved evidence",
    )
    yield " " + _link(
        "annotate-source",
        _annotation_composer_url(source, options.notebook_parameters(), root_path),
        "Create a separate note on this frozen passage",
    )
    yield "</article>"


router = APIRouter(route_class=_RunExplorerRoute, tags=["Human review notebook"])


@router.get("/explore/run/notebook", response_class=HTMLResponse, responses=_RESPONSES)
def explore_notebook(
    request: Request, options: Annotated[_NotebookParameters, Query()]
) -> HTMLResponse:
    """Read independently bounded histories; no browser writes or consensus inference."""
    root_path = _prepare(request, options)
    container: AppContainer = request.app.state.container
    bundle = container.saved_run_evidence.export(options.run_id)
    reviews = container.notebook_reviews.list_reviews(
        options.run_id, limit=options.review_limit, cursor=options.review_cursor
    )
    annotations = container.evidence_annotations.list_annotations(
        options.run_id, limit=options.annotation_limit, cursor=options.annotation_cursor
    )
    return _page(
        "Human-review notebook",
        _notebook(bundle, reviews, annotations, options, root_path),
        saved_runs=True,
    )


@router.get("/explore/run/annotation", response_class=HTMLResponse, responses=_RESPONSES)
def explore_annotation(
    request: Request, options: Annotated[_AnnotationParameters, Query()]
) -> HTMLResponse:
    """Follow a stable run-scoped UUID, not a moving page position or a guessed text match."""
    root_path = _prepare(request, options)
    container: AppContainer = request.app.state.container
    saved, source = container.evidence_annotations.get_annotation(
        options.run_id, options.annotation_id
    )
    return _annotation_page(saved, source, options, root_path)


def _annotation_page(
    saved: SavedEvidenceAnnotation,
    source: EvidenceSource,
    options: _AnnotationParameters,
    root_path: str,
) -> HTMLResponse:
    return _page(
        "Annotated frozen source",
        _source(saved, source, options, root_path),
        saved_runs=True,
    )
