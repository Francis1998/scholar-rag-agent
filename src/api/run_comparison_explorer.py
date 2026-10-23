"""Native GET selection and exact, bounded inspection of two frozen saved runs."""

import json
from collections.abc import Iterator
from typing import Annotated, Self

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from agent.comparison_models import ComparedRun, RunComparison, SourceSummary, TextSummary
from api.dependencies import AppContainer
from api.explorer import (
    _escaped,
    _identity,
    _link,
    _literal,
    _page,
    _return_pairs,
    _UnrepresentableTextError,
    _url,
)
from api.run_explorer import (
    _RESPONSES,
    _HistoryCursor,
    _HistoryLimit,
    _HistoryParameters,
    _navigation,
    _RunExplorerRoute,
)
from storage.evidence_export import EvidenceExportError
from storage.run_comparison import ComparisonSide, RunComparisonError, SavedRunComparator
from storage.run_history import Identity, RunHistoryPage

router = APIRouter(route_class=_RunExplorerRoute, tags=["Saved run browser"])
PATH = "/explore/runs/compare"
_SIDES: tuple[ComparisonSide, ComparisonSide] = ("baseline", "candidate")


def _selection(value: object) -> object:
    if value == "":
        return None
    if (
        not isinstance(value, str)
        or len(value) > 3074
        or not value.lstrip(" \t\r\n").startswith('"')
    ):
        raise ValueError("Use a bounded JSON string for an exact run ID.")
    decoded = json.loads(value)
    if not isinstance(decoded, str):
        raise ValueError("A run ID must be a JSON string.")
    decoded.encode("utf-8")
    return decoded


def _history(value: str) -> _HistoryParameters:
    try:
        decoded = json.loads(value, object_pairs_hook=_return_pairs)
    except RecursionError as exc:
        raise ValueError("History return state is too deeply nested.") from exc
    history = _HistoryParameters.model_validate(decoded)
    json.dumps(history.parameters(), ensure_ascii=False).encode("utf-8")
    return history


class _Parameters(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    baseline: Annotated[Identity | None, BeforeValidator(_selection)] = None
    candidate: Annotated[Identity | None, BeforeValidator(_selection)] = None
    history: str = Field(default="{}", max_length=12_000)
    limit: _HistoryLimit = 20
    cursor: _HistoryCursor | None = None

    @model_validator(mode="after")
    def valid_history(self) -> Self:
        _history(self.history)
        return self

    def parameters(self) -> dict[str, str | int | None]:
        return {
            "baseline": json.dumps(self.baseline, ensure_ascii=True)
            if self.baseline is not None
            else None,
            "candidate": json.dumps(self.candidate, ensure_ascii=True)
            if self.candidate is not None
            else None,
            "history": self.history,
            "limit": self.limit,
            "cursor": self.cursor,
        }


def _chooser(options: _Parameters, page: RunHistoryPage | None, root_path: str) -> Iterator[str]:
    yield '<section aria-labelledby="selection"><h2 id="selection">Choose an explicit baseline</h2>'
    yield (
        "<p>Baseline is the reference; candidate is compared against it. Nothing runs again. "
        "Choose both explicitly, including the same run for a no-change control. "
        "IDs in these native controls are JSON strings, so slashes, Unicode and control "
        "characters survive submission without trimming or normalization.</p>"
        '<form id="comparison-selection" method="get" autocomplete="off" '
        f'action="{_escaped(root_path + PATH)}">'
    )
    choices = (
        {
            run.run_id: (
                f"{json.dumps(run.run_id, ensure_ascii=True)} | "
                f"{run.recorded_state or 'Not recorded'} | "
                f"{json.dumps(run.query_summary, ensure_ascii=True)}"
                + (" [query preview truncated]" if run.query_truncated else "")
            )
            for run in page.runs
        }
        if page is not None
        else {}
    )
    for selected in (options.baseline, options.candidate):
        if selected is not None and selected not in choices:
            choices[selected] = json.dumps(selected, ensure_ascii=True) + " | pinned exact ID"
    for side in ("baseline", "candidate"):
        selected = options.baseline if side == "baseline" else options.candidate
        yield (
            f'<div><label for="{side}">{side.capitalize()} saved run</label>'
            f'<select id="{side}" name="{side}" required aria-describedby="selection-help">'
            f'<option value=""{" selected" if selected is None else ""}>Choose a saved run</option>'
        )
        for run_id, label in choices.items():
            yield (
                f'<option value="{_escaped(json.dumps(run_id, ensure_ascii=True))}"'
                f"{' selected' if run_id == selected else ''}>{_escaped(label)}</option>"
            )
        yield "</select></div>"
    yield (
        '<div><label for="limit">Choices per page</label>'
        f'<input id="limit" name="limit" type="number" min="1" max="100" '
        f'value="{options.limit}" required></div>'
        f'<input type="hidden" name="history" value="{_escaped(options.history)}">'
        '<div class="actions"><button type="submit">Compare saved runs</button></div>'
        '</form><p id="selection-help">Recorded states are not exportability checks. '
        "Pinned IDs may be outside this choice page. Neither side is shown unless both "
        "complete captures validate; an incomplete selection only shows this chooser.</p>"
    )
    if page is not None:
        yield (
            f"<p>{len(page.runs)} saved choices on this page; newest first-event ID first. "
            "Choice pages are not one frozen history snapshot.</p>"
        )
        if not page.runs:
            yield "<p>No saved choices on this page. No run was created.</p>"
        yield '<nav aria-label="Saved run choices">'
        yield _link(
            "first-choices",
            _url(root_path + PATH, {**options.parameters(), "cursor": None}),
            "First choices",
        )
        if page.next_cursor is not None:
            yield _link(
                "next-choices",
                _url(root_path + PATH, {**options.parameters(), "cursor": page.next_cursor}),
                "Next saved choices",
            )
        # Pin either side across pages without auto-selecting its counterpart.
        for number, run in enumerate(page.runs, 1):
            for side in ("baseline", "candidate"):
                yield _link(
                    f"pin-{side}-{number}",
                    _url(
                        root_path + PATH,
                        {**options.parameters(), side: json.dumps(run.run_id, ensure_ascii=True)},
                    ),
                    f"Pin choice {number} as {side}",
                )
        yield "</nav>"
    else:
        yield _link(
            "change-selection",
            _url(root_path + PATH, {**options.parameters(), "candidate": None, "cursor": None}),
            "Browse other saved runs (keep baseline)",
        )
    yield "</section>"


def _json(value: object, css_class: str = "") -> str:
    return _literal(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True), css_class)


def _text(value: TextSummary, css_class: str) -> str:
    return (
        _literal(value.preview, css_class)
        + f"<p>{value.characters} characters / {value.utf8_bytes} UTF-8 bytes. "
        + (
            "Preview truncated at 240 characters; open the frozen reader for full text."
            if value.truncated
            else "Complete text."
        )
        + "</p>"
        + _literal(value.sha256, "sha256")
    )


def _run_fields(run: ComparedRun, side: ComparisonSide) -> list[tuple[str, str]]:
    try:
        return [
            ("Run ID (JSON)", _identity(run.run_id, "compared-run-id")),
            (
                "Agent ID / completed time (not compared)",
                _json({"agent_id": run.agent_id, "completed_at": run.completed_at}),
            ),
            ("Recorded query", _text(run.query, "query-preview")),
            ("Document scope (ordered; null = unrestricted)", _json(run.document_ids)),
            (
                "Recorded evidence policy (null = not recorded)",
                _json(run.evidence_policy.model_dump(mode="json") if run.evidence_policy else None),
            ),
            ("Recorded configuration", _json(run.configuration.model_dump(mode="json"))),
            (
                "Recorded provider / configured model / task",
                _json(run.generation.model_dump(mode="json")),
            ),
            ("Recorded answer", _text(run.answer.text, "answer-preview")),
            (
                "Answer / claims / citations / warnings fingerprints and counts",
                _json(run.answer.model_dump(mode="json", exclude={"text"})),
            ),
            (
                "Request / context fingerprints",
                _json(
                    {
                        "request_sha256": run.request_sha256,
                        "context_sha256": run.context_sha256,
                    }
                ),
            ),
        ]
    except (_UnrepresentableTextError, UnicodeError) as exc:
        raise RunComparisonError(
            side, EvidenceExportError("html_text_not_representable", "")
        ) from exc


def _reader_url(run: ComparedRun, history: _HistoryParameters, root_path: str) -> str:
    return _url(root_path + "/explore/run", {"run_id": run.run_id, **history.parameters()})


def _source(
    source: SourceSummary | None, side: ComparisonSide, reader: str, identifier: str
) -> str:
    if source is None:
        return "<p>Not present on this side.</p>"
    try:
        return (
            _identity(source.chunk_id, f"{side}-chunk-id")
            + _identity(source.document_id, f"{side}-document-id")
            + f"<p>Frozen rank {source.rank}; score {source.score!r}.</p>"
            + _text(source.title, "source-title")
            + _text(source.retriever, "source-retriever")
            + "<details><summary>Exact source fingerprints</summary>"
            + _json(
                source.model_dump(
                    mode="json",
                    exclude={"title", "retriever", "chunk_id", "document_id", "rank", "score"},
                )
            )
            + "</details>"
            + _link(identifier, reader + f"#source-{source.rank}", "Open exact frozen source")
        )
    except (_UnrepresentableTextError, UnicodeError) as exc:
        raise RunComparisonError(
            side, EvidenceExportError("html_text_not_representable", "")
        ) from exc


def _table(caption: str) -> str:
    return (
        '<div class="comparison-grid"><table class="comparison-table">'
        f'<caption>{caption}</caption><thead><tr><th scope="col">Recorded field</th>'
        '<th scope="col">Baseline</th><th scope="col">Candidate</th></tr></thead><tbody>'
    )


def _results(
    result: RunComparison, options: _Parameters, history: _HistoryParameters, root_path: str
) -> Iterator[str]:
    left = _reader_url(result.baseline, history, root_path)
    right = _reader_url(result.candidate, history, root_path)
    yield '<section aria-labelledby="comparison"><h2 id="comparison">Baseline to candidate</h2>'
    yield (
        '<p id="comparison-outcome">'
        + (
            "Changes in compared saved fields."
            if result.any_changes
            else "No changes in compared saved fields."
        )
        + "</p>"
    )
    yield (
        "<p>No changes does not mean full trace equality, a quality improvement, a safe release "
        "or a fair A/B experiment. Trace IDs, clocks and non-allowlisted diagnostics are not "
        "compared. Configured model names are not resolved provider revisions.</p><ul>"
    )
    for notice in result.notices:
        yield "<li>" + _literal(notice, "comparison-notice") + "</li>"
    yield '</ul><nav aria-label="Comparison controls">'
    yield _link("baseline-reader", left, "Read complete baseline evidence")
    yield _link("candidate-reader", right, "Read complete candidate evidence")
    yield _link(
        "swap-comparison",
        _url(
            root_path + PATH,
            {
                **options.parameters(),
                "baseline": json.dumps(options.candidate, ensure_ascii=True),
                "candidate": json.dumps(options.baseline, ensure_ascii=True),
            },
        ),
        "Swap baseline and candidate (reverse direction)",
    )
    yield _link(
        "same-run-control",
        _url(
            root_path + PATH,
            {
                **options.parameters(),
                "candidate": json.dumps(options.baseline, ensure_ascii=True),
            },
        ),
        "Compare baseline to itself",
    )
    yield "</nav>" + _table(
        "Saved inputs and recorded outputs; text previews include exact digests"
    )
    for (label, baseline), (_, candidate) in zip(
        _run_fields(result.baseline, "baseline"),
        _run_fields(result.candidate, "candidate"),
        strict=True,
    ):
        yield f'<tr><th scope="row">{label}</th><td>{baseline}</td><td>{candidate}</td></tr>'
    yield "</tbody></table></div><h3>Exact change flags</h3><table><caption>"
    yield "Existing saved-run comparison semantics</caption><tbody>"
    for name, changed in result.changes.model_dump().items():
        yield (
            f'<tr><th scope="row">{name.replace("_", " ")}</th><td id="{name}">'
            + ("Changed" if changed else "Unchanged")
            + "</td></tr>"
        )
    yield "</tbody></table><h3>Frozen evidence changes</h3>"
    yield _json(result.evidence.statistics.model_dump(mode="json"), "evidence-statistics")
    yield (
        "<p>Identity is the exact (chunk ID, document ID) pair, not title or rank. Added means "
        "candidate-only; removed means baseline-only. Shared rows follow baseline order; "
        "positive rank delta means a later candidate position. Empty/empty overlap is null, "
        "not perfect support. Reader links use the saved run and that side's frozen rank.</p>"
    )
    if result.evidence.reassigned_chunk_ids:
        yield "<p>Reassigned chunk IDs (different document owners; not shared identities):</p>"
        yield _json(result.evidence.reassigned_chunk_ids, "reassigned-chunk-ids")
    yield _table("Exact frozen source pairs; full passages remain in each saved reader")
    for number, shared in enumerate(result.evidence.shared, 1):
        yield '<tr><th scope="row">Shared'
        yield f'<p class="rank-delta">Candidate rank delta: {shared.rank_delta:+d}</p>'
        yield _json(shared.changes.model_dump(), "source-changes") + "</th>"
        for side in _SIDES:
            source = shared.baseline if side == "baseline" else shared.candidate
            reader = left if side == "baseline" else right
            yield "<td>" + _source(source, side, reader, f"shared-{number}-{side}-source") + "</td>"
        yield "</tr>"
    for label, sources in (("Added", result.evidence.added), ("Removed", result.evidence.removed)):
        for number, source in enumerate(sources, 1):
            yield f'<tr><th scope="row">{label} {number}</th>'
            for side in _SIDES:
                reader = left if side == "baseline" else right
                present = (label == "Added") == (side == "candidate")
                yield (
                    "<td>"
                    + _source(
                        source if present else None,
                        side,
                        reader,
                        f"{label.lower()}-{number}-{side}-source",
                    )
                    + "</td>"
                )
            yield "</tr>"
    yield "</tbody></table></div></section>"


@router.get(PATH, response_class=HTMLResponse, responses=_RESPONSES)
def explore_comparison(request: Request, options: Annotated[_Parameters, Query()]) -> HTMLResponse:
    """Choose saved IDs, then compare two bounded validated captures without live work."""
    history = _history(options.history)
    request.state.saved_run_options = history
    root_path = request.scope.get("root_path", "").rstrip("/")
    container: AppContainer = request.app.state.container
    result = None
    choices = None
    if options.baseline is not None and options.candidate is not None:
        result = SavedRunComparator(container.saved_run_evidence).compare(
            options.baseline, options.candidate
        )
    else:
        choices = container.run_history.list_runs(limit=options.limit, cursor=options.cursor)

    def fragments() -> Iterator[str]:
        yield _navigation(history, root_path)
        yield from _chooser(options, choices, root_path)
        if result is not None:
            yield from _results(result, options, history, root_path)

    return _page("Saved-run comparison workbench", fragments(), saved_runs=True)
