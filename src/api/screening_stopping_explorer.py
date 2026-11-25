"""Native, read-only browser page for advisory screening stopping rules; never a decision."""

import json
import re
import sqlite3
from collections.abc import Iterator
from typing import Annotated, Self
from urllib.parse import urlencode

from fastapi import APIRouter, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator
from starlette.concurrency import run_in_threadpool
from starlette.types import Receive, Scope, Send

from api.collection_explorer import _ERRORS
from api.dependencies import AppContainer
from api.explorer import (
    _collection_link,
    _CollectionReturn,
    _CollectionReturnToken,
    _decode_collection_return,
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
from api.native_forms import FormTooLargeError, parse_fields, read_body, unsigned_decimal
from api.screening_explorer import PATH as SCREENING_PATH
from api.screening_explorer import PRIORITY_PATH, STOPPING_PATH
from storage.paper_collections import CollectionError, CollectionId, Revision
from storage.screening_stopping import (
    DEFAULT_CONFIDENCE,
    DEFAULT_MIN_SCREENED_PERCENT,
    DEFAULT_TARGET_RECALL,
    RandomSampleRecallTest,
    ScreeningStoppingAssessment,
    assess_screening_stopping,
)

MAX_QUERY_BYTES = 65_536
MAX_FORM_BYTES = 65_536
_DECIMAL = re.compile(r"[01](?:\.[0-9]{1,6})?")


def _optional_integer(value: object) -> object:
    return None if value == "" else unsigned_decimal(value)


def _optional_decimal(value: object) -> object:
    if value == "":
        return None
    if isinstance(value, str):
        if not _DECIMAL.fullmatch(value):
            raise ValueError("Use a decimal such as 0.95 with at most 6 places.")
        return float(value)
    return value


def _optional_text(value: object) -> object:
    return None if value == "" else value


_Integer = Annotated[int | None, BeforeValidator(_optional_integer)]
_Decimal = Annotated[float | None, BeforeValidator(_optional_decimal)]


class _StoppingOpen(BaseModel):
    """One exact collection revision plus optional assessment parameters from a GET form."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    collection_id: CollectionId
    collection_revision: Annotated[Revision, BeforeValidator(unsigned_decimal)]
    collection_return: _CollectionReturnToken | None = None
    target_recall: _Decimal = None
    confidence: _Decimal = None
    irrelevant_run: _Integer = None
    irrelevant_run_percent: _Integer = None
    min_screened_percent: _Integer = None
    estimated_relevant: _Integer = None
    key_document_ids: str = Field(default="", max_length=16_384)
    sample_seed: _Integer = None
    sample_after: _Integer = None
    sample_urn_sha256: Annotated[str | None, BeforeValidator(_optional_text)] = None

    @model_validator(mode="after")
    def matching_return(self) -> Self:
        if self.collection_return is not None:
            navigation = _decode_collection_return(self.collection_return)
            if (
                navigation.collection_id != self.collection_id
                or navigation.collection_revision != self.collection_revision
            ):
                raise ValueError("Return state must match this exact collection revision.")
        return self

    def navigation(self) -> str:
        return (
            self.collection_return
            or _CollectionReturn(
                collection_id=self.collection_id, collection_revision=self.collection_revision
            ).token()
        )

    def key_ids(self) -> tuple[str, ...]:
        """One exact ID per line (CRLF or LF); blank lines are ignored, nothing is trimmed."""
        return tuple(line for line in re.split(r"\r\n|\n", self.key_document_ids) if line)


_FAILURES = {
    **_ERRORS,
    "invalid_screening_stopping_request": (
        "Use an exact collection_id, a positive integer collection_revision and documented "
        "parameters: decimals such as 0.95 with at most 6 places, whole numbers, either a run "
        "length or a run percentage, a seed with sample_after, and one exact key paper ID per "
        "line. Unknown/duplicate fields, malformed encoding and GET bodies are rejected. "
        "Nothing was recorded."
    ),
    "screening_stopping_request_too_large": "The URL exceeds 65536 bytes. Nothing was recorded.",
    "screening_stopping_method": (
        "The stopping assessment is read with GET. Record decisions only through the explicit "
        "human screening form. Nothing was recorded."
    ),
    "stopping_key_document_not_member": (
        "A key paper ID is not a member of this exact collection revision. Use exact IDs."
    ),
    "stopping_parameter_out_of_range": (
        "A run length or estimate exceeds the collection, or sample_after exceeds the members "
        "with a current decision. Nothing was recorded."
    ),
    "stopping_sample_boundary_tied": (
        "Two members on either side of sample_after share one recorded time, so the sample "
        "cannot be reconstructed unambiguously. Draw a new sample."
    ),
    "stopping_sample_urn_changed": (
        "The members not screened before sample_after differ from the urn this sample was "
        "drawn from. Draw a new sample; nothing was assessed."
    ),
    "screening_stopping_history_too_large": (
        "Too many retained decisions to read in one assessment. Nothing was assessed."
    ),
    "screening_stopping_too_large": (
        "The complete assessment exceeds 524288 UTF-8 bytes. No partial assessment is shown."
    ),
    "invalid_screening_stopping": (
        "The screening snapshot could not be assessed consistently. Nothing was saved or repaired."
    ),
    "html_page_too_large": "The complete escaped page exceeds 1 MiB. No partial page is shown.",
    "html_text_not_representable": (
        "Saved text cannot be displayed faithfully as UTF-8 HTML. Use the JSON/Python readers."
    ),
}


def _failure(status: int, code: str, root_path: str) -> HTMLResponse:
    return _error(
        status,
        code,
        "Stopping rules unavailable",
        _FAILURES.get(code, "The assessment could not be read. No success is reported."),
        navigation=_link(
            "back-to-collections", root_path + "/explore/collections", "Reopen a saved collection"
        ),
        stopping=True,
    )


class _StoppingExplorerRoute(APIRoute):
    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None:
        if self.methods is not None and scope["method"] not in self.methods:
            root_path = scope.get("root_path", "").rstrip("/")
            response = _failure(405, "screening_stopping_method", root_path)
            response.headers["Allow"] = "GET"
            await response(scope, receive, send)
            return
        await super().handle(scope, receive, send)


def _decimal_text(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _pairs(options: _StoppingOpen, *, sample: bool = True) -> list[tuple[str, str]]:
    """The caller's explicit parameters, in a stable order; defaults stay implicit."""
    pairs = [
        (name, _decimal_text(value))
        for name in ("target_recall", "confidence")
        if (value := getattr(options, name)) is not None
    ]
    names = ["irrelevant_run", "irrelevant_run_percent", "min_screened_percent"]
    names += ["estimated_relevant"] + (["sample_seed", "sample_after"] if sample else [])
    pairs += [(name, str(value)) for name in names if (value := getattr(options, name)) is not None]
    if sample and options.sample_urn_sha256 is not None:
        pairs.append(("sample_urn_sha256", options.sample_urn_sha256))
    return pairs


def _page_url(root_path: str, options: _StoppingOpen, pairs: list[tuple[str, str]]) -> str:
    base = [
        ("collection_id", options.collection_id),
        ("collection_revision", str(options.collection_revision)),
        ("collection_return", options.navigation()),
    ]
    keys = [("key_document_ids", "\r\n".join(options.key_ids()))] if options.key_ids() else []
    return root_path + STOPPING_PATH + "?" + urlencode(base + pairs + keys)


def _hidden(pairs: list[tuple[str, str]]) -> str:
    return "".join(
        f'<input type="hidden" name="{name}" value="{_escaped(value)}">' for name, value in pairs
    )


def _status(identifier: str, value: str) -> str:
    return f'<strong class="rule-status" id="{identifier}">{_escaped(value)}</strong>'


def _member(identifier: str) -> str:
    return f'<code class="member-id" translate="no">{_escaped(json.dumps(identifier))}</code>'


def _screen_link(
    prefix: str, position: int, identifier: str, options: _StoppingOpen, root_path: str
) -> str:
    return _link(
        f"{prefix}-{position}",
        _url(
            root_path + SCREENING_PATH,
            {
                "collection_id": options.collection_id,
                "collection_revision": options.collection_revision,
                "document_id": identifier,
                "collection_return": options.navigation(),
            },
        ),
        "Open the explicit human screening form",
    )


def _summary(result: ScreeningStoppingAssessment) -> Iterator[str]:
    counts = result.counts
    yield "<h2>Current human screening (all members)</h2><dl>"
    yield f'<dt>members</dt><dd id="count-members">{result.total_documents}</dd>'
    yield f'<dt>screened</dt><dd id="count-screened">{result.timeline.screened}</dd>'
    for status, count in counts.model_dump().items():
        yield f'<dt>{status}</dt><dd id="count-{status}">{count}</dd>'
    yield "</dl>"
    if result.all_screened:
        yield (
            '<p class="notice" id="all-screened">Every member has a current decision; there '
            "is nothing left to stop screening. The rules below are shown for audit.</p>"
        )


def _consecutive(result: ScreeningStoppingAssessment) -> Iterator[str]:
    rule = result.consecutive_irrelevant
    yield (
        "<h2>1. Consecutive irrelevant records (ASReview-style heuristic)</h2>"
        f"<p>Status: {_status('consecutive-status', rule.status)}. Current run "
        f'<strong id="consecutive-run">{rule.current_run}</strong> of threshold '
        f'<strong id="consecutive-threshold">{rule.threshold}</strong> '
        f"({rule.threshold_percent_of_collection}% of the collection).</p>"
        f'<p id="consecutive-explanation">{_escaped(rule.explanation)}</p>'
    )
    if rule.last_relevant_at is not None:
        yield (
            "<p>Most recent include or unsure decision recorded at "
            f'<time id="last-relevant-at">{rule.last_relevant_at.isoformat()}</time>.</p>'
        )
    if rule.run_document_ids:
        yield "<details><summary>Excludes in the current run</summary><ol>"
        for identifier in rule.run_document_ids:
            yield f'<li class="run-member">{_member(identifier)}</li>'
        yield "</ol></details>"


def _safe(result: ScreeningStoppingAssessment) -> Iterator[str]:
    checkpoint = result.safe_checkpoint
    yield (
        "<h2>2. SAFE-style checkpoint (end of the active-learning phase)</h2>"
        f"<p>Overall: {_status('safe-status', checkpoint.status)}. "
        f'<span id="safe-explanation">{_escaped(checkpoint.explanation)}</span></p>'
        '<table id="safe-criteria"><thead><tr><th scope="col">Condition</th>'
        '<th scope="col">Status</th><th scope="col">Detail</th></tr></thead><tbody>'
    )
    for criterion in checkpoint.criteria:
        yield (
            f'<tr class="safe-criterion"><th scope="row"><code>{criterion.name}</code></th>'
            f'<td class="criterion-status">{criterion.status}</td>'
            f"<td>{_escaped(criterion.detail)}</td></tr>"
        )
    yield "</tbody></table>"
    if checkpoint.key_documents_not_included:
        yield "<p>Key papers that are not current includes:</p><ul>"
        for identifier in checkpoint.key_documents_not_included:
            yield f'<li class="missing-key">{_member(identifier)}</li>'
        yield "</ul>"


def _start_sample(
    test: RandomSampleRecallTest, options: _StoppingOpen, root_path: str
) -> Iterator[str]:
    yield (
        f'<p id="recall-explanation">{_escaped(test.explanation)}</p>'
        '<form id="start-sample" method="get" '
        f'action="{_escaped(root_path + STOPPING_PATH)}" accept-charset="UTF-8" '
        'autocomplete="off" aria-describedby="sample-help">'
        + _hidden(
            [
                ("collection_id", options.collection_id),
                ("collection_revision", str(options.collection_revision)),
                ("collection_return", options.navigation()),
                *_pairs(options, sample=False),
                ("key_document_ids", "\r\n".join(options.key_ids())),
                ("sample_after", str(test.sample_after)),
                ("sample_urn_sha256", test.urn_sha256),
            ]
        )
        + '<div><label for="sample-seed">Seed: any whole number you choose now</label>'
        '<input id="sample-seed" name="sample_seed" required inputmode="numeric" '
        'pattern="[0-9]{1,19}" maxlength="19"></div>'
        '<p id="sample-help">Choose the seed before you see the sample and use one seed per '
        f"assessment. The sample orders the {test.urn_size} pending members by SHA-256; "
        "reading it records nothing. Screen them in the listed order, then return here.</p>"
        '<div class="actions"><button type="submit">Show the seeded random sample</button>'
        "</div></form>"
    )


def _recall(
    result: ScreeningStoppingAssessment, options: _StoppingOpen, root_path: str
) -> Iterator[str]:
    test = result.recall_test
    yield (
        "<h2>3. Random-sample recall test (buscar)</h2>"
        f"<p>Status: {_status('recall-status', test.status)}. H0: recall &lt; "
        f'<span id="target-recall">{test.target_recall}</span> at '
        f'<span id="confidence">{test.confidence}</span> confidence '
        f"(alpha {test.alpha:.6g}).</p>"
    )
    if test.status == "not_started":
        yield from _start_sample(test, options, root_path)
        return
    yield f'<p id="recall-explanation">{_escaped(test.explanation)}</p><dl>'
    rows = (
        ("p-value", "p-value", "-" if test.p_value is None else f"{test.p_value:.6g}"),
        ("p-value-exact", "exact p", test.p_value_exact or "-"),
        ("k-hat", "fewest relevant urn records under H0", test.h0_min_relevant_in_urn),
        ("sample-size", "sample records observed in order", test.sample_size),
        ("sample-relevant", "relevant in the sample", test.sample_relevant),
        ("relevant-before", "includes before the sample", test.relevant_before_sample),
        ("decided-outside", "urn members decided outside the sample", test.decided_outside_sample),
        ("urn-size", "urn (pending when the sample was drawn)", test.urn_size),
        (
            "recall-estimate",
            "recall estimate",
            "-" if test.recall_estimate is None else f"{test.recall_estimate:.4f}",
        ),
        (
            "recall-lower-bound",
            "recall lower bound",
            "-" if test.recall_lower_bound is None else f"{test.recall_lower_bound:.4f}",
        ),
        (
            "additional-needed",
            "pending sample records to screen if all are irrelevant",
            "-" if test.additional_irrelevant_needed is None else test.additional_irrelevant_needed,
        ),
    )
    for identifier, label, value in rows:
        shown = "-" if value is None else value
        yield f'<dt>{label}</dt><dd id="{identifier}">{_escaped(str(shown))}</dd>'
    yield "</dl>"
    yield (
        "<p>Keep this exact sample: seed, boundary and urn digest. Bookmark this page or "
        "note the values below; a different seed is a different sample.</p>"
    )
    yield _literal(
        json.dumps(
            {
                "sample_seed": test.sample_seed,
                "sample_after": test.sample_after,
                "sample_urn_sha256": test.urn_sha256,
            },
            indent=2,
        ),
        "sample-identity",
    )
    if test.next_document_ids:
        yield (
            f"<h3>Screen next, in this order ({len(test.next_document_ids)} remaining)</h3>"
            "<p>Open each form, record your own decision, then return with Back or your "
            "bookmark. Skipping ahead makes the sample non-random.</p><ol>"
        )
        for position, identifier in enumerate(test.next_document_ids, 1):
            link = _screen_link("screen-next", position, identifier, options, root_path)
            yield f'<li class="next-member">{_member(identifier)} {link}</li>'
        yield "</ol>"
    yield (
        "<details><summary>Complete sample order</summary>"
        '<table id="sample-order"><thead><tr><th scope="col">#</th><th scope="col">Member</th>'
        '<th scope="col">Status</th><th scope="col">Observed</th></tr></thead><tbody>'
    )
    for item in test.sample_order:
        yield (
            f'<tr class="sample-item"><td>{item.sample_position}</td>'
            f"<td>{_member(item.document_id)}</td><td>{item.status}</td>"
            f"<td>{'yes' if item.observed else 'no'}</td></tr>"
        )
    yield "</tbody></table></details>"


def _timeline(result: ScreeningStoppingAssessment) -> Iterator[str]:
    timeline = result.timeline
    yield (
        "<h2>Recorded decision order (this collection revision)</h2>"
        "<p>Ordered by each member's first recorded decision time at this revision; ties "
        f'share a time. Order reliable: <strong id="order-reliable">'
        f"{'yes' if timeline.order_reliable else 'no'}</strong>; tied positions: "
        f'<span id="tied-positions">{timeline.tied_positions}</span>.</p>'
    )
    for warning in timeline.order_warnings:
        yield f'<p class="notice order-warning">{_escaped(warning)}</p>'
    if not timeline.items:
        yield "<p>No member has a current decision at this revision yet.</p>"
        return
    yield (
        "<details><summary>All screened members in recorded order</summary>"
        '<table id="timeline"><thead><tr><th scope="col">#</th><th scope="col">Member</th>'
        '<th scope="col">Decision</th><th scope="col">First recorded</th>'
        '<th scope="col">Current decision recorded</th><th scope="col">Includes so far</th>'
        "</tr></thead><tbody>"
    )
    for item in timeline.items:
        tie = " (tie)" if item.tied_with_previous else ""
        retained = "" if item.first_decision_retained else " (earlier history not retained)"
        yield (
            f'<tr class="timeline-item"><td>{item.position}{tie}</td>'
            f"<td>{_member(item.document_id)}</td><td>{item.decision}</td>"
            f"<td>{item.first_decided_at.isoformat()}{retained}</td>"
            f"<td>{item.decided_at.isoformat()}</td><td>{item.includes_so_far}</td></tr>"
        )
    yield "</tbody></table></details>"


def _input(name: str, label: str, value: str, help_text: str, pattern: str) -> str:
    return (
        f'<div><label for="{name}">{label}</label>'
        f'<input id="{name}" name="{name}" value="{_escaped(value)}" pattern="{pattern}" '
        f'inputmode="decimal" aria-describedby="{name}-help">'
        f'<p id="{name}-help">{help_text}</p></div>'
    )


def _parameters_form(
    result: ScreeningStoppingAssessment, options: _StoppingOpen, root_path: str
) -> Iterator[str]:
    parameters = result.parameters
    sample = [
        (name, value)
        for name, value in _pairs(options)
        if name in ("sample_seed", "sample_after", "sample_urn_sha256")
    ]
    whole = "[0-9]{1,3}"
    yield (
        "<h2>Adjust the parameters (read-only)</h2>"
        '<form id="stopping-parameters" method="get" '
        f'action="{_escaped(root_path + STOPPING_PATH)}" accept-charset="UTF-8" '
        'autocomplete="off">'
        + _hidden(
            [
                ("collection_id", options.collection_id),
                ("collection_revision", str(options.collection_revision)),
                ("collection_return", options.navigation()),
                *sample,
            ]
        )
        + _input(
            "target_recall",
            "Target recall",
            _decimal_text(parameters.target_recall),
            f"Default {DEFAULT_TARGET_RECALL}; a decimal above 0 and at most 1.",
            r"[01](\.[0-9]{1,6})?",
        )
        + _input(
            "confidence",
            "Confidence",
            _decimal_text(parameters.confidence),
            f"Default {DEFAULT_CONFIDENCE}; a decimal above 0 and below 1.",
            r"0(\.[0-9]{1,6})?",
        )
        + _input(
            "irrelevant_run",
            "Consecutive excludes (records)",
            str(options.irrelevant_run or ""),
            "Leave blank to use a percentage of the collection instead.",
            whole,
        )
        + _input(
            "irrelevant_run_percent",
            "Consecutive excludes (% of the collection)",
            str(options.irrelevant_run_percent or ""),
            "Default 10% (rounded up); give either records or a percentage.",
            whole,
        )
        + _input(
            "min_screened_percent",
            "SAFE minimum screened (%)",
            str(parameters.min_screened_percent),
            f"Default {DEFAULT_MIN_SCREENED_PERCENT}%.",
            whole,
        )
        + _input(
            "estimated_relevant",
            "SAFE RR_T: estimated relevant records in the collection",
            "" if parameters.estimated_relevant is None else str(parameters.estimated_relevant),
            "From a random pilot sample: relevant fraction times members. Blank = not evaluated.",
            whole,
        )
        + '<div class="worksheet-input"><label for="key_document_ids">Key papers (one exact '
        "ID per line)</label>"
        '<textarea id="key_document_ids" name="key_document_ids" spellcheck="false" '
        'aria-describedby="key-help">'
        + _escaped("\n".join(options.key_ids()))
        + '</textarea><p id="key-help">Known relevant papers chosen before screening. IDs '
        "that contain line breaks need the JSON API.</p></div>"
        '<div class="actions"><button type="submit">Reassess (nothing is recorded)</button>'
        "</div></form>"
    )


def _notes(result: ScreeningStoppingAssessment) -> Iterator[str]:
    yield '<h2>Assumptions</h2><ul id="assumptions">'
    yield from (f"<li>{_escaped(text)}</li>" for text in result.assumptions)
    yield '</ul><h2>Limitations</h2><ul id="limitations">'
    yield from (f"<li>{_escaped(text)}</li>" for text in result.limitations)
    yield "</ul>"


def _fragments(
    result: ScreeningStoppingAssessment, options: _StoppingOpen, root_path: str
) -> Iterator[str]:
    navigation = options.navigation()
    json_pairs = [("collection_revision", str(result.collection_revision)), *_pairs(options)]
    json_pairs += [("key_document_id", key) for key in options.key_ids()]
    yield '<nav aria-label="Screening navigation">'
    yield _collection_link(navigation, root_path)
    yield _link(
        "stopping-priority",
        _url(
            root_path + PRIORITY_PATH,
            {
                "collection_id": result.collection_id,
                "collection_revision": result.collection_revision,
                "collection_return": navigation,
            },
        ),
        "Most likely relevant first: the prioritized queue",
    )
    yield _link(
        "stopping-json",
        f"{root_path}/collections/{result.collection_id}/screening-stopping?"
        + urlencode(json_pairs),
        "Read this assessment as JSON (a separate read)",
    )
    yield _link(
        "stopping-reload", _page_url(root_path, options, _pairs(options)), "Reload this assessment"
    )
    yield "</nav>"
    yield _literal(result.collection_name, "collection-name")
    yield _identity(result.collection_id, "collection-id")
    yield (
        f"<p>Collection revision: {result.collection_revision}.</p>"
        '<p class="notice" id="stopping-guarantee">Advisory only. Nothing on this page '
        "records, changes or suggests a decision, excludes or hides a member, or stops "
        "screening. Each rule below is evidence for your own documented decision about when "
        "to stop; unscreened and stale members still need your explicit decision.</p>"
    )
    yield from _summary(result)
    yield from _consecutive(result)
    yield from _safe(result)
    yield from _recall(result, options, root_path)
    yield from _timeline(result)
    yield from _parameters_form(result, options, root_path)
    yield from _notes(result)


def _stopping_page(container: AppContainer, options: _StoppingOpen, root_path: str) -> HTMLResponse:
    result = assess_screening_stopping(
        container.paper_screening,
        options.collection_id,
        collection_revision=options.collection_revision,
        target_recall=DEFAULT_TARGET_RECALL
        if options.target_recall is None
        else options.target_recall,
        confidence=DEFAULT_CONFIDENCE if options.confidence is None else options.confidence,
        irrelevant_run=options.irrelevant_run,
        irrelevant_run_percent=options.irrelevant_run_percent,
        min_screened_percent=DEFAULT_MIN_SCREENED_PERCENT
        if options.min_screened_percent is None
        else options.min_screened_percent,
        estimated_relevant=options.estimated_relevant,
        key_document_ids=options.key_ids(),
        sample_seed=options.sample_seed,
        sample_after=options.sample_after,
        sample_urn_sha256=options.sample_urn_sha256,
    )
    return _page(
        "When could you stop screening?",
        _fragments(result, options, root_path),
        stopping=True,
    )


router = APIRouter(route_class=_StoppingExplorerRoute, tags=["Human screening stopping rules"])


@router.get(
    STOPPING_PATH,
    response_class=HTMLResponse,
    responses={
        404: {"description": "Exact collection not found"},
        405: {"description": "Only GET is supported; nothing is recorded"},
        409: {"description": "Stale revision, invalid saved data, tied boundary or changed urn"},
        413: {"description": "URL, retained history, assessment or escaped HTML exceeds its bound"},
        422: {"description": "Invalid collection, revision, return state or parameters"},
        503: {"description": "Existing collection or screening storage unavailable"},
    },
)
async def explore_screening_stopping(request: Request) -> Response:
    """Advisory stopping rules over recorded decisions; read-only, never a decision or stop."""
    root_path = request.scope.get("root_path", "").rstrip("/")
    try:
        raw_query = request.scope.get("query_string", b"")
        if len(raw_query) > MAX_QUERY_BYTES:
            raise FormTooLargeError()
        if await read_body(request, max_bytes=MAX_FORM_BYTES):
            raise ValueError("GET cannot accept a body.")
        options = _StoppingOpen.model_validate(parse_fields(raw_query, max_fields=13))
        container: AppContainer = request.app.state.container
        return await run_in_threadpool(_stopping_page, container, options, root_path)
    except CollectionError as exc:
        return _failure(exc.status_code, exc.code, root_path)
    except sqlite3.Error:
        return _failure(503, "screening_storage_unavailable", root_path)
    except FormTooLargeError:
        return _failure(413, "screening_stopping_request_too_large", root_path)
    except _PageTooLargeError:
        return _failure(413, "html_page_too_large", root_path)
    except _UnrepresentableTextError:
        return _failure(409, "html_text_not_representable", root_path)
    except (ValueError, UnicodeError, RecursionError):
        return _failure(422, "invalid_screening_stopping_request", root_path)
