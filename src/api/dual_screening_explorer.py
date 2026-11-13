"""Script-free native pages for independent reviewers, conflicts and agreement.

Reviewer pages show only one pseudonymous label's own decisions until blinding lifts for a
paper. The team page shows progress in blind mode and conflicts, resolutions and agreement
only after an explicit protocol change. Every write is an explicit same-origin POST.
"""

import json
import sqlite3
from collections.abc import Iterator
from typing import Annotated, Literal, Self

from fastapi import APIRouter, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator
from starlette.concurrency import run_in_threadpool
from starlette.types import Receive, Scope, Send

from api.collection_explorer import _ERRORS, _history_review
from api.dependencies import AppContainer
from api.explorer import (
    _WRITE_HEADERS,
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
    _return_pairs,
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
from storage.document_catalog import MAX_TITLE_CHARACTERS
from storage.dual_screening import (
    MAX_NAMED_REVIEWERS,
    BasisReference,
    ConflictItem,
    FinalDecision,
    FinalDecisionValue,
    NamedReviewer,
    ProtocolChange,
    ResolutionSubmission,
    ReviewerDecision,
    ReviewerItem,
    ReviewerSubmission,
    TeamOverview,
    agreement_line,
)
from storage.paper_collections import MAX_REVISION, CollectionError, CollectionId, Revision
from storage.paper_screening import MAX_REASON_CHARACTERS, Decision, Reason
from storage.screening_priority import PriorityItem
from storage.source_context import ContextDocumentIdentity

REVIEWER_PATH = "/explore/reviewer-screening"
TEAM_PATH = "/explore/screening-conflicts"
PROTOCOL_PATH = "/explore/screening-protocol"
MAX_FORM_BYTES = 65_536
MAX_QUERY_BYTES = 65_536
HISTORY_LIMIT = 5
_LABEL_PATTERN = r"[a-z0-9](?:[a-z0-9._\-]*[a-z0-9])?"
_LABEL_HELP = (
    "1-32 lowercase letters, digits, '.', '_' or '-', starting and ending with a letter or "
    "digit, for example r1 or reviewer-b. Use a pseudonymous code, never a name, email or "
    "other personal data. 'default' is reserved for existing unlabeled decisions."
)
_Revision = Annotated[Revision, BeforeValidator(unsigned_decimal)]
_ExpectedRevision = Annotated[
    int, Field(strict=True, ge=0, le=MAX_REVISION), BeforeValidator(unsigned_decimal)
]


def _matching(token: str | None, collection_id: str, revision: int) -> None:
    if token is not None:
        navigation = _decode_collection_return(token)
        if navigation.collection_id != collection_id or navigation.collection_revision != revision:
            raise ValueError("Return state must match this exact collection revision.")


def _navigation_token(token: str | None, collection_id: str, revision: int) -> str:
    return (
        token
        or _CollectionReturn(collection_id=collection_id, collection_revision=revision).token()
    )


class _ReviewerOpen(BaseModel):
    """GET state: no reviewer opens the label form; a document opens one member."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    collection_id: CollectionId
    collection_revision: _Revision
    reviewer: NamedReviewer | None = None
    document_id: ContextDocumentIdentity | None = None
    blind: Literal["true"] | None = None
    order: Literal["priority"] | None = None
    collection_return: _CollectionReturnToken | None = None
    saved_revision: _Revision | None = None

    @model_validator(mode="after")
    def consistent(self) -> Self:
        _matching(self.collection_return, self.collection_id, self.collection_revision)
        if self.reviewer is None and (
            self.document_id or self.blind or self.order or self.saved_revision
        ):
            raise ValueError("Choose a reviewer label before opening reviewer state.")
        if self.document_id is not None and self.order is not None:
            raise ValueError("A member page has no queue order.")
        if self.saved_revision is not None and self.document_id is None:
            raise ValueError("Confirmations belong to one member.")
        return self

    def state(self, **updates: str | int | None) -> dict[str, str | int | None]:
        return {
            "collection_id": self.collection_id,
            "collection_revision": self.collection_revision,
            "reviewer": self.reviewer,
            "blind": self.blind,
            "collection_return": self.collection_return,
            **updates,
        }

    def navigation(self) -> str:
        return _navigation_token(
            self.collection_return, self.collection_id, self.collection_revision
        )


class _ReviewerSelection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    collection_id: CollectionId
    collection_revision: Revision
    reviewer: NamedReviewer
    document_id: ContextDocumentIdentity
    blind: Literal["true"] | None = None
    collection_return: _CollectionReturnToken | None = None

    @model_validator(mode="after")
    def matching_return(self) -> Self:
        _matching(self.collection_return, self.collection_id, self.collection_revision)
        return self


class _ReviewerForm(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    selection: str = Field(max_length=16_384)
    expected_decision_revision: _ExpectedRevision
    decision: Decision
    reason: Reason


class _TeamOpen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    collection_id: CollectionId
    collection_revision: _Revision
    collection_return: _CollectionReturnToken | None = None
    saved_document_id: ContextDocumentIdentity | None = None
    saved_resolution_revision: _Revision | None = None
    saved_protocol_revision: _Revision | None = None

    @model_validator(mode="after")
    def consistent(self) -> Self:
        _matching(self.collection_return, self.collection_id, self.collection_revision)
        if (self.saved_document_id is None) != (self.saved_resolution_revision is None):
            raise ValueError("A resolution confirmation names one member and one revision.")
        return self

    def navigation(self) -> str:
        return _navigation_token(
            self.collection_return, self.collection_id, self.collection_revision
        )


class _TeamSelection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    collection_id: CollectionId
    collection_revision: Revision
    collection_return: _CollectionReturnToken | None = None

    @model_validator(mode="after")
    def matching_return(self) -> Self:
        _matching(self.collection_return, self.collection_id, self.collection_revision)
        return self


class _ConflictSelection(_TeamSelection):
    document_id: ContextDocumentIdentity
    expected_basis: tuple[BasisReference, ...] = Field(
        min_length=2, max_length=MAX_NAMED_REVIEWERS + 1
    )


class _ResolutionForm(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    selection: str = Field(max_length=16_384)
    expected_resolution_revision: _ExpectedRevision
    resolver: NamedReviewer
    decision: FinalDecisionValue
    rationale: Reason


class _ProtocolForm(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    selection: str = Field(max_length=16_384)
    expected_protocol_revision: _ExpectedRevision
    blind: Literal["on", "off"]
    exclusion_reasons_must_match: Literal["yes", "no"]
    changed_by: NamedReviewer


_FAILURES = {
    **_ERRORS,
    "invalid_reviewer_screening_form": (
        "Use the exact opened collection, revision, member and a pseudonymous reviewer label; "
        "choose a decision explicitly and write a nonblank reason or rationale of 1-1000 "
        "characters. Unknown/duplicate fields, malformed encoding, invalid UTF-8, GET bodies "
        "and POST query fields are rejected. No input is echoed and nothing was written. "
        + _LABEL_HELP
    ),
    "reviewer_screening_form_too_large": (
        "The URL or form body exceeds 65536 bytes. Nothing was written."
    ),
    "reviewer_screening_form_encoding": "Use a UTF-8 URL-encoded form without content encoding.",
    "reviewer_screening_form_origin": (
        "The submission did not come from this exact origin (scheme, host and port), or its "
        "Origin/Fetch Metadata headers are missing, ambiguous or cross-origin. Open the local "
        "page again. Nothing was written."
    ),
    "reviewer_screening_form_method": "Use GET to inspect or an explicit same-origin POST.",
    "screening_blind_mode_on": (
        "Blind mode is on for this collection. Other reviewers' decisions, conflicts, "
        "resolutions and agreement stay hidden; a page request cannot turn blind mode off. "
        "Change the screening protocol deliberately on the team page."
    ),
    "screening_revision_conflict": (
        "Your decision changed after this form opened. Nothing was overwritten. Reopen the "
        "member, inspect your latest decision, then decide again."
    ),
    "screening_revision_exhausted": "The decision revision cannot increase. Nothing was written.",
    "screening_reviewer_limit": (
        f"This collection already has {MAX_NAMED_REVIEWERS} named reviewers. Nothing was written."
    ),
    "screening_no_conflict": (
        "Reviewer decisions on this member no longer disagree. Nothing was written; reopen the "
        "team page."
    ),
    "screening_conflict_changed": (
        "A reviewer decision changed after the team page opened. Nothing was written; reopen "
        "the conflict and inspect the current decisions."
    ),
    "screening_resolution_revision_conflict": (
        "Another resolution was recorded after the team page opened. Nothing was overwritten."
    ),
    "screening_resolution_revision_exhausted": "The resolution revision cannot increase.",
    "screening_protocol_revision_conflict": (
        "The protocol changed after the team page opened. Nothing was overwritten."
    ),
    "screening_protocol_unchanged": (
        "Choose a blind mode or reason rule different from the current protocol."
    ),
    "screening_protocol_revision_exhausted": "The protocol revision cannot increase.",
    "invalid_reviewer_screening_record": "Saved reviewer decisions are invalid; nothing repaired.",
    "invalid_reviewer_screening_history_record": (
        "Saved reviewer history is invalid or inconsistent; nothing was repaired."
    ),
    "invalid_screening_resolution_record": "Saved resolutions are invalid; nothing repaired.",
    "invalid_screening_protocol_record": "The saved protocol is invalid; nothing repaired.",
    "invalid_screening_priority": "The ordering could not be computed consistently.",
    "screening_priority_too_large": "The complete ordering exceeds its byte limit.",
    "screening_response_too_large": "The complete response exceeds its byte limit.",
    "screening_history_response_too_large": "The history window exceeds its byte limit.",
    "html_page_too_large": "The complete escaped page exceeds 1 MiB. No partial page is shown.",
    "html_text_not_representable": (
        "Saved text cannot be displayed faithfully as UTF-8 HTML. Use the JSON/Python readers."
    ),
}


def _failure(status: int, code: str, root_path: str) -> HTMLResponse:
    return _error(
        status,
        code,
        "Independent screening unavailable",
        _FAILURES.get(code, "The operation failed. No success is being reported."),
        navigation=_link(
            "back-to-collections", root_path + "/explore/collections", "Reopen a saved collection"
        ),
        dual_review=True,
    )


class _DualExplorerRoute(APIRoute):
    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None:
        if self.methods is not None and scope["method"] not in self.methods:
            root_path = scope.get("root_path", "").rstrip("/")
            response = _failure(405, "reviewer_screening_form_method", root_path)
            response.headers["Allow"] = "POST" if self.path == PROTOCOL_PATH else "GET, POST"
            await response(scope, receive, send)
            return
        await super().handle(scope, receive, send)


def _code(value: str, css_class: str) -> str:
    return f'<code class="{css_class}" translate="no">{_escaped(value)}</code>'


def _blind_notice(blind: bool, collection_blind: bool) -> str:
    if blind:
        source = "for this collection" if collection_blind else "at your request"
        return (
            f'<p class="notice" id="blind-mode">Blind mode is on {source}. Until you record '
            "your own decision on a paper at this collection revision, nothing about other "
            "reviewers' decisions on it is shown: not their decisions, reasons, revisions or "
            "timestamps, whether they decided at all, conflicts, resolutions or the team "
            "outcome. Your counts and ordering use only your own decisions. A collection edit "
            "makes decisions stale and hides members again.</p>"
        )
    return (
        '<p class="notice" id="blind-mode">Blind mode is off for this collection: other '
        "reviewers' latest decisions and the team outcome are shown for every member. New "
        "decisions recorded now are marked as not blinded.</p>"
    )


def _decision_summary(decisions: tuple[ReviewerDecision, ...]) -> str:
    if not decisions:
        return "none"
    return ", ".join(
        f'<span class="other-decision">{_escaped(entry.reviewer)}: {entry.decision}'
        + ("" if entry.current else " (stale)")
        + "</span>"
        for entry in decisions
    )


def _final_summary(final: FinalDecision) -> str:
    decision = f" {final.decision}" if final.decision else ""
    source = f" by {final.source}" if final.source else ""
    return (
        f'<strong class="final-status">{final.status}</strong>{decision}{source}; '
        f'resolution: <span class="resolution-status">{final.resolution_status}</span>'
    )


def _reviewer_decision(entry: ReviewerDecision, css_class: str) -> Iterator[str]:
    blinded = {True: "yes", False: "no", None: "unknown (unlabeled default)"}[entry.blinded]
    yield (
        f'<article class="{css_class}"><h3>Reviewer {_code(entry.reviewer, "reviewer-label")}'
        f'</h3><p>Decision: <strong class="{css_class}-decision">{entry.decision}</strong>'
        + ("" if entry.current else " (stale: earlier collection revision)")
        + f"; decision revision {entry.revision}; reviewed collection revision "
        f"{entry.collection_revision}; recorded under enforced blinding: {blinded}; at "
        f'<time datetime="{entry.updated_at.isoformat()}">{entry.updated_at.isoformat()}'
        "</time>.</p>"
    )
    yield _literal(entry.reason, f"{css_class}-reason")
    yield "</article>"


def _resolution(final: FinalDecision, prefix: str) -> Iterator[str]:
    resolution = final.resolution
    if resolution is None:
        yield "<p>No resolution has been recorded for this member.</p>"
        return
    state = (
        "current: it is the final decision"
        if final.resolution_status == "current"
        else "outdated: a reviewer decision changed afterwards, so it is not final"
    )
    yield (
        f'<p id="{prefix}resolution">Latest resolution revision {resolution.revision} by '
        f"{_code(resolution.resolver, 'resolver-label')}: "
        f'<strong class="resolution-decision">{resolution.decision}</strong> ({state}); '
        f"recorded at <time>{resolution.updated_at.isoformat()}</time>.</p>"
    )
    yield _literal(resolution.rationale, "resolution-rationale")


def _entry_fragments(options: _ReviewerOpen, root_path: str) -> Iterator[str]:
    yield '<nav aria-label="Independent screening navigation">'
    yield _collection_link(options.navigation(), root_path)
    yield _link(
        "team-view",
        _url(
            root_path + TEAM_PATH,
            {
                "collection_id": options.collection_id,
                "collection_revision": options.collection_revision,
                "collection_return": options.collection_return,
            },
        ),
        "Team view: progress, conflicts and agreement",
    )
    yield "</nav>"
    yield _identity(options.collection_id, "collection-id")
    yield (
        f"<p>Collection revision: {options.collection_revision}. Each reviewer screens every "
        "member independently under their own label; the unlabeled collection pages remain "
        "the default reviewer's workspace.</p>"
        '<form id="reviewer-label-form" method="get" '
        f'action="{_escaped(root_path + REVIEWER_PATH)}" autocomplete="off">'
        '<div class="worksheet-input"><label for="reviewer">Your '
        "pseudonymous reviewer label</label>"
        f'<input id="reviewer" name="reviewer" required maxlength="32" '
        f'pattern="{_escaped(_LABEL_PATTERN)}" spellcheck="false" '
        f'aria-describedby="reviewer-help"><p id="reviewer-help">{_escaped(_LABEL_HELP)}</p>'
        "</div>"
        f'<input type="hidden" name="collection_id" value="{_escaped(options.collection_id)}">'
        f'<input type="hidden" name="collection_revision" value="{options.collection_revision}">'
    )
    if options.collection_return is not None:
        yield (
            f'<input type="hidden" name="collection_return" '
            f'value="{_escaped(options.collection_return)}">'
        )
    yield (
        '<div class="collection-choice worksheet-input"><label for="blind">'
        '<input id="blind" type="checkbox" name="blind" value="true">Keep other reviewers '
        "hidden from me even if the team turns blind mode off</label></div>"
        '<div class="actions"><button type="submit">Open my independent queue</button></div>'
        "</form>"
    )


def _queue_card(item: ReviewerItem, index: int, options: _ReviewerOpen, root_path: str) -> str:
    parts = [
        f'<article class="reviewer-member"><h3>Member {index}</h3>',
        _identity(item.document_id, "document-id"),
        _literal(item.title, "member-title"),
        _truncation("Title", item.title_truncated, MAX_TITLE_CHARACTERS),
        f'<p>Your status: <strong class="own-status">{item.status}</strong>.</p>',
    ]
    if item.others_hidden or item.other_decisions is None or item.final is None:
        parts.append(
            '<p class="others-hidden">Other reviewers: hidden until you decide (blind mode).</p>'
        )
    else:
        parts.append(
            "<p>Other reviewers: " + _decision_summary(item.other_decisions) + ".</p>"
            "<p>Team outcome: " + _final_summary(item.final) + ".</p>"
        )
    parts.append(
        _link(
            f"open-member-{index}",
            _url(root_path + REVIEWER_PATH, options.state(document_id=item.document_id)),
            "Open your decision form",
        )
    )
    parts.append("</article>")
    return "".join(parts)


def _queue_fragments(
    container: AppContainer, options: _ReviewerOpen, reviewer: str, root_path: str
) -> Iterator[str]:
    queue = container.dual_screening.reviewer_queue(
        options.collection_id,
        reviewer=reviewer,
        collection_revision=options.collection_revision,
        blind=True if options.blind else None,
        limit=100,
    )
    yield '<nav aria-label="Independent screening navigation">'
    yield _collection_link(options.navigation(), root_path)
    yield _link(
        "change-reviewer",
        _url(root_path + REVIEWER_PATH, options.state(reviewer=None, blind=None)),
        "Use a different reviewer label",
    )
    yield _link(
        "reviewer-priority",
        _url(root_path + REVIEWER_PATH, options.state(order="priority")),
        "Most likely relevant first, from your own decisions only",
    )
    yield _link(
        "reviewer-queue-json",
        _url(
            f"{root_path}/collections/{queue.collection_id}/reviewer-screening",
            {
                "reviewer": queue.reviewer,
                "collection_revision": queue.collection_revision,
                "blind": options.blind,
                "limit": 100,
            },
        ),
        "Read your queue as JSON (a separate read)",
    )
    yield "</nav>"
    yield f"<h2>Reviewer {_code(queue.reviewer, 'reviewer-label')}</h2>"
    yield _identity(queue.collection_id, "collection-id")
    yield f"<p>Collection revision: {queue.collection_revision}.</p>"
    yield _blind_notice(queue.blind, queue.collection_blind)
    yield "<h2>Your own decisions (all members)</h2><dl>"
    for status, count in queue.counts.model_dump().items():
        yield f'<dt>{status}</dt><dd id="own-count-{status}">{count}</dd>'
    yield "</dl><h2>Members in saved order</h2>"
    for index, item in enumerate(queue.items, 1):
        yield _queue_card(item, index, options, root_path)


def _priority_card(
    item: PriorityItem, prefix: str, options: _ReviewerOpen, root_path: str
) -> Iterator[str]:
    yield f'<article class="priority-item"><h3>Rank {item.rank}</h3>'
    yield _identity(item.document_id, "document-id")
    yield _literal(item.title, "member-title")
    yield f'<p>Your status: <strong class="own-status">{item.status}</strong>.</p>'
    if item.score is not None:
        terms = ", ".join(_escaped(term.term) for term in item.top_terms) or "none"
        yield (
            f'<p>Score from your own decisions: <strong class="priority-score">{item.score:.3f}'
            f"</strong>; top terms: {terms}.</p>"
        )
    yield _link(
        f"{prefix}open-member-{item.rank}",
        _url(root_path + REVIEWER_PATH, options.state(document_id=item.document_id)),
        "Open your decision form",
    )
    yield "</article>"


def _priority_fragments(
    container: AppContainer, options: _ReviewerOpen, reviewer: str, root_path: str
) -> Iterator[str]:
    result = container.dual_screening.reviewer_priority(
        options.collection_id,
        reviewer=reviewer,
        collection_revision=options.collection_revision,
    )
    priority = result.priority
    yield '<nav aria-label="Independent screening navigation">'
    yield _link(
        "reviewer-queue",
        _url(root_path + REVIEWER_PATH, options.state()),
        "Back to your queue in saved order",
    )
    yield _link(
        "reviewer-priority-json",
        _url(
            f"{root_path}/collections/{priority.collection_id}/reviewer-screening-priority",
            {"reviewer": result.reviewer, "collection_revision": priority.collection_revision},
        ),
        "Read this ordering as JSON (a separate read)",
    )
    yield "</nav>"
    yield f"<h2>Reviewer {_code(result.reviewer, 'reviewer-label')}</h2>"
    yield (
        '<p class="notice" id="priority-scope">Ordering only, trained on your own current '
        "include and exclude decisions; other reviewers' decisions are never used, so this "
        "page reveals nothing about them. Nothing is recorded or suggested.</p>"
        f'<p>Model status: <strong class="model-status">{priority.model_status}</strong>; '
        f"{priority.training_relevant} include(s) and {priority.training_irrelevant} "
        "exclude(s) of yours.</p>"
        f"<h2>Your pending members: {len(priority.items)}</h2>"
    )
    for item in priority.items:
        yield from _priority_card(item, "", options, root_path)
    yield f"<h2>Your unsure members: {len(priority.unsure_items)}</h2>"
    for item in priority.unsure_items:
        yield from _priority_card(item, "unsure-", options, root_path)


def _member_fragments(
    container: AppContainer, options: _ReviewerOpen, reviewer: str, document_id: str, root_path: str
) -> Iterator[str]:
    member = container.dual_screening.reviewer_member(
        options.collection_id,
        document_id,
        reviewer=reviewer,
        collection_revision=options.collection_revision,
        blind=True if options.blind else None,
        history_limit=HISTORY_LIMIT,
    )
    item = member.item
    yield '<nav aria-label="Independent screening navigation">'
    yield _link(
        "reviewer-queue",
        _url(root_path + REVIEWER_PATH, options.state()),
        "Back to your independent queue",
    )
    yield _link(
        "reviewer-paper",
        _url(
            root_path + "/explore/document",
            {"document_id": item.document_id, "collection_return": options.navigation()},
        ),
        "Read this exact paper / choose source context",
    )
    yield _link(
        "reviewer-history-json",
        _url(
            f"{root_path}/collections/{member.collection_id}/reviewer-screening-history",
            {
                "reviewer": member.reviewer,
                "document_id": item.document_id,
                "collection_revision": member.collection_revision,
            },
        ),
        "Read your complete history for this member as JSON",
    )
    yield "</nav>"
    yield _literal(member.collection_name, "collection-name")
    yield _identity(member.collection_id, "collection-id")
    yield f"<h2>Reviewer {_code(member.reviewer, 'reviewer-label')}</h2>"
    yield _identity(item.document_id, "document-id")
    yield _literal(item.title, "member-title")
    yield (
        f"<p>Collection revision: {member.collection_revision}; your status: "
        f'<strong class="own-status">{item.status}</strong>.</p>'
    )
    if options.saved_revision is not None:
        if item.review is not None and item.review.revision == options.saved_revision:
            yield (
                '<p class="notice" role="status" id="reviewer-saved">Your stored decision '
                f"revision {options.saved_revision} is confirmed.</p>"
            )
        else:
            yield (
                '<p class="notice">The requested confirmation is not your latest decision. No '
                "success is inferred from the URL; inspect your history before retrying.</p>"
            )
    yield _blind_notice(member.blind, member.collection_blind)
    yield "<h2>Your latest decision</h2>"
    if item.review is None:
        yield "<p>You have not recorded a decision for this member.</p>"
    else:
        yield from _history_review(item.review, "own-review")
    yield (
        "<h2>Record your independent decision</h2>"
        '<p id="reviewer-decision-help">Read the paper first. Choose a decision and write your '
        "own reason; nothing is preselected or suggested. This appends one decision under your "
        "label only.</p>"
        '<form id="reviewer-form" method="post" enctype="application/x-www-form-urlencoded" '
        f'action="{_escaped(root_path + REVIEWER_PATH)}" accept-charset="UTF-8" '
        'autocomplete="off" aria-describedby="reviewer-decision-help">'
    )
    selection = _ReviewerSelection(
        collection_id=member.collection_id,
        collection_revision=member.collection_revision,
        reviewer=member.reviewer,
        document_id=item.document_id,
        blind=options.blind,
        collection_return=options.collection_return,
    )
    token = json.dumps(selection.model_dump(), ensure_ascii=True, separators=(",", ":"))
    expected = item.review.revision if item.review else 0
    yield (
        f'<input type="hidden" name="selection" value="{_escaped(token)}">'
        f'<input type="hidden" name="expected_decision_revision" value="{expected}">'
        '<div><label for="decision">Your decision</label>'
        '<select id="decision" name="decision" required>'
        '<option value="" selected>Choose explicitly</option>'
        '<option value="include">Include</option><option value="exclude">Exclude</option>'
        '<option value="unsure">Unsure</option></select></div>'
        '<div class="worksheet-input"><label for="reason">Reason for your decision</label>'
        f'<textarea id="reason" name="reason" required maxlength="{MAX_REASON_CHARACTERS}" '
        'spellcheck="false"></textarea>'
        "<p>1-1000 characters, nonblank; no credentials or personal data.</p></div>"
        '<div class="actions"><button type="submit">Record my decision</button></div></form>'
        "<h2>Other reviewers</h2>"
    )
    if item.others_hidden or item.other_decisions is None or item.final is None:
        yield (
            '<p class="notice" id="others-hidden">Hidden by blind mode until you record your '
            "own decision on this member at this collection revision.</p>"
        )
    else:
        if not item.other_decisions:
            yield "<p>No other reviewer has decided this member.</p>"
        for entry in item.other_decisions:
            yield from _reviewer_decision(entry, "other-review")
        yield "<p>Team outcome: " + _final_summary(item.final) + ".</p>"
        yield from _resolution(item.final, "member-")
    yield f"<h2>Your recent history (up to {HISTORY_LIMIT})</h2>"
    if not member.recent_history:
        yield "<p>No retained decisions of yours for this member.</p>"
    for review in member.recent_history:
        yield f"<article><h3>Your decision revision {review.revision}</h3>"
        yield from _history_review(review, "own-history")
        yield "</article>"
    if member.older_history:
        yield "<p>Older revisions are available through the JSON history link.</p>"


def _reviewer_page(container: AppContainer, options: _ReviewerOpen, root_path: str) -> HTMLResponse:
    reviewer = options.reviewer
    if reviewer is None:
        return _page(
            "Independent screening: choose your label",
            _entry_fragments(options, root_path),
            dual_review=True,
        )
    if options.document_id is not None:
        member = _member_fragments(container, options, reviewer, options.document_id, root_path)
        return _page("Your independent decision", iter(list(member)), dual_review=True)
    if options.order == "priority":
        ordered = _priority_fragments(container, options, reviewer, root_path)
        return _page(
            "Your most likely relevant papers first", iter(list(ordered)), dual_review=True
        )
    queue = _queue_fragments(container, options, reviewer, root_path)
    return _page("Your independent screening queue", iter(list(queue)), dual_review=True)


def _same_origin(request: Request) -> None:
    try:
        require_same_origin(request)
    except ValueError as exc:
        raise CollectionError("reviewer_screening_form_origin", "", 403) from exc
    try:
        require_urlencoded(request)
    except ValueError as exc:
        raise CollectionError("reviewer_screening_form_encoding", "", 415) from exc


async def _request_fields(request: Request, *, max_fields: int) -> tuple[bool, dict[str, str]]:
    """Return whether this is a POST and its strictly parsed fields."""
    post = request.method == "POST"
    if post:
        _same_origin(request)
    raw_query = request.scope.get("query_string", b"")
    if len(raw_query) > MAX_QUERY_BYTES:
        raise FormTooLargeError()
    if post and raw_query:
        raise ValueError("POST fields belong only in the body.")
    body = await read_body(request, max_bytes=MAX_FORM_BYTES)
    if not post and body:
        raise ValueError("GET cannot accept a body.")
    return post, parse_fields(body if post else raw_query, max_fields=max_fields)


def _handled(exc: Exception, root_path: str) -> HTMLResponse:
    if isinstance(exc, CollectionError):
        return _failure(exc.status_code, exc.code, root_path)
    if isinstance(exc, sqlite3.Error):
        return _failure(503, "screening_storage_unavailable", root_path)
    if isinstance(exc, FormTooLargeError):
        return _failure(413, "reviewer_screening_form_too_large", root_path)
    if isinstance(exc, _PageTooLargeError):
        return _failure(413, "html_page_too_large", root_path)
    if isinstance(exc, _UnrepresentableTextError):
        return _failure(409, "html_text_not_representable", root_path)
    return _failure(422, "invalid_reviewer_screening_form", root_path)


router = APIRouter(route_class=_DualExplorerRoute, tags=["Independent screening browser"])
_WRITE_RESPONSES: dict[int | str, dict[str, str]] = {
    303: {"description": "One record persisted; local confirmation redirect"},
    403: {"description": "Missing, ambiguous or cross-origin browser origin"},
    404: {"description": "Exact collection not found"},
    405: {"description": "Unsupported method"},
    409: {"description": "Blind mode, stale revision or basis, or invalid saved data"},
    413: {"description": "Bounded URL, body or complete HTML exceeded"},
    415: {"description": "Unsupported form encoding"},
    422: {"description": "Invalid exact selection, label, decision, reason or fields"},
    503: {"description": "Existing screening storage unavailable"},
}


@router.get(REVIEWER_PATH, response_class=HTMLResponse)
@router.post(REVIEWER_PATH, response_class=HTMLResponse, responses=_WRITE_RESPONSES)
async def explore_reviewer_screening(request: Request) -> Response:
    """Your own blind queue, member form and ordering; POST records one decision."""
    root_path = request.scope.get("root_path", "").rstrip("/")
    container: AppContainer = request.app.state.container
    try:
        post, fields = await _request_fields(request, max_fields=8)
        if not post:
            options = _ReviewerOpen.model_validate(fields)
            return await run_in_threadpool(_reviewer_page, container, options, root_path)
        form = _ReviewerForm.model_validate(fields)
        selection = _ReviewerSelection.model_validate(
            json.loads(form.selection, object_pairs_hook=_return_pairs)
        )
        submission = ReviewerSubmission(
            reviewer=selection.reviewer,
            collection_revision=selection.collection_revision,
            expected_decision_revision=form.expected_decision_revision,
            decision=form.decision,
            reason=form.reason,
        )
        # Build the bounded local redirect before the only write; reasons never enter URLs.
        redirect = RedirectResponse(
            _url(
                root_path + REVIEWER_PATH,
                {
                    **selection.model_dump(),
                    "saved_revision": submission.expected_decision_revision + 1,
                },
            ),
            status_code=303,
            headers=_WRITE_HEADERS,
        )
        await run_in_threadpool(
            container.dual_screening.submit,
            selection.collection_id,
            selection.document_id,
            submission,
        )
        return redirect
    except (
        CollectionError,
        sqlite3.Error,
        FormTooLargeError,
        _PageTooLargeError,
        _UnrepresentableTextError,
        ValueError,
        RecursionError,
    ) as exc:
        return _handled(exc, root_path)


def _resolution_form(
    item: ConflictItem, index: int, options: _TeamOpen, root_path: str
) -> Iterator[str]:
    selection = _ConflictSelection(
        collection_id=options.collection_id,
        collection_revision=options.collection_revision,
        collection_return=options.collection_return,
        document_id=item.document_id,
        expected_basis=item.expected_basis,
    )
    token = json.dumps(selection.model_dump(), ensure_ascii=True, separators=(",", ":"))
    action = "Record a corrected resolution" if item.final.resolution else "Record a resolution"
    yield (
        f'<form id="resolution-form-{index}" class="resolution-form" method="post" '
        f'enctype="application/x-www-form-urlencoded" action="{_escaped(root_path + TEAM_PATH)}"'
        ' accept-charset="UTF-8" autocomplete="off">'
        f'<input type="hidden" name="selection" value="{_escaped(token)}">'
        '<input type="hidden" name="expected_resolution_revision" '
        f'value="{item.resolution_revision}">'
        f'<div><label for="resolver-{index}">Resolver label</label>'
        f'<input id="resolver-{index}" name="resolver" required maxlength="32" '
        f'pattern="{_escaped(_LABEL_PATTERN)}" spellcheck="false"></div>'
        f'<div><label for="resolution-decision-{index}">Final decision</label>'
        f'<select id="resolution-decision-{index}" name="decision" required>'
        '<option value="" selected>Choose explicitly</option>'
        '<option value="include">Include</option><option value="exclude">Exclude</option>'
        "</select></div>"
        f'<div class="worksheet-input"><label for="rationale-{index}">Rationale</label>'
        f'<textarea id="rationale-{index}" name="rationale" required '
        f'maxlength="{MAX_REASON_CHARACTERS}" spellcheck="false"></textarea>'
        "<p>Appended beside the reviewer decisions; it never rewrites them. For a final "
        "exclude, this text is the final exclusion reason.</p></div>"
        f'<div class="actions"><button type="submit">{action}</button></div></form>'
    )


def _conflict_card(
    item: ConflictItem, index: int, options: _TeamOpen, root_path: str
) -> Iterator[str]:
    yield f'<article class="conflict-item"><h3>Disagreement {index}</h3>'
    yield _identity(item.document_id, "document-id")
    yield _literal(item.title, "member-title")
    yield _truncation("Title", item.title_truncated, MAX_TITLE_CHARACTERS)
    yield (
        f'<p>Conflict type: <strong class="conflict-type">{item.conflict_type}</strong>. Team '
        f"outcome: {_final_summary(item.final)}.</p>"
    )
    for entry in item.decisions:
        yield from _reviewer_decision(entry, "conflict-review")
    yield from _resolution(item.final, f"conflict-{index}-")
    yield from _resolution_form(item, index, options, root_path)
    yield "</article>"


def _protocol_form(overview: TeamOverview, options: _TeamOpen, root_path: str) -> str:
    protocol = overview.summary.protocol
    selection = _TeamSelection(
        collection_id=options.collection_id,
        collection_revision=options.collection_revision,
        collection_return=options.collection_return,
    )
    token = json.dumps(selection.model_dump(), ensure_ascii=True, separators=(",", ":"))
    return (
        "<h2>Change the screening protocol</h2>"
        '<p id="protocol-help">Turn blind mode off only after every reviewer has finished '
        "independent screening; it reveals all decisions on team pages and in reviewer pages "
        "that do not request blinding. Changing the protocol never edits a decision.</p>"
        '<form id="protocol-form" method="post" enctype="application/x-www-form-urlencoded" '
        f'action="{_escaped(root_path + PROTOCOL_PATH)}" accept-charset="UTF-8" '
        'autocomplete="off" aria-describedby="protocol-help">'
        f'<input type="hidden" name="selection" value="{_escaped(token)}">'
        '<input type="hidden" name="expected_protocol_revision" '
        f'value="{protocol.revision}">'
        '<div><label for="protocol-blind">Blind mode</label>'
        '<select id="protocol-blind" name="blind" required>'
        '<option value="" selected>Choose explicitly</option>'
        '<option value="on">On: reviewers stay independent</option>'
        '<option value="off">Off: show decisions, conflicts and agreement</option></select></div>'
        '<div><label for="protocol-reasons">Exclusion reasons must match</label>'
        '<select id="protocol-reasons" name="exclusion_reasons_must_match" required>'
        '<option value="" selected>Choose explicitly</option>'
        '<option value="no">No: two excludes agree</option>'
        '<option value="yes">Yes: different exact reasons conflict</option></select></div>'
        '<div><label for="changed-by">Your label</label>'
        f'<input id="changed-by" name="changed_by" required maxlength="32" '
        f'pattern="{_escaped(_LABEL_PATTERN)}" spellcheck="false"></div>'
        '<div class="actions"><button type="submit">Save protocol change</button></div></form>'
    )


def _team_fragments(container: AppContainer, options: _TeamOpen, root_path: str) -> Iterator[str]:
    overview = container.dual_screening.team_overview(
        options.collection_id, collection_revision=options.collection_revision
    )
    summary = overview.summary
    protocol = summary.protocol
    yield '<nav aria-label="Team screening navigation">'
    yield _collection_link(options.navigation(), root_path)
    yield _link(
        "reviewer-entry",
        _url(
            root_path + REVIEWER_PATH,
            {
                "collection_id": options.collection_id,
                "collection_revision": options.collection_revision,
                "collection_return": options.collection_return,
            },
        ),
        "Open an independent reviewer queue",
    )
    yield "</nav>"
    yield _literal(overview.collection_name, "collection-name")
    yield _identity(overview.collection_id, "collection-id")
    yield f"<p>Collection revision: {overview.collection_revision}.</p>"
    if options.saved_protocol_revision is not None:
        yield (
            '<p class="notice" role="status" id="protocol-saved">Protocol revision '
            f"{options.saved_protocol_revision} confirmed.</p>"
            if protocol.revision == options.saved_protocol_revision
            else '<p class="notice">The requested protocol revision is not current. Inspect '
            "the protocol below before retrying.</p>"
        )
    if options.saved_document_id is not None:
        saved = next(
            (
                item
                for item in overview.conflicts or ()
                if item.document_id == options.saved_document_id
                and item.final.resolution is not None
                and item.final.resolution.revision == options.saved_resolution_revision
            ),
            None,
        )
        yield (
            '<p class="notice" role="status" id="resolution-saved">Resolution revision '
            f"{options.saved_resolution_revision} confirmed for the member below.</p>"
            + _identity(options.saved_document_id, "saved-document-id")
            if saved is not None
            else '<p class="notice">The requested resolution is not the latest one on this '
            "page. Inspect the conflict before retrying.</p>"
        )
    yield (
        f"<h2>Protocol revision {protocol.revision}</h2><p>Blind mode: "
        f'<strong id="protocol-blind-state">{"on" if protocol.blind else "off"}</strong>; '
        "exclusion reasons must match: "
        f"<strong>{'yes' if protocol.exclusion_reasons_must_match else 'no'}</strong>"
        + (
            "."
            if protocol.changed_by is None or protocol.updated_at is None
            else f"; changed by {_code(protocol.changed_by, 'reviewer-label')} at "
            f"<time>{protocol.updated_at.isoformat()}</time>."
        )
        + "</p><h2>Reviewer progress</h2>"
        "<table><caption>Decided members per label, never which members</caption><thead><tr>"
        '<th scope="col">Reviewer</th><th scope="col">Current decisions</th>'
        '<th scope="col">Stale decisions</th></tr></thead><tbody>'
    )
    for entry in summary.reviewers:
        label = entry.reviewer if entry.labeled else f"{entry.reviewer} (unlabeled)"
        yield (
            f"<tr><td>{_code(label, 'progress-reviewer')}</td>"
            f"<td>{entry.current_decisions}</td><td>{entry.stale_decisions}</td></tr>"
        )
    yield "</tbody></table>"
    if not summary.reviewers:
        yield "<p>No reviewer has recorded a decision yet.</p>"
    if summary.hidden or summary.final_counts is None or summary.conflicts is None:
        yield (
            '<p class="notice" id="team-hidden">Blind mode is on: decisions, conflicts, '
            "resolutions, team outcomes and agreement stay hidden here and in the JSON team "
            "routes. Turn blind mode off below only when independent screening is done.</p>"
        )
        yield _protocol_form(overview, options, root_path)
        return
    yield '<nav aria-label="Team downloads">'
    for identifier, endpoint, format, label in (
        ("agreement-json", "screening-agreement", "json", "Agreement report JSON"),
        ("agreement-markdown", "screening-agreement", "markdown", "Agreement report Markdown"),
        ("agreement-csv", "screening-agreement", "csv", "Agreement statistics CSV"),
        ("final-json", "screening-final-decisions", "json", "Final decisions JSON"),
        ("final-csv", "screening-final-decisions", "csv", "Final decisions CSV"),
    ):
        yield _link(
            f"download-{identifier}",
            _url(
                f"{root_path}/collections/{overview.collection_id}/{endpoint}",
                {"collection_revision": overview.collection_revision, "format": format},
            ),
            label,
        )
    yield "</nav><h2>Inter-rater agreement</h2>"
    if not summary.pairs:
        yield "<p>Fewer than two reviewers have current decisions; nothing is compared.</p>"
    for pair in summary.pairs or ():
        yield (
            f'<p class="pair-agreement">{_code(pair.reviewer_a, "reviewer-label")} and '
            f"{_code(pair.reviewer_b, 'reviewer-label')}, {pair.scale}: "
            f'<span class="agreement-line">{_escaped(agreement_line(pair))}</span>.</p>'
        )
    counts = summary.final_counts
    conflicts = summary.conflicts
    yield (
        '<p class="notice">Kappa uses independent reviewer decisions only; resolutions never '
        "change it. Landis-Koch labels are conventional, not quality thresholds.</p>"
        "<h2>Team outcomes</h2><dl>"
    )
    for label, value, identifier in (
        ("Final include", counts.final_include, "final-include"),
        ("Final exclude", counts.final_exclude, "final-exclude"),
        ("Needs human review", counts.needs_review, "needs-review"),
        ("Independently double-screened", counts.double_screened, "double-screened"),
        ("Unresolved conflicts", conflicts.unresolved, "unresolved-conflicts"),
        ("Resolved conflicts", conflicts.resolved, "resolved-conflicts"),
    ):
        yield f'<dt>{label}</dt><dd id="{identifier}">{value}</dd>'
    items = overview.conflicts or ()
    yield f"</dl><h2>Disagreements: {len(items)}</h2>"
    if not items:
        yield "<p>No current disagreements between independent decisions.</p>"
    for index, item in enumerate(items, 1):
        yield from _conflict_card(item, index, options, root_path)
    yield _protocol_form(overview, options, root_path)


def _team_page(container: AppContainer, options: _TeamOpen, root_path: str) -> HTMLResponse:
    fragments = list(_team_fragments(container, options, root_path))
    return _page("Team screening: conflicts and agreement", iter(fragments), dual_review=True)


@router.get(TEAM_PATH, response_class=HTMLResponse)
@router.post(TEAM_PATH, response_class=HTMLResponse, responses=_WRITE_RESPONSES)
async def explore_screening_conflicts(request: Request) -> Response:
    """Team progress; once unblinded, conflicts, agreement and resolution forms."""
    root_path = request.scope.get("root_path", "").rstrip("/")
    container: AppContainer = request.app.state.container
    try:
        post, fields = await _request_fields(request, max_fields=6)
        if not post:
            options = _TeamOpen.model_validate(fields)
            return await run_in_threadpool(_team_page, container, options, root_path)
        form = _ResolutionForm.model_validate(fields)
        selection = _ConflictSelection.model_validate(
            json.loads(form.selection, object_pairs_hook=_return_pairs)
        )
        submission = ResolutionSubmission(
            resolver=form.resolver,
            collection_revision=selection.collection_revision,
            expected_resolution_revision=form.expected_resolution_revision,
            expected_basis=selection.expected_basis,
            decision=form.decision,
            rationale=form.rationale,
        )
        redirect = RedirectResponse(
            _url(
                root_path + TEAM_PATH,
                {
                    "collection_id": selection.collection_id,
                    "collection_revision": selection.collection_revision,
                    "collection_return": selection.collection_return,
                    "saved_document_id": selection.document_id,
                    "saved_resolution_revision": submission.expected_resolution_revision + 1,
                },
            ),
            status_code=303,
            headers=_WRITE_HEADERS,
        )
        await run_in_threadpool(
            container.dual_screening.resolve,
            selection.collection_id,
            selection.document_id,
            submission,
        )
        return redirect
    except (
        CollectionError,
        sqlite3.Error,
        FormTooLargeError,
        _PageTooLargeError,
        _UnrepresentableTextError,
        ValueError,
        RecursionError,
    ) as exc:
        return _handled(exc, root_path)


@router.post(PROTOCOL_PATH, response_class=HTMLResponse, responses=_WRITE_RESPONSES)
async def explore_screening_protocol(request: Request) -> Response:
    """Explicitly change blind mode or the exclusion-reason rule; never a decision."""
    root_path = request.scope.get("root_path", "").rstrip("/")
    container: AppContainer = request.app.state.container
    try:
        _, fields = await _request_fields(request, max_fields=5)
        form = _ProtocolForm.model_validate(fields)
        selection = _TeamSelection.model_validate(
            json.loads(form.selection, object_pairs_hook=_return_pairs)
        )
        change = ProtocolChange(
            blind=form.blind == "on",
            exclusion_reasons_must_match=form.exclusion_reasons_must_match == "yes",
            expected_protocol_revision=form.expected_protocol_revision,
            changed_by=form.changed_by,
        )
        redirect = RedirectResponse(
            _url(
                root_path + TEAM_PATH,
                {
                    **selection.model_dump(),
                    "saved_protocol_revision": change.expected_protocol_revision + 1,
                },
            ),
            status_code=303,
            headers=_WRITE_HEADERS,
        )
        await run_in_threadpool(
            container.dual_screening.set_protocol, selection.collection_id, change
        )
        return redirect
    except (
        CollectionError,
        sqlite3.Error,
        FormTooLargeError,
        _PageTooLargeError,
        _UnrepresentableTextError,
        ValueError,
        RecursionError,
    ) as exc:
        return _handled(exc, root_path)
