"""Native human verdicts must preview without writes and reopen the exact saved opinion."""

import importlib
import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from unittest.mock import Mock
from urllib.parse import parse_qs, urlencode, urlsplit
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from scripts.demo_corpus_explorer import browsing_guards
from scripts.demo_evidence_export import offline_settings
from starlette.types import Message, Scope

from agent.models import AgentAnswer
from agent.review_models import (
    MAX_REVIEW_SEQUENCE,
    AnswerReviewError,
    ReviewDecision,
    ReviewSubmission,
    SavedAnswerReview,
)
from api import explorer
from api.application import create_app
from storage.evidence_export import EvidenceExportError
from tests.test_corpus_explorer import (
    ExplorerAPI,
    ExplorerDOM,
    assert_headers,
    href,
    identities,
    no_work,
    values,
)
from tests.test_corpus_explorer import api as api
from tests.test_evidence_html import Node
from tests.test_review_notebook import COMMENT, TEXT, annotate, fixture_run, opinion
from tests.test_run_comparison import ComparisonAPI, _save, _source
from tests.test_run_explorer import reader_headers

COMPOSER = "/explore/run/review"
SAVE = COMPOSER + "/save"
SAVED = COMPOSER + "/saved"
ORIGIN = {"origin": "http://testserver", "sec-fetch-site": "same-origin"}
FORM_HEADERS = {**ORIGIN, "content-type": "application/x-www-form-urlencoded"}


def controls(root: Node, identifier: str = "review-draft") -> dict[str, str]:
    return {
        node.attrs["name"]: node.attrs.get("value", "")
        for node in root.by_id(identifier).by_tag("input")
        if node.attrs.get("type") == "hidden"
    }


def opened(api: ExplorerAPI, run_id: str = "saved", **params: str | int) -> Node:
    notebook = api.page("/explore/run/notebook", run_id=run_id, **params)
    return api.page(href(notebook, "review-answer"), referrer_policy="same-origin")


def draft(
    root: Node,
    *,
    decision: str = "needs_revision",
    comment: str = COMMENT,
    cite: tuple[int, ...] = (1,),
) -> dict[str, str]:
    fields = {**controls(root), "decision": decision, "comment": comment}
    for rank in cite:
        checkbox = root.by_id(f"cite-{rank}")
        fields[checkbox.attrs["name"]] = checkbox.attrs["value"]
    return fields


def preview(api: ExplorerAPI, root: Node, **updates: str) -> Node:
    response = api.client.post(COMPOSER, data={**draft(root), **updates}, headers=ORIGIN)
    assert response.status_code == 200, response.text[:1000]
    assert_headers(response, referrer_policy="same-origin")
    return ExplorerDOM(response.text).root


def save(api: ExplorerAPI, page: Node) -> httpx.Response:
    return api.client.post(
        page.by_id("review-save").attrs["action"],
        data=controls(page, "review-save"),
        headers=ORIGIN,
        follow_redirects=False,
    )


def failure(response: httpx.Response, status: int) -> None:
    assert response.status_code == status, response.text[:1000]
    assert_headers(response, referrer_policy="same-origin")
    root = ExplorerDOM(response.text).root
    assert not root.by_tag("form") and not root.by_class("review-comment")
    assert "PRIVATE_" not in response.text
    assert "location" not in response.headers and "set-cookie" not in response.headers


def test_native_route_opens_without_a_default_decision_or_a_write(api: ExplorerAPI) -> None:
    bundle = fixture_run(api)
    before = api.path.read_bytes()
    response = api.client.get(COMPOSER, params={"selection": json.dumps({"run_id": bundle.run_id})})
    assert response.status_code == 200
    assert_headers(response, referrer_policy="same-origin")
    page = ExplorerDOM(response.text).root
    form = page.by_id("review-draft")
    assert form.attrs["method"] == "post" and form.attrs["action"] == COMPOSER
    assert UUID(controls(page)["review_id"])
    assert page.by_id("comment").text() == ""
    options = page.by_id("decision").by_tag("option")
    assert [option.attrs["value"] for option in options] == [
        "",
        "accepted",
        "needs_revision",
        "rejected",
    ]
    assert not any("selected" in item.attrs for item in options[1:])
    assert not any("checked" in item.attrs for item in form.by_tag("input"))
    assert values(page, "answer-text") == [bundle.answer.answer]
    assert values(page, "passage-text") == [TEXT]
    assert values(page, "bundle-warning") == bundle.warnings
    assert api.path.read_bytes() == before


def test_reader_and_empty_notebook_discover_the_review_composer(api: ExplorerAPI) -> None:
    bundle = fixture_run(api)
    notebook = api.page("/explore/run/notebook", run_id=bundle.run_id)
    reader = reader_headers(api.client.get(href(notebook, "back-to-evidence")))
    for parent in (notebook, reader):
        page = api.page(href(parent, "review-answer"), referrer_policy="same-origin")
        assert identities(page, "run-id") == [bundle.run_id]
        assert values(page, "passage-text") == [TEXT]
        assert "No authenticated reviewer" in page.text()


@pytest.mark.parametrize("decision", list(ReviewDecision))
def test_preview_then_save_preserves_the_exact_human_submission_and_retry(
    api: ExplorerAPI, monkeypatch: pytest.MonkeyPatch, decision: ReviewDecision
) -> None:
    bundle = fixture_run(api)
    root = opened(api)
    create = Mock(wraps=api.container.notebook_reviews.create)
    monkeypatch.setattr(api.container.notebook_reviews, "create", create)
    before = api.path.read_bytes()
    with browsing_guards(api.container) as guards:
        page = preview(api, root, decision=decision.value)
        assert values(page, "review-comment") == [COMMENT]
        assert values(page, "review-decision") == [decision.value]
        assert "Nothing has been saved" in page.text()
        assert api.path.read_bytes() == before and not create.call_count
        fields = controls(page, "review-save")
        payload = ReviewSubmission.model_validate_json(fields["submission"])
        assert payload.review_id == UUID(controls(root)["review_id"])
        assert payload.comment == COMMENT and payload.decision == decision
        assert payload.cited_chunk_ids == (bundle.snapshot.sources[0].chunk.chunk_id,)
        assert fields["selection"].isascii() and fields["submission"].isascii()
        assert page.by_id("cite-1").attrs.get("checked") is not None
        saved = save(api, page)
        assert saved.status_code == 303
        assert saved.headers["cache-control"] == "no-store"
        assert saved.headers["referrer-policy"] == "same-origin"
        location = saved.headers["location"]
        parts = urlsplit(location)
        assert parts.path == SAVED and parts.fragment == "saved-review"
        assert not parts.netloc and not parts.scheme
        assert not {"comment", "decision", "submission"} & parse_qs(parts.query).keys()
        create.assert_called_once_with(bundle.run_id, payload)
        persisted = api.page(location)
        assert values(persisted, "review-comment") == [COMMENT]
        assert values(persisted, "review-decision") == [decision.value]
        assert values(persisted, "review-id") == [str(payload.review_id)]
        original = api.container.notebook_reviews.get_review(bundle.run_id, payload.review_id)
        after = api.path.read_bytes()
        retry = save(api, page)
        assert retry.status_code == 303 and retry.headers["location"] == location
        assert api.path.read_bytes() == after
        assert (
            api.container.notebook_reviews.get_review(bundle.run_id, payload.review_id) == original
        )
        assert len(api.container.notebook_reviews.list_reviews(bundle.run_id).reviews) == 1
        assert not any(guard.call_count for guard in guards.values())


def test_zero_sources_and_no_citations_are_explicit_not_auto_selected(api: ExplorerAPI) -> None:
    _save(ComparisonAPI(api.app, api.client, api.container, api.path), "empty", [])
    page = opened(api, "empty")
    assert "No frozen passages" in page.text()
    assert not any(node.attrs.get("type") == "checkbox" for node in page.by_tag("input"))
    response = api.client.post(COMPOSER, data=draft(page, cite=()), headers=ORIGIN)
    assert response.status_code == 200
    previewed = ExplorerDOM(response.text).root
    assert json.loads(controls(previewed, "review-save")["submission"])["cited_chunk_ids"] == []
    saved = save(api, previewed)
    assert saved.status_code == 303
    assert "This review cites no frozen passages" in api.page(saved.headers["location"]).text()


def test_both_saved_warning_sets_are_literal_and_complete(api: ExplorerAPI) -> None:
    answer = AgentAnswer(
        answer="\n  Exact answer\r\n<script>inert</script>",
        citations=[],
        claims=[],
        ungrounded=True,
        warnings=["  PRIVATE_warning is saved data, not an error. <b>literal</b>\r\n"],
    )
    _save(
        ComparisonAPI(api.app, api.client, api.container, api.path),
        "warnings",
        [_source("one")],
        answer=answer,
    )
    bundle = api.container.saved_run_evidence.export("warnings")
    page = opened(api, "warnings")
    assert values(page, "answer-text") == [answer.answer]
    assert values(page, "answer-warning") == answer.warnings
    assert values(page, "bundle-warning") == bundle.warnings
    assert values(page, "ungrounded") == ["true"]
    assert not page.by_tag("script") and not page.by_tag("img")


@pytest.mark.parametrize(
    "identifier", [".", "..", "/", "//external.invalid/x", "'\"?%#&+\\", "\u7814/e\u0301"]
)
def test_opaque_ids_and_independent_notebook_cursors_round_trip(
    api: ExplorerAPI, identifier: str
) -> None:
    bundle = fixture_run(api, run_id=identifier, chunk_id=identifier, document_id=identifier)
    for number in range(3):
        opinion(api, bundle, f"Review {number}")
        annotate(api, bundle, note=f"Note {number}")
    notebook = api.page(
        "/explore/run/notebook",
        run_id=identifier,
        review_limit=1,
        annotation_limit=1,
        limit=2,
        cursor=99,
        state="DONE",
        question=" literal %_\U0001f52c ",
        source="return\r\nfilter\0",
        title="literal /?#",
        catalog_limit=3,
        catalog_cursor="cursor/\u7814",
    )
    older = api.page(href(notebook, "reviews-next-page"))
    notebook_url = href(older, "annotations-next-page")
    older = api.page(notebook_url)
    page = api.page(href(older, "review-answer"), referrer_policy="same-origin")
    assert href(page, "back-to-notebook") == notebook_url
    assert href(page, "back-to-runs") == href(older, "back-to-runs")
    assert href(page, "back-to-catalog") == href(older, "back-to-catalog")
    assert json.loads(page.by_id("cite-1").attrs["value"]) == identifier
    saved = save(api, preview(api, page))
    assert saved.status_code == 303
    persisted = api.page(saved.headers["location"])
    assert href(persisted, "back-to-notebook") == notebook_url
    assert values(persisted, "review-comment") == [COMMENT]
    assert values(api.page(notebook_url), "review-comment") == ["Review 1"]
    sequence = values(older, "review-sequence")[0]
    exact_old = api.page(href(older, f"review-detail-{sequence}"))
    assert values(exact_old, "review-comment") == ["Review 1"]
    assert href(exact_old, "back-to-notebook") == notebook_url
    separate = api.page(href(persisted, "review-answer"), referrer_policy="same-origin")
    assert controls(separate)["review_id"] != controls(page)["review_id"]


@pytest.mark.parametrize("prefix,mounted", [("/library", False), ("/prefix/library", True)])
def test_mounts_preserve_all_form_actions_links_and_redirects(
    api: ExplorerAPI, prefix: str, mounted: bool
) -> None:
    bundle = fixture_run(api, run_id="run/../one")
    app = api.app
    if mounted:
        app = FastAPI()
        app.mount("/library", api.app)
    with TestClient(app, root_path="/prefix" if mounted else prefix) as client:
        response = client.get(
            prefix + COMPOSER,
            params={"selection": json.dumps({"run_id": bundle.run_id, "question": " Exact %_ "})},
        )
        assert response.status_code == 200
        page = ExplorerDOM(response.text).root
        previewed = client.post(
            page.by_id("review-draft").attrs["action"], data=draft(page), headers=ORIGIN
        )
        assert previewed.status_code == 200
        view = ExplorerDOM(previewed.text).root
        response = client.post(
            view.by_id("review-save").attrs["action"],
            data=controls(view, "review-save"),
            headers=ORIGIN,
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert response.headers["location"].startswith(prefix + SAVED + "?")
        persisted = client.get(response.headers["location"])
        assert persisted.status_code == 200
        assert parse_qs(urlsplit(href(ExplorerDOM(persisted.text).root, "back-to-runs")).query)[
            "question"
        ] == [" Exact %_ "]
        error = client.post(prefix + SAVE, data={"selection": "{}"}, headers=ORIGIN)
        failure(error, 422)
        for root in (page, view, ExplorerDOM(persisted.text).root, ExplorerDOM(error.text).root):
            for link in root.by_tag("a"):
                parts = urlsplit(link.attrs["href"])
                assert not parts.netloc and not parts.scheme
                assert not parts.path or parts.path.startswith(prefix + "/")
            for form in root.by_tag("form"):
                assert form.attrs["action"].startswith(prefix + "/")


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"origin": "null"},
        {"origin": "http://testserver/"},
        {"origin": "http://testserver:81"},
        {"origin": "https://testserver"},
        {"origin": "http://testserver.evil.invalid"},
        {"origin": "http://testserver@external.invalid"},
        {"origin": "http://testserver", "sec-fetch-site": "cross-site"},
        {"origin": "http://testserver", "sec-fetch-site": "same-site"},
        {"origin": "http://testserver", "sec-fetch-site": "none"},
        {"referer": "http://testserver/explore/run/review"},
        {"origin": "https://external.invalid", "x-forwarded-host": "external.invalid"},
        [("origin", "http://testserver"), ("origin", "http://testserver")],
        [("origin", "http://testserver"), ("host", "testserver"), ("host", "testserver")],
        [
            ("origin", "http://testserver"),
            ("sec-fetch-site", "same-origin"),
            ("sec-fetch-site", "same-origin"),
        ],
    ],
)
def test_origin_rejection_precedes_all_reads_and_writes(
    api: ExplorerAPI,
    monkeypatch: pytest.MonkeyPatch,
    headers: dict[str, str] | list[tuple[str, str]],
) -> None:
    fixture_run(api)
    root = opened(api)
    page = preview(api, root)
    monkeypatch.setattr(api.container.saved_run_evidence, "export", no_work)
    monkeypatch.setattr(api.container.notebook_reviews, "create", no_work)
    before = api.path.read_bytes()
    for path, fields in ((COMPOSER, draft(root)), (SAVE, controls(page, "review-save"))):
        failure(api.client.post(path, data=fields, headers=headers, follow_redirects=False), 403)
    assert api.path.read_bytes() == before


@pytest.mark.parametrize(
    "updates",
    [
        {"decision": ""},
        {"decision": "Accepted"},
        {"decision": "PRIVATE_auto_accept"},
        {"comment": ""},
        {"comment": " \r\n\t"},
        {"comment": "PRIVATE_\0invalid"},
        {"comment": "x" * 4001},
        {"review_id": ""},
        {"review_id": "PRIVATE_not_uuid"},
        {"selection": "{}"},
        {"selection": "[]"},
        {"selection": "[" * 1100},
        {"selection": '{"run_id":"saved","run_id":"different"}'},
        {"cite_1": '"PRIVATE_not_a_frozen_chunk"'},
        {"cite_1": '["chunk/one"]'},
        {"cite_01": '"chunk/one"'},
        {"cite_2": '"chunk/one"'},
        {"return_to": "//external.invalid/PRIVATE_"},
        {"created_at": "2026-10-27T01:00:00Z"},
    ],
)
def test_invalid_preview_fields_never_write(api: ExplorerAPI, updates: dict[str, str]) -> None:
    fixture_run(api)
    root = opened(api)
    before = api.path.read_bytes()
    failure(api.client.post(COMPOSER, data={**draft(root), **updates}, headers=ORIGIN), 422)
    assert api.path.read_bytes() == before


@pytest.mark.parametrize(
    "updates,status",
    [
        ({"run_id": "missing"}, 404),
        ({"run_id": ""}, 422),
        ({"run_id": " trailing "}, 422),
        ({"run_id": "control\0"}, 422),
        ({"run_id": "\ud800"}, 422),
        ({"run_id": True}, 422),
        ({"review_limit": 0}, 422),
        ({"annotation_limit": 101}, 422),
        ({"review_cursor": 0}, 422),
        ({"annotation_cursor": ""}, 422),
        ({"catalog_cursor": "\ud800"}, 422),
        ({"return_to": "https://external.invalid/PRIVATE_"}, 422),
        ({"comment": "PRIVATE_no_comments_in_get"}, 422),
    ],
)
def test_selection_rejects_invalid_exact_ids_and_unknown_navigation(
    api: ExplorerAPI, updates: dict[str, str | int], status: int
) -> None:
    fixture_run(api)
    before = api.path.read_bytes()
    failure(
        api.client.get(COMPOSER, params={"selection": json.dumps({"run_id": "saved", **updates})}),
        status,
    )
    assert api.path.read_bytes() == before


@pytest.mark.parametrize(
    "updates",
    [
        {"decision": "PRIVATE_invalid"},
        {"comment": "\ud800"},
        {"comment": "PRIVATE_\0invalid"},
        {"comment": "x" * 4001},
        {"cited_chunk_ids": ["unknown"]},
        {"cited_chunk_ids": ["chunk/one", "chunk/one"]},
        {"cited_chunk_ids": "chunk/one"},
        {"cited_chunk_ids": [False]},
        {"created_at": "2026-10-27T01:00:00Z"},
    ],
)
def test_tampered_save_fails_without_persistence(
    api: ExplorerAPI, updates: dict[str, object]
) -> None:
    fixture_run(api)
    page = preview(api, opened(api))
    fields = controls(page, "review-save")
    payload = {**json.loads(fields["submission"]), **updates}
    before = api.path.read_bytes()
    failure(
        api.client.post(SAVE, data={**fields, "submission": json.dumps(payload)}, headers=ORIGIN),
        422,
    )
    assert api.path.read_bytes() == before


@pytest.mark.parametrize("changed", ["comment", "decision", "cited_chunk_ids"])
def test_reused_uuid_with_different_valid_content_conflicts(api: ExplorerAPI, changed: str) -> None:
    fixture_run(api)
    page = preview(api, opened(api))
    response = save(api, page)
    assert response.status_code == 303
    fields = controls(page, "review-save")
    payload = json.loads(fields["submission"])
    payload[changed] = {
        "comment": "Changed opinion",
        "decision": "accepted",
        "cited_chunk_ids": [],
    }[changed]
    before = api.path.read_bytes()
    failure(
        api.client.post(SAVE, data={**fields, "submission": json.dumps(payload)}, headers=ORIGIN),
        409,
    )
    assert api.path.read_bytes() == before
    assert values(api.page(response.headers["location"]), "review-comment") == [COMMENT]


@pytest.mark.parametrize(
    "suffix",
    [
        "&comment=PRIVATE_duplicate",
        "&comment=%ff",
        "&x=%",
        "&x=%Q0",
        "&x",
        "&=x",
        "&",
        "&x=%ED%A0%80",
    ],
)
def test_strict_wire_parsing_fails_closed(api: ExplorerAPI, suffix: str) -> None:
    fixture_run(api)
    root = opened(api)
    before = api.path.read_bytes()
    failure(
        api.client.post(COMPOSER, content=urlencode(draft(root)) + suffix, headers=FORM_HEADERS),
        422,
    )
    assert api.path.read_bytes() == before


@pytest.mark.parametrize(
    "submission", ["{}", "[]", "[" * 1100, '{"comment":"one","comment":"two"}', "NaN"]
)
def test_save_json_rejects_ambiguous_or_malformed_content(
    api: ExplorerAPI, submission: str
) -> None:
    fixture_run(api)
    root = opened(api)
    before = api.path.read_bytes()
    failure(
        api.client.post(
            SAVE,
            data={"selection": controls(root)["selection"], "submission": submission},
            headers=ORIGIN,
        ),
        422,
    )
    assert api.path.read_bytes() == before


def test_encoding_queries_methods_and_streamed_body_bounds(api: ExplorerAPI) -> None:
    fixture_run(api)
    root = opened(api)
    raw = urlencode(draft(root)).encode()
    before = api.path.read_bytes()
    for path in (COMPOSER, SAVE):
        for media in (
            "application/json",
            "text/plain",
            "multipart/form-data; boundary=fixture",
            "application/x-www-form-urlencoded; charset=latin-1",
            "application/x-www-form-urlencoded; extra=ignored",
        ):
            failure(
                api.client.post(path, content=raw, headers={**ORIGIN, "content-type": media}), 415
            )
        failure(
            api.client.post(
                path, content=raw, headers={**FORM_HEADERS, "content-encoding": "gzip"}
            ),
            415,
        )
        for length in ("not-numeric", "9" * 20, "0", str(len(raw) + 1)):
            failure(
                api.client.post(
                    path, content=raw, headers={**FORM_HEADERS, "content-length": length}
                ),
                422,
            )
        failure(
            api.client.post(
                path + "?comment=PRIVATE_not_in_url", content=raw, headers=FORM_HEADERS
            ),
            422,
        )
        failure(api.client.post(path, content=b"x" * 524_289, headers=FORM_HEADERS), 413)
        failure(
            api.client.post(path, content=iter([b"x" * 262_144] * 3), headers=FORM_HEADERS), 413
        )
        for method in ("PUT", "PATCH", "DELETE", "OPTIONS"):
            response = api.client.request(method, path)
            failure(response, 405)
            assert response.headers["allow"] == ("POST" if path == SAVE else "GET, POST")
    query = "?" + urlencode({"selection": controls(root)["selection"]})
    for suffix in ("&selection=duplicate", "&x=%Q0", "&x=%ff", "&x", "&comment=PRIVATE_", "&"):
        failure(api.client.get(COMPOSER + query + suffix), 422)
    failure(api.client.request("GET", COMPOSER + query, content=b"comment=PRIVATE_"), 422)
    failure(api.client.get(SAVE), 405)
    failure(api.client.post(SAVED, data={}, headers=ORIGIN), 405)
    assert api.path.read_bytes() == before


def test_maximum_code_point_comment_keeps_exact_payload_and_no_browser_maxlength(
    api: ExplorerAPI,
) -> None:
    fixture_run(api)
    root = opened(api)
    comment = "\U0001f52c" * 4000
    page = preview(api, root, comment=comment)
    assert "maxlength" not in root.by_id("comment").attrs
    assert values(page, "review-comment") == [comment]
    response = save(api, page)
    assert response.status_code == 303
    assert values(api.page(response.headers["location"]), "review-comment") == [comment]


def test_non_reviewable_chunk_identity_is_shown_not_normalized_or_invented(
    api: ExplorerAPI,
) -> None:
    fixture_run(api, chunk_id="legacy\r\n\0chunk")
    root = opened(api)
    assert "disabled" in root.by_id("cite-1").attrs
    assert "cannot be cited under the existing review ID contract" in root.text()
    assert values(root, "passage-text") == [TEXT]
    response = api.client.post(COMPOSER, data=draft(root, cite=()), headers=ORIGIN)
    assert response.status_code == 200
    assert save(api, ExplorerDOM(response.text).root).status_code == 303


def test_output_and_return_bounds_are_checked_before_the_only_write(
    api: ExplorerAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture_run(api)
    root = opened(api)
    response = api.client.post(COMPOSER, data=draft(root), headers=ORIGIN)
    assert response.status_code == 200
    page = ExplorerDOM(response.text).root
    before = api.path.read_bytes()
    with monkeypatch.context() as patch:
        patch.setattr(explorer, "MAX_HTML_BYTES", len(response.content))
        assert (
            api.client.post(COMPOSER, data=draft(root), headers=ORIGIN).content == response.content
        )
        patch.setattr(explorer, "MAX_HTML_BYTES", len(response.content) - 1)
        failure(api.client.post(COMPOSER, data=draft(root), headers=ORIGIN), 413)
        failure(save(api, page), 413)
    module = importlib.import_module("api.review_explorer")
    with monkeypatch.context() as patch:
        patch.setattr(module, "MAX_RETURN_QUERY_BYTES", 1)
        failure(save(api, page), 413)
    assert api.path.read_bytes() == before


def test_unrepresentable_frozen_text_cannot_be_saved_through_the_browser(api: ExplorerAPI) -> None:
    fixture_run(api, text="valid prefix\0unrepresentable source")
    selection = json.dumps({"run_id": "saved"})
    before = api.path.read_bytes()
    failure(api.client.get(COMPOSER, params={"selection": selection}), 409)
    payload = ReviewSubmission(
        review_id=uuid4(), decision=ReviewDecision.REJECTED, comment="An explicit human opinion."
    )
    failure(
        api.client.post(
            SAVE,
            data={"selection": selection, "submission": payload.model_dump_json()},
            headers=ORIGIN,
        ),
        409,
    )
    assert api.path.read_bytes() == before


def test_restart_and_corpus_deletion_keep_opinion_evidence_and_events_unchanged(
    api: ExplorerAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = fixture_run(api)
    root = opened(api)
    previewed = preview(api, root)
    response = save(api, previewed)
    assert response.status_code == 303
    location = response.headers["location"]
    original_page = api.client.get(location).content
    original_events = api.container.event_log.list_events()
    with closing(sqlite3.connect(api.path)) as connection, connection:
        connection.execute("DELETE FROM chunks")
        connection.execute("DELETE FROM documents")
    app = create_app(offline_settings(api.path))
    container = app.state.container
    for target, method in (
        (container.document_catalog, "list_documents"),
        (container.document_chunks, "list_chunks"),
        (container.source_context, "read"),
        (container.event_log, "list_events"),
        (container.event_log, "append_transition"),
        (container.evidence_exporter, "export"),
        (container.answer_reviews, "create"),
        (container.evidence_annotations, "create"),
        (container.paper_screening, "submit"),
    ):
        monkeypatch.setattr(target, method, no_work)
    before = api.path.read_bytes()
    with TestClient(app) as client, browsing_guards(container) as guards:
        assert client.get(location).content == original_page
        retry = client.post(
            SAVE, data=controls(previewed, "review-save"), headers=ORIGIN, follow_redirects=False
        )
        assert retry.status_code == 303 and retry.headers["location"] == location
        assert client.get(location).content == original_page
        assert container.saved_run_evidence.export(bundle.run_id) == bundle
        assert not any(guard.call_count for guard in guards.values())
    assert api.path.read_bytes() == before
    assert api.container.event_log.list_events() == original_events


def test_exact_review_lookup_is_run_scoped_not_a_newest_page_guess(api: ExplorerAPI) -> None:
    bundle = fixture_run(api)
    other = fixture_run(api, run_id="other")
    payload = ReviewSubmission(
        review_id=uuid4(),
        decision=ReviewDecision.NEEDS_REVISION,
        comment=COMMENT,
        cited_chunk_ids=(bundle.snapshot.sources[0].chunk.chunk_id,),
    )
    original, _ = api.container.notebook_reviews.create(bundle.run_id, payload)
    for number in range(25):
        opinion(api, bundle, f"Later opinion {number}")
    before = api.path.read_bytes()
    assert api.container.notebook_reviews.get_review(bundle.run_id, payload.review_id) == original
    with pytest.raises(AnswerReviewError) as error:
        api.container.notebook_reviews.get_review(other.run_id, payload.review_id)
    assert error.value.code == "review_not_found" and error.value.status_code == 404
    assert api.path.read_bytes() == before


@pytest.mark.parametrize("operation", ["export", "create"])
def test_storage_failures_are_sanitized_and_insert_errors_roll_back(
    api: ExplorerAPI, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    fixture_run(api)
    page = preview(api, opened(api))
    target = (
        api.container.saved_run_evidence
        if operation == "export"
        else api.container.notebook_reviews
    )
    error = (
        EvidenceExportError("evidence_storage_unavailable", "PRIVATE_database", 503)
        if operation == "export"
        else AnswerReviewError("review_storage_unavailable", "PRIVATE_database", 503)
    )
    with monkeypatch.context() as patch:
        patch.setattr(target, operation, Mock(side_effect=error))
        before = api.path.read_bytes()
        failure(save(api, page), 503)
        assert api.path.read_bytes() == before
    with closing(sqlite3.connect(api.path)) as connection, connection:
        connection.execute(
            """CREATE TRIGGER fail_review BEFORE INSERT ON answer_reviews
               BEGIN SELECT RAISE(ABORT, 'PRIVATE_insert_failure'); END"""
        )
    before = api.path.read_bytes()
    failure(save(api, page), 503)
    assert api.path.read_bytes() == before


def test_json_api_and_notebook_pagination_keep_their_contracts(api: ExplorerAPI) -> None:
    bundle = fixture_run(api)
    page = preview(api, opened(api))
    payload = json.loads(controls(page, "review-save")["submission"])
    response = api.client.post(f"/runs/{bundle.run_id}/reviews", json=payload)
    assert response.status_code == 201
    retry = api.client.post(f"/runs/{bundle.run_id}/reviews", json=payload)
    assert retry.status_code == 200 and retry.json() == response.json()
    native = save(api, page)
    assert native.status_code == 303
    history = api.client.get(f"/runs/{bundle.run_id}/reviews?limit=1").json()
    assert history == {"reviews": [response.json()], "next_cursor": None}
    assert values(api.page(native.headers["location"]), "review-comment") == [COMMENT]
    assert response.headers["content-type"] == "application/json"


def test_native_openapi_has_unique_operations_and_no_new_tables(api: ExplorerAPI) -> None:
    paths = api.app.openapi()["paths"]
    assert set(paths[COMPOSER]) == {"get", "post"}
    assert set(paths[SAVE]) == {"post"} and set(paths[SAVED]) == {"get"}
    operations = [
        operation["operationId"] for path in paths.values() for operation in path.values()
    ]
    assert len(operations) == len(set(operations))
    with closing(sqlite3.connect(api.path)) as connection:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert not any("draft" in name or "composer" in name or "preview" in name for name in tables)


@pytest.mark.parametrize(
    "origin", ["http://127.0.0.1:8765", "https://example.test:8443", "http://[::1]:8765"]
)
async def test_review_origin_supports_ports_tls_and_ipv6(api: ExplorerAPI, origin: str) -> None:
    fixture_run(api)
    root = opened(api)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api.app), base_url=origin
    ) as client:
        response = await client.post(COMPOSER, data=draft(root), headers={"origin": origin})
        assert response.status_code == 200
        page = ExplorerDOM(response.text).root
        fields = controls(page, "review-save")
        before = api.path.read_bytes()
        failure(await client.post(SAVE, data=fields, headers={"origin": origin + "/"}), 403)
        assert api.path.read_bytes() == before
        saved = await client.post(SAVE, data=fields, headers={"origin": origin})
        assert saved.status_code == 303
        persisted = await client.get(saved.headers["location"])
        assert values(ExplorerDOM(persisted.text).root, "review-comment") == [COMMENT]


@pytest.mark.parametrize("size,status", [(65_536, 422), (65_537, 413)])
async def test_exact_native_query_bound(api: ExplorerAPI, size: int, status: int) -> None:
    scope: Scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "path": COMPOSER,
        "raw_path": COMPOSER.encode(),
        "root_path": "",
        "scheme": "http",
        "query_string": b"x" * size,
        "headers": [(b"host", b"testserver")],
        "server": ("testserver", 80),
        "client": ("testclient", 123),
    }
    messages: list[Message] = []

    async def receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: Message) -> None:
        messages.append(message)

    before = api.path.read_bytes()
    await api.app(scope, receive, send)
    failure(
        httpx.Response(
            messages[0]["status"],
            headers=messages[0]["headers"],
            content=b"".join(item.get("body", b"") for item in messages[1:]),
        ),
        status,
    )
    assert api.path.read_bytes() == before


def test_exact_streamed_save_body_bound_and_duplicate_headers(api: ExplorerAPI) -> None:
    fixture_run(api)
    page = preview(api, opened(api))
    fields = controls(page, "review-save")
    raw = urlencode(fields).encode()
    for duplicate in ("content-length", "content-type"):
        headers = [*FORM_HEADERS.items(), ("content-length", str(len(raw)))]
        headers.append(
            (duplicate, str(len(raw)) if duplicate == "content-length" else FORM_HEADERS[duplicate])
        )
        failure(
            api.client.post(SAVE, content=raw, headers=headers),
            422 if duplicate == "content-length" else 415,
        )
    padding, remainder = divmod(524_288 - len(raw), 3)
    exact = raw + b"%20" * padding + b"+" * remainder
    assert len(exact) == 524_288
    before = api.path.read_bytes()
    failure(api.client.post(SAVE, content=iter([exact, b"+"]), headers=FORM_HEADERS), 413)
    assert api.path.read_bytes() == before
    response = api.client.post(
        SAVE,
        content=iter([exact[:300_000], exact[300_000:]]),
        headers=FORM_HEADERS,
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert values(api.page(response.headers["location"]), "review-comment") == [COMMENT]


def test_preflight_covers_the_larger_persisted_opinion_not_only_preview(
    api: ExplorerAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture_run(api, document_id="<" * 75_000)
    root = api.page(
        COMPOSER, selection=json.dumps({"run_id": "saved"}), referrer_policy="same-origin"
    )
    page = preview(api, root)
    create = Mock(wraps=api.container.notebook_reviews.create)
    monkeypatch.setattr(api.container.notebook_reviews, "create", create)
    before = api.path.read_bytes()
    failure(save(api, page), 413)
    create.assert_not_called()
    assert api.path.read_bytes() == before


@pytest.mark.parametrize("extra_byte", [0, 1])
def test_persisted_page_preflight_at_exact_escaped_byte_limit(
    api: ExplorerAPI, extra_byte: int
) -> None:
    module = importlib.import_module("api.review_explorer")
    bundle = fixture_run(api, document_id="d")
    selection = module._Selection(run_id=bundle.run_id)
    payload = ReviewSubmission(
        review_id=uuid4(),
        decision=ReviewDecision.NEEDS_REVISION,
        comment=COMMENT,
        cited_chunk_ids=("chunk/one",),
    )
    widest = SavedAnswerReview.model_validate(
        {
            **payload.model_dump(),
            "run_id": bundle.run_id,
            "schema_version": "1.0",
            "sequence": MAX_REVIEW_SEQUENCE,
            "created_at": datetime.max.replace(tzinfo=UTC).isoformat().replace("+00:00", "Z"),
        }
    )
    base = module._saved_page(widest, bundle, selection, "")
    bundle.snapshot.sources[0].chunk.document_id += "<"
    bundle.answer.citations[0].document_id += "<"
    unit = len(module._saved_page(widest, bundle, selection, "").body) - len(base.body)
    assert unit == 16
    count, remainder = divmod(explorer.MAX_HTML_BYTES - len(base.body), unit)
    document_id = "d" + "<" * count
    # Each document-ID character appears four times; use the comment for a one-byte remainder.
    adjusted = payload.model_copy(update={"comment": COMMENT + "x" * (remainder + extra_byte)})
    with closing(sqlite3.connect(api.path)) as connection, connection:
        connection.execute("DELETE FROM agent_events")
        connection.execute(
            "INSERT INTO sqlite_sequence (name, seq) VALUES ('answer_reviews', ?)",
            (MAX_REVIEW_SEQUENCE - 1,),
        )
    fixture_run(api, document_id=document_id)
    response = api.client.post(
        SAVE,
        data={"selection": selection.token(), "submission": adjusted.model_dump_json()},
        headers=ORIGIN,
        follow_redirects=False,
    )
    if extra_byte:
        failure(response, 413)
        assert api.container.notebook_reviews.list_reviews(bundle.run_id).reviews == []
    else:
        assert response.status_code == 303
        persisted = api.client.get(response.headers["location"])
        assert persisted.status_code == 200 and len(persisted.content) <= explorer.MAX_HTML_BYTES
        root = ExplorerDOM(persisted.text).root
        assert values(root, "review-comment") == [adjusted.comment]
        assert values(root, "review-sequence") == [str(MAX_REVIEW_SEQUENCE)]


@pytest.mark.parametrize(
    "column,value,status",
    [
        ("comment", " ", 409),
        ("comment", b"PRIVATE_blob", 409),
        ("comment", b"\xff", 409),
        ("comment", "x" * 262_145, 413),
        ("created_at", "PRIVATE_invalid", 409),
        ("cited_chunk_ids", '["outside-frozen-bundle"]', 409),
        ("cited_chunk_ids", '{"PRIVATE_invalid":1}', 409),
    ],
)
def test_precise_reader_rejects_corrupt_or_oversized_saved_rows(
    api: ExplorerAPI, column: str, value: str | bytes, status: int
) -> None:
    fixture_run(api)
    response = save(api, preview(api, opened(api)))
    assert response.status_code == 303
    with closing(sqlite3.connect(api.path)) as connection, connection:
        connection.execute("PRAGMA ignore_check_constraints = ON")
        connection.execute(f"UPDATE answer_reviews SET {column} = ?", (value,))  # noqa: S608
    before = api.path.read_bytes()
    failure(api.client.get(response.headers["location"]), status)
    assert api.path.read_bytes() == before
