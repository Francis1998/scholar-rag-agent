"""Native annotation authoring uses exact frozen evidence and the existing append-only store."""

import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from unittest.mock import Mock
from urllib.parse import parse_qs, urlencode, urlsplit
from uuid import UUID

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from scripts.demo_corpus_explorer import browsing_guards
from scripts.demo_evidence_export import offline_settings
from starlette.types import Message, Scope

from agent.evidence import EvidenceBundle, EvidenceSource, text_digest
from api import annotation_explorer, explorer
from api.application import create_app
from api.explorer import _page
from api.review_notebook import _AnnotationParameters
from api.review_notebook import _source as annotation_fragments
from storage.evidence_annotations import (
    AnnotationError,
    AnnotationSubmission,
    SavedEvidenceAnnotation,
)
from storage.evidence_export import EvidenceExportError
from storage.run_history import MAX_EVENT_ID
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
from tests.test_review_notebook import NOTE, TEXT, annotate, fixture_run, opinion
from tests.test_run_comparison import ComparisonAPI, _save, _source
from tests.test_run_explorer import reader_headers

COMPOSER = "/explore/run/annotate"
SAVE = COMPOSER + "/save"
ORIGIN = {"origin": "http://testserver", "sec-fetch-site": "same-origin"}


def controls(root: Node, identifier: str = "annotation-draft") -> dict[str, str]:
    return {
        node.attrs["name"]: node.attrs.get("value", "")
        for node in root.by_id(identifier).by_tag("input")
    }


def open_draft(api: ExplorerAPI, bundle: EvidenceBundle, **params: str | int) -> Node:
    notebook = api.page("/explore/run/notebook", run_id=bundle.run_id, **params)
    return api.page(href(notebook, "annotate-source-1"), referrer_policy="same-origin")


def preview(
    api: ExplorerAPI, opened: Node, *, quote: str = "Repeated quote", note: str = NOTE
) -> httpx.Response:
    text = values(opened, "passage-text")[0]
    start = text.rindex(quote)
    return api.client.post(
        opened.by_id("annotation-draft").attrs["action"],
        data={
            **controls(opened),
            "start": str(start),
            "end": str(start + len(quote)),
            "note": note,
        },
        headers=ORIGIN,
    )


def failure(response: httpx.Response, status: int) -> None:
    assert response.status_code == status, response.text[:1000]
    assert_headers(response, referrer_policy="same-origin")
    root = ExplorerDOM(response.text).root
    assert not root.by_tag("form") and not root.by_class("annotation-note")
    assert "PRIVATE_" not in response.text
    assert "location" not in response.headers and "set-cookie" not in response.headers


def test_reader_and_empty_notebook_open_the_exact_frozen_composer(api: ExplorerAPI) -> None:
    bundle = fixture_run(api)
    source = bundle.snapshot.sources[0]
    notebook = api.page("/explore/run/notebook", run_id=bundle.run_id)
    reader = reader_headers(api.client.get(href(notebook, "back-to-evidence")))
    before = api.path.read_bytes()
    for parent in (reader, notebook):
        opened = api.page(href(parent, "annotate-source-1"), referrer_policy="same-origin")
        draft = opened.by_id("annotation-draft")
        assert draft.attrs["method"] == "post" and draft.attrs["action"] == COMPOSER
        assert values(opened, "passage-text") == [source.chunk.text]
        assert identities(opened, "run-id") == [bundle.run_id]
        assert identities(opened) == [source.chunk.document_id]
        assert identities(opened, "chunk-id") == [source.chunk.chunk_id]
        fields = controls(opened)
        assert fields["start"] == fields["end"] == ""
        assert opened.by_id("note").text() == ""
        assert UUID(fields["annotation_id"])
        selection = json.loads(fields["selection"])
        assert selection["run_id"] == bundle.run_id
        assert selection["document_id"] == source.chunk.document_id
        assert selection["chunk_id"] == source.chunk.chunk_id
        assert selection["source_text_sha256"] == source.text_sha256
        assert "Unicode code points" in opened.text() and "Preview exact annotation" in draft.text()
    assert api.path.read_bytes() == before
    assert api.container.evidence_annotations.list_annotations(bundle.run_id).annotations == []


def test_preview_then_save_reuses_store_and_uuid_with_original_timestamp(
    api: ExplorerAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = fixture_run(api)
    opened = open_draft(api, bundle)
    create = Mock(wraps=api.container.evidence_annotations.create)
    monkeypatch.setattr(api.container.evidence_annotations, "create", create)
    before = api.path.read_bytes()
    with browsing_guards(api.container) as guards:
        response = preview(api, opened)
        assert response.status_code == 200
        assert_headers(response, referrer_policy="same-origin")
        page = ExplorerDOM(response.text).root
        assert values(page, "passage-text") == [TEXT]
        assert page.by_id("annotation-preview-quote").text() == "Repeated quote"
        assert len(page.by_tag("mark")) == 1
        assert values(page, "annotation-note") == [NOTE]
        assert "Nothing has been saved" in page.text()
        assert api.path.read_bytes() == before and create.call_count == 0
        fields = controls(page, "annotation-save")
        submission = AnnotationSubmission.model_validate_json(fields["submission"])
        assert submission.annotation_id == UUID(controls(opened)["annotation_id"])
        assert submission.start == TEXT.rindex("Repeated quote")
        assert submission.quote == "Repeated quote" and submission.note == NOTE
        assert controls(page)["annotation_id"] == str(submission.annotation_id)
        assert page.by_id("note").text() == NOTE
        action = page.by_id("annotation-save").attrs["action"]
        assert action == SAVE
        saved = api.client.post(action, data=fields, headers=ORIGIN, follow_redirects=False)
        assert saved.status_code == 303
        assert saved.headers["cache-control"] == "no-store"
        assert saved.headers["referrer-policy"] == "same-origin"
        location = saved.headers["location"]
        parts = urlsplit(location)
        assert parts.path == "/explore/run/annotation" and parts.fragment == "annotation-quote"
        assert not parts.netloc and not parts.scheme
        assert not {"note", "quote", "submission"} & parse_qs(parts.query).keys()
        assert NOTE not in location
        create.assert_called_once_with(bundle.run_id, submission)
        persisted = api.page(location)
        assert values(persisted, "annotation-note") == [NOTE]
        assert persisted.by_id("annotation-quote").text() == submission.quote
        original, _ = api.container.evidence_annotations.get_annotation(
            bundle.run_id, submission.annotation_id
        )
        after = api.path.read_bytes()
        retry = api.client.post(action, data=fields, headers=ORIGIN, follow_redirects=False)
        assert retry.status_code == 303 and retry.headers["location"] == location
        assert api.path.read_bytes() == after
        repeated, _ = api.container.evidence_annotations.get_annotation(
            bundle.run_id, submission.annotation_id
        )
        assert repeated == original
        assert (
            len(api.container.evidence_annotations.list_annotations(bundle.run_id).annotations) == 1
        )
        assert not any(guard.call_count for guard in guards.values())


@pytest.mark.parametrize("quote", ["Repeated quote", "\U0001f52c", "\u0301", "\r\n"])
def test_explicit_code_point_offsets_preview_only_the_chosen_occurrence(
    api: ExplorerAPI, quote: str
) -> None:
    bundle = fixture_run(api)
    opened = open_draft(api, bundle)
    before = api.path.read_bytes()
    response = preview(api, opened, quote=quote)
    assert response.status_code == 200
    page = ExplorerDOM(response.text).root
    marked = page.by_id("annotation-preview-quote")
    assert marked.text() == quote and len(page.by_tag("mark")) == 1
    code = page.by_class("passage-text")[0].by_tag("code")[0]
    mark_index = code.children.index(marked)
    assert (
        "".join(child for child in code.children[:mark_index] if isinstance(child, str))
        == (TEXT[: TEXT.rindex(quote)])
    )
    fields = controls(page, "annotation-save")
    payload = json.loads(fields["submission"])
    assert payload["start"] == TEXT.rindex(quote) and payload["quote"] == quote
    assert api.path.read_bytes() == before
    saved = api.client.post(SAVE, data=fields, headers=ORIGIN, follow_redirects=False)
    assert saved.status_code == 303
    assert api.page(saved.headers["location"]).by_id("annotation-quote").text() == quote


def test_changed_payload_with_reused_uuid_conflicts_without_overwriting(api: ExplorerAPI) -> None:
    bundle = fixture_run(api)
    opened = open_draft(api, bundle)
    page = ExplorerDOM(preview(api, opened).text).root
    fields = controls(page, "annotation-save")
    response = api.client.post(SAVE, data=fields, headers=ORIGIN, follow_redirects=False)
    assert response.status_code == 303
    before = api.path.read_bytes()
    payload = json.loads(fields["submission"])
    payload["note"] = "Changed human opinion must not replace the original."
    failure(
        api.client.post(
            SAVE,
            data={**fields, "submission": json.dumps(payload)},
            headers=ORIGIN,
            follow_redirects=False,
        ),
        409,
    )
    assert api.path.read_bytes() == before
    assert values(api.page(response.headers["location"]), "annotation-note") == [NOTE]


@pytest.mark.parametrize(
    "identifier",
    [".", "..", "/", "//external.invalid/path", "'\"?%#&+\\", "\u7814/e\u0301"],
)
def test_opaque_ids_and_independent_return_pages_survive_authoring(
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
    opened = api.page(href(older, "annotate-source-1"), referrer_policy="same-origin")
    assert identities(opened, "run-id") == identities(opened) == [identifier]
    assert identities(opened, "chunk-id") == [identifier]
    assert href(opened, "back-to-notebook") == notebook_url
    assert href(opened, "back-to-runs") == href(older, "back-to-runs")
    assert href(opened, "back-to-catalog") == href(older, "back-to-catalog")
    selection = controls(opened)["selection"]
    assert selection.isascii() and "\0" not in selection and "\r" not in selection
    page = ExplorerDOM(preview(api, opened).text).root
    saved = api.client.post(
        SAVE, data=controls(page, "annotation-save"), headers=ORIGIN, follow_redirects=False
    )
    assert saved.status_code == 303
    persisted = api.page(saved.headers["location"])
    assert href(persisted, "back-to-notebook") == notebook_url
    assert values(api.page(notebook_url), "annotation-note") == ["Note 1"]
    assert values(persisted, "annotation-note") == [NOTE]
    separate = api.page(href(persisted, "annotate-source"), referrer_policy="same-origin")
    assert controls(separate)["annotation_id"] != controls(opened)["annotation_id"]
    assert href(separate, "back-to-notebook") == notebook_url


def test_nul_identities_survive_json_inputs_and_redirects_without_becoming_html(
    api: ExplorerAPI,
) -> None:
    identifier = "opaque/../\r\nid\0?x#fragment"
    bundle = fixture_run(api, run_id=identifier, chunk_id=identifier, document_id=identifier)
    opened = open_draft(api, bundle, review_cursor=42, annotation_cursor=42)
    page = ExplorerDOM(preview(api, opened).text).root
    fields = controls(page, "annotation-save")
    assert fields["selection"].isascii() and fields["submission"].isascii()
    assert json.loads(fields["submission"])["chunk_id"] == identifier
    response = api.client.post(SAVE, data=fields, headers=ORIGIN, follow_redirects=False)
    assert response.status_code == 303
    saved = api.page(response.headers["location"])
    assert identities(saved, "run-id") == identities(saved) == [identifier]
    assert href(saved, "back-to-notebook") == href(opened, "back-to-notebook")


@pytest.mark.parametrize("prefix,mounted", [("/library", False), ("/prefix/library", True)])
def test_mounts_cover_composer_forms_preview_save_redirect_and_errors(
    api: ExplorerAPI, prefix: str, mounted: bool
) -> None:
    bundle = fixture_run(api, run_id="run/../one", document_id="paper/one\r\n\0")
    application = api.app
    if mounted:
        application = FastAPI()
        application.mount("/library", api.app)
    with TestClient(application, root_path="/prefix" if mounted else prefix) as client:
        root = ExplorerDOM(
            client.get(
                prefix + "/explore/run/notebook",
                params={"run_id": bundle.run_id, "question": " Exact %_ "},
            ).text
        ).root
        opened = ExplorerDOM(client.get(href(root, "annotate-source-1")).text).root
        draft = controls(opened)
        start = TEXT.rindex("Repeated quote")
        response = client.post(
            opened.by_id("annotation-draft").attrs["action"],
            data={**draft, "start": str(start), "end": str(start + 14), "note": NOTE},
            headers=ORIGIN,
        )
        assert response.status_code == 200
        page = ExplorerDOM(response.text).root
        response = client.post(
            page.by_id("annotation-save").attrs["action"],
            data=controls(page, "annotation-save"),
            headers=ORIGIN,
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert response.headers["location"].startswith(prefix + "/explore/run/annotation?")
        persisted = client.get(response.headers["location"])
        assert persisted.status_code == 200
        error = client.post(prefix + SAVE, data={"selection": "{}"}, headers=ORIGIN)
        failure(error, 422)
        for root in (opened, page, ExplorerDOM(persisted.text).root, ExplorerDOM(error.text).root):
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
        {"referer": "http://testserver/explore/run/annotate"},
        {"origin": "https://external.invalid", "x-forwarded-host": "external.invalid"},
        [("origin", "http://testserver"), ("origin", "http://testserver")],
        [
            ("origin", "http://testserver"),
            ("sec-fetch-site", "same-origin"),
            ("sec-fetch-site", "same-origin"),
        ],
        [("origin", "http://testserver"), ("host", "testserver"), ("host", "testserver")],
    ],
)
def test_invalid_origins_fail_before_preview_or_save_reads(
    api: ExplorerAPI,
    monkeypatch: pytest.MonkeyPatch,
    headers: dict[str, str] | list[tuple[str, str]],
) -> None:
    bundle = fixture_run(api)
    opened = open_draft(api, bundle)
    page = ExplorerDOM(preview(api, opened).text).root
    monkeypatch.setattr(api.container.saved_run_evidence, "export", no_work)
    monkeypatch.setattr(api.container.evidence_annotations, "create", no_work)
    before = api.path.read_bytes()
    for path, fields in (
        (COMPOSER, {**controls(opened), "start": "0", "end": "1", "note": NOTE}),
        (SAVE, controls(page, "annotation-save")),
    ):
        failure(api.client.post(path, data=fields, headers=headers, follow_redirects=False), 403)
    assert api.path.read_bytes() == before


@pytest.mark.parametrize(
    "updates",
    [
        {"start": ""},
        {"start": "-1"},
        {"start": "+0"},
        {"start": "1.0"},
        {"start": "false"},
        {"start": "\u0661"},
        {"start": "1" * 20},
        {"start": "262145"},
        {"end": ""},
        {"end": "0"},
        {"end": "1.0"},
        {"end": "262145"},
        {"start": "2", "end": "1"},
        {"end": str(len(TEXT) + 1)},
        {"note": ""},
        {"note": " \r\n\t"},
        {"note": "PRIVATE_\0invalid"},
        {"note": "x" * 1001},
        {"annotation_id": ""},
        {"annotation_id": "PRIVATE_not_uuid"},
        {"selection": "{}"},
        {"selection": "[]"},
        {"selection": "[" * 1100},
        {"selection": '{"run_id":"saved","run_id":"different"}'},
        {"return_to": "//external.invalid/PRIVATE_"},
        {"created_at": "2026-10-26T01:00:00Z"},
    ],
)
def test_invalid_drafts_never_create_an_annotation(
    api: ExplorerAPI, updates: dict[str, str]
) -> None:
    bundle = fixture_run(api)
    opened = open_draft(api, bundle)
    before = api.path.read_bytes()
    failure(
        api.client.post(
            COMPOSER,
            data={**controls(opened), "start": "0", "end": "1", "note": NOTE, **updates},
            headers=ORIGIN,
        ),
        422,
    )
    assert api.path.read_bytes() == before


@pytest.mark.parametrize(
    "updates,status",
    [
        ({"run_id": "missing"}, 404),
        ({"run_id": ""}, 422),
        ({"run_id": "\ud800"}, 422),
        ({"chunk_id": "missing"}, 404),
        ({"document_id": "wrong"}, 422),
        ({"source_text_sha256": "0" * 64}, 422),
        ({"source_text_sha256": "bad"}, 422),
        ({"chunk_id": "x" * 257}, 422),
        ({"review_limit": 0}, 422),
        ({"annotation_limit": 101}, 422),
        ({"review_cursor": 0}, 422),
        ({"annotation_cursor": ""}, 422),
        ({"catalog_cursor": "\ud800"}, 422),
        ({"source": "\ud800"}, 422),
        ({"return_to": "https://external.invalid/PRIVATE_"}, 422),
        ({"note": "PRIVATE_no_notes_in_get"}, 422),
    ],
)
def test_selection_is_exact_typed_and_does_not_ignore_fields(
    api: ExplorerAPI, updates: dict[str, str | int], status: int
) -> None:
    bundle = fixture_run(api)
    opened = open_draft(api, bundle)
    selection = {**json.loads(controls(opened)["selection"]), **updates}
    before = api.path.read_bytes()
    failure(api.client.get(COMPOSER, params={"selection": json.dumps(selection)}), status)
    assert api.path.read_bytes() == before


@pytest.mark.parametrize(
    "updates",
    [
        {"document_id": "different"},
        {"chunk_id": "missing"},
        {"source_text_sha256": "0" * 64},
        {"start": 0},
        {"end": len(TEXT) + 1},
        {"quote": "Repeated QUOTE"},
        {"quote": ""},
        {"note": "PRIVATE_\0invalid"},
        {"note": "\ud800"},
        {"note": "x" * 1001},
        {"created_at": "2026-10-26T01:00:00Z"},
        {"start": True},
        {"end": 1.0},
    ],
)
def test_tampered_save_contract_fails_without_writing(
    api: ExplorerAPI, updates: dict[str, str | int | float]
) -> None:
    bundle = fixture_run(api)
    page = ExplorerDOM(preview(api, open_draft(api, bundle)).text).root
    fields = controls(page, "annotation-save")
    payload = {**json.loads(fields["submission"]), **updates}
    before = api.path.read_bytes()
    failure(
        api.client.post(SAVE, data={**fields, "submission": json.dumps(payload)}, headers=ORIGIN),
        404 if "chunk_id" in updates else 422,
    )
    assert api.path.read_bytes() == before


@pytest.mark.parametrize(
    "suffix",
    ["&note=PRIVATE_duplicate", "&note=%ff", "&x=%", "&x=%Q0", "&x", "&=x", "&", "&x=%ED%A0%80"],
)
def test_strict_wire_parsing_rejects_malformed_or_duplicate_fields(
    api: ExplorerAPI, suffix: str
) -> None:
    bundle = fixture_run(api)
    opened = open_draft(api, bundle)
    fields = {**controls(opened), "start": "0", "end": "1", "note": NOTE}
    before = api.path.read_bytes()
    failure(
        api.client.post(
            COMPOSER,
            content=urlencode(fields) + suffix,
            headers={**ORIGIN, "content-type": "application/x-www-form-urlencoded"},
        ),
        422,
    )
    assert api.path.read_bytes() == before


@pytest.mark.parametrize(
    "submission", ["{}", "[]", "[" * 1100, '{"note":"one","note":"two"}', "NaN"]
)
def test_save_json_is_strict_and_bounded(api: ExplorerAPI, submission: str) -> None:
    bundle = fixture_run(api)
    opened = open_draft(api, bundle)
    before = api.path.read_bytes()
    failure(
        api.client.post(
            SAVE,
            data={"selection": controls(opened)["selection"], "submission": submission},
            headers=ORIGIN,
        ),
        422,
    )
    assert api.path.read_bytes() == before


def test_encoding_queries_methods_and_streamed_body_bounds(
    api: ExplorerAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = fixture_run(api)
    opened = open_draft(api, bundle)
    fields = {**controls(opened), "start": "0", "end": "1", "note": NOTE}
    raw = urlencode(fields).encode()
    headers = {**ORIGIN, "content-type": "application/x-www-form-urlencoded"}
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
            api.client.post(path, content=raw, headers={**headers, "content-encoding": "gzip"}), 415
        )
        failure(
            api.client.post(
                path,
                content=raw,
                headers=[*headers.items(), ("content-type", headers["content-type"])],
            ),
            415,
        )
        for length in ("not-numeric", "9" * 20, "0", str(len(raw) + 1)):
            failure(
                api.client.post(path, content=raw, headers={**headers, "content-length": length}),
                422,
            )
        failure(
            api.client.post(
                path,
                content=raw,
                headers=[
                    *headers.items(),
                    ("content-length", str(len(raw))),
                    ("content-length", str(len(raw))),
                ],
            ),
            422,
        )
        failure(
            api.client.post(path + "?note=PRIVATE_not_in_url", content=raw, headers=headers), 422
        )
        failure(api.client.post(path, content=b"x" * 65_537, headers=headers), 413)
        failure(api.client.post(path, content=iter([b"x" * 32_768] * 3), headers=headers), 413)
        for method in ("PUT", "PATCH", "DELETE", "OPTIONS"):
            response = api.client.request(method, path)
            failure(response, 405)
            assert response.headers["allow"] == ("POST" if path == SAVE else "GET, POST")
        response = api.client.head(path)
        assert response.status_code == 405 and not response.content
        assert response.headers["cache-control"] == "no-store"
    query = "?" + urlencode({"selection": controls(opened)["selection"]})
    for suffix in ("&selection=duplicate", "&x=%Q0", "&x=%ff", "&x", "&note=PRIVATE_", "&"):
        failure(api.client.get(COMPOSER + query + suffix), 422)
    failure(api.client.request("GET", COMPOSER + query, content=b"note=PRIVATE_"), 422)
    failure(api.client.get(SAVE), 405)
    assert api.path.read_bytes() == before
    monkeypatch.setattr(annotation_explorer, "MAX_FORM_BYTES", len(raw) - 1)
    failure(api.client.post(COMPOSER, content=raw, headers=headers), 413)
    assert api.path.read_bytes() == before


@pytest.mark.parametrize("size,status", [(65_536, 422), (65_537, 413)])
async def test_exact_query_byte_boundary(api: ExplorerAPI, size: int, status: int) -> None:
    before = api.path.read_bytes()
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


def test_maximum_unicode_quote_and_note_save_without_utf16_or_byte_confusion(
    api: ExplorerAPI,
) -> None:
    text = "\U0001f52c" * 1000
    bundle = fixture_run(api, text=text, chunk_id="\U0001f52c" * 256, document_id="\u7814" * 256)
    opened = open_draft(api, bundle)
    response = preview(api, opened, quote=text, note=text)
    assert response.status_code == 200
    page = ExplorerDOM(response.text).root
    assert values(page, "quote-span") == ["[0, 1000)"]
    saved = api.client.post(
        SAVE, data=controls(page, "annotation-save"), headers=ORIGIN, follow_redirects=False
    )
    assert saved.status_code == 303
    assert values(api.page(saved.headers["location"]), "annotation-note") == [text]


def test_output_and_return_limits_fail_before_the_only_write(
    api: ExplorerAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = fixture_run(api)
    opened = open_draft(api, bundle)
    response = preview(api, opened)
    assert response.status_code == 200
    fields = controls(ExplorerDOM(response.text).root, "annotation-save")
    before = api.path.read_bytes()
    with monkeypatch.context() as patch:
        patch.setattr(explorer, "MAX_HTML_BYTES", len(response.content))
        assert preview(api, opened).content == response.content
        patch.setattr(explorer, "MAX_HTML_BYTES", len(response.content) - 1)
        failure(preview(api, opened), 413)
        failure(api.client.post(SAVE, data=fields, headers=ORIGIN), 413)
    with monkeypatch.context() as patch:
        patch.setattr(annotation_explorer, "MAX_RETURN_QUERY_BYTES", 1)
        failure(api.client.post(SAVE, data=fields, headers=ORIGIN), 413)
    assert api.path.read_bytes() == before


def test_unrepresentable_passages_cannot_be_saved_through_the_browser(api: ExplorerAPI) -> None:
    bundle = fixture_run(api, text="valid prefix\0unrepresentable source")
    notebook = api.page("/explore/run/notebook", run_id=bundle.run_id)
    url = href(notebook, "annotate-source-1")
    before = api.path.read_bytes()
    failure(api.client.get(url), 409)
    selection = parse_qs(urlsplit(url).query)["selection"][0]
    failure(
        api.client.post(
            COMPOSER,
            data={
                "selection": selection,
                "annotation_id": "00000000-0000-4000-8000-000000000001",
                "start": "0",
                "end": "1",
                "note": NOTE,
            },
            headers=ORIGIN,
        ),
        409,
    )
    assert api.path.read_bytes() == before


def test_corpus_change_deletion_and_restart_do_not_reanchor_or_repeat_work(
    api: ExplorerAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    api.seed("paper/one", ("chunk/one",), text=TEXT)
    bundle = fixture_run(api)
    opened = open_draft(api, bundle)
    previewed = preview(api, opened)
    page = ExplorerDOM(previewed.text).root
    fields = controls(page, "annotation-save")
    events = api.container.event_log.list_events()
    for stage in ("changed", "deleted"):
        with closing(sqlite3.connect(api.path)) as connection, connection:
            if stage == "changed":
                connection.execute("UPDATE chunks SET text = 'PRIVATE_CURRENT_CORPUS'")
            else:
                for table in (
                    "entity_edges",
                    "entity_mentions",
                    "graph_chunks",
                    "chunks",
                    "documents",
                ):
                    connection.execute(f"DELETE FROM {table}")  # noqa: S608
        restarted = create_app(offline_settings(api.path))
        container = restarted.state.container
        with (
            TestClient(restarted) as client,
            browsing_guards(container) as guards,
            monkeypatch.context() as patch,
        ):
            for target, method in (
                (container.document_catalog, "list_documents"),
                (container.document_chunks, "list_chunks"),
                (container.source_context, "read"),
                (container.event_log, "list_events"),
                (container.event_log, "append_transition"),
                (container.evidence_exporter, "export"),
            ):
                patch.setattr(target, method, no_work)
            before = api.path.read_bytes()
            response = client.post(
                COMPOSER,
                data={
                    **controls(opened),
                    "start": str(TEXT.rindex("Repeated quote")),
                    "end": str(TEXT.rindex("Repeated quote") + 14),
                    "note": NOTE,
                },
                headers=ORIGIN,
            )
            assert response.content == previewed.content and api.path.read_bytes() == before
            saved = client.post(SAVE, data=fields, headers=ORIGIN, follow_redirects=False)
            assert saved.status_code == 303
            persisted = client.get(saved.headers["location"])
            assert persisted.status_code == 200
            root = ExplorerDOM(persisted.text).root
            assert values(root, "passage-text") == [TEXT]
            assert values(root, "annotation-note") == [NOTE]
            if stage == "changed":
                original = container.evidence_annotations.list_annotations(
                    bundle.run_id
                ).annotations[0]
            else:
                assert container.evidence_annotations.list_annotations(
                    bundle.run_id
                ).annotations == [original]
                assert api.path.read_bytes() == before
            assert not any(guard.call_count for guard in guards.values())
        assert container.event_log.list_events() == events


@pytest.mark.parametrize("operation", ["export", "create"])
def test_failures_are_sanitized_and_insert_errors_roll_back(
    api: ExplorerAPI, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    bundle = fixture_run(api)
    page = ExplorerDOM(preview(api, open_draft(api, bundle)).text).root
    fields = controls(page, "annotation-save")
    target = (
        api.container.saved_run_evidence
        if operation == "export"
        else api.container.evidence_annotations
    )
    error = (
        EvidenceExportError("evidence_storage_unavailable", "PRIVATE_database", 503)
        if operation == "export"
        else AnnotationError("annotation_storage_unavailable", "PRIVATE_database", 503)
    )
    with monkeypatch.context() as patch:
        patch.setattr(target, operation, Mock(side_effect=error))
        before = api.path.read_bytes()
        failure(api.client.post(SAVE, data=fields, headers=ORIGIN), 503)
        assert api.path.read_bytes() == before
    with closing(sqlite3.connect(api.path)) as connection, connection:
        connection.execute(
            """CREATE TRIGGER fail_annotation BEFORE INSERT ON evidence_annotations
               BEGIN SELECT RAISE(ABORT, 'PRIVATE_insert_failure'); END"""
        )
    before = api.path.read_bytes()
    failure(api.client.post(SAVE, data=fields, headers=ORIGIN), 503)
    assert api.path.read_bytes() == before


def test_existing_json_api_keeps_its_origin_and_response_contract(api: ExplorerAPI) -> None:
    bundle = fixture_run(api)
    page = ExplorerDOM(preview(api, open_draft(api, bundle)).text).root
    payload = json.loads(controls(page, "annotation-save")["submission"])
    response = api.client.post(f"/runs/{bundle.run_id}/annotations", json=payload)
    assert response.status_code == 201
    retry = api.client.post(f"/runs/{bundle.run_id}/annotations", json=payload)
    assert retry.status_code == 200 and retry.json() == response.json()
    assert response.headers["content-type"] == "application/json"


def test_no_sources_means_no_invented_composer(api: ExplorerAPI) -> None:
    _save(ComparisonAPI(api.app, api.client, api.container, api.path), "empty", [])
    page = api.page("/explore/run/notebook", run_id="empty")
    assert "No frozen passages were captured" in page.text()
    assert not any(node.attrs.get("id", "").startswith("annotate-source-") for node in page.walk())


def test_native_openapi_has_unique_operations_and_no_new_storage_schema(api: ExplorerAPI) -> None:
    paths = api.app.openapi()["paths"]
    assert set(paths[COMPOSER]) == {"get", "post"} and set(paths[SAVE]) == {"post"}
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


def test_save_preflights_complete_reader_metadata_before_persistence(
    api: ExplorerAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    _save(
        ComparisonAPI(api.app, api.client, api.container, api.path),
        "metadata-heavy",
        [_source("chunk", text=TEXT, metadata={"expanded": "<" * 300_000})],
    )
    bundle = api.container.saved_run_evidence.export("metadata-heavy")
    page = ExplorerDOM(preview(api, open_draft(api, bundle)).text).root
    fields = controls(page, "annotation-save")
    before = api.path.read_bytes()
    create = Mock(wraps=api.container.evidence_annotations.create)
    monkeypatch.setattr(api.container.evidence_annotations, "create", create)
    response = api.client.post(SAVE, data=fields, headers=ORIGIN, follow_redirects=False)
    failure(response, 413)
    create.assert_not_called()
    assert api.path.read_bytes() == before
    assert api.container.evidence_annotations.list_annotations(bundle.run_id).annotations == []


@pytest.mark.parametrize("extra_byte", [0, 1])
def test_complete_reader_preflight_byte_boundary_with_maximum_record_fields(
    api: ExplorerAPI, extra_byte: int
) -> None:
    run_id = "\U0001f52c" * 256
    chunk_id = "\U0001f52c" * 256
    document_id = "\0\u0301" * 128
    quote = note = "\U0001f52c" * 1000
    text = quote + "\r\n"
    result = _source(chunk_id, document_id=document_id, text=text, metadata={"expanded": ""})
    source = EvidenceSource(**result.model_dump(), rank=1, text_sha256=text_digest(text))
    payload = AnnotationSubmission(
        annotation_id=UUID("00000000-0000-4000-8000-000000000001"),
        document_id=document_id,
        chunk_id=chunk_id,
        source_text_sha256=source.text_sha256,
        start=0,
        end=1000,
        quote=quote,
        note=note,
    )
    widest_timestamp = datetime.max.replace(tzinfo=UTC).isoformat().replace("+00:00", "Z")
    envelope = SavedEvidenceAnnotation.model_validate(
        {
            **payload.model_dump(),
            "schema_version": "1.0",
            "run_id": run_id,
            "sequence": MAX_EVENT_ID,
            "created_at": widest_timestamp,
        }
    )
    options = _AnnotationParameters(
        run_id=run_id,
        annotation_id=payload.annotation_id,
        review_limit=1,
        review_cursor=42,
        annotation_limit=1,
        annotation_cursor=43,
        limit=2,
        cursor=99,
        state="DONE",
        source="return\r\nfilter\0",
        title="\U0001f52c" * 100,
        catalog_limit=3,
        catalog_cursor="cursor/\u7814",
    )
    base = _page(
        "Annotated frozen source",
        annotation_fragments(envelope, source, options, "/library"),
        saved_runs=True,
    )
    count, remainder = divmod(explorer.MAX_HTML_BYTES - len(base.body), 4)
    result.chunk.metadata["expanded"] = "<" * count + "x" * (remainder + extra_byte)
    _save(ComparisonAPI(api.app, api.client, api.container, api.path), run_id, [result])
    with closing(sqlite3.connect(api.path)) as connection, connection:
        connection.execute(
            "INSERT INTO sqlite_sequence (name, seq) VALUES ('evidence_annotations', ?)",
            (MAX_EVENT_ID - 1,),
        )
    with TestClient(api.app, root_path="/library") as client:
        notebook = ExplorerDOM(
            client.get(
                "/library/explore/run/notebook",
                params={
                    key: value
                    for key, value in options.notebook_parameters().items()
                    if value is not None
                },
            ).text
        ).root
        opened = ExplorerDOM(client.get(href(notebook, "annotate-source-1")).text).root
        response = client.post(
            "/library" + COMPOSER,
            data={**controls(opened), "start": "0", "end": "1000", "note": note},
            headers=ORIGIN,
        )
        assert response.status_code == 200
        fields = controls(ExplorerDOM(response.text).root, "annotation-save")
        before = api.path.read_bytes()
        response = client.post(
            "/library" + SAVE, data=fields, headers=ORIGIN, follow_redirects=False
        )
        if extra_byte:
            failure(response, 413)
            assert api.path.read_bytes() == before
            assert api.container.evidence_annotations.list_annotations(run_id).annotations == []
            return
        assert response.status_code == 303
        location = response.headers["location"]
        persisted = client.get(location)
        assert persisted.status_code == 200
        assert len(persisted.content) <= explorer.MAX_HTML_BYTES
        root = ExplorerDOM(persisted.text).root
        assert root.by_id("annotation-quote").text() == quote
        assert values(root, "annotation-note") == [note]
        assert values(root, "annotation-sequence") == [str(MAX_EVENT_ID)]
        assert href(root, "back-to-notebook") == href(opened, "back-to-notebook")
        saved, _ = api.container.evidence_annotations.get_annotation(
            run_id, UUID(json.loads(fields["submission"])["annotation_id"])
        )
        assert len(saved.created_at.isoformat()) <= len(envelope.created_at.isoformat())
        before = api.path.read_bytes()
        retry = client.post("/library" + SAVE, data=fields, headers=ORIGIN, follow_redirects=False)
        assert retry.status_code == 303 and retry.headers["location"] == location
        assert client.get(location).content == persisted.content
        assert api.path.read_bytes() == before
