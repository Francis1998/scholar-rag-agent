"""Native human decisions use the existing revisioned service, never retrieval."""

import json
import sqlite3
from contextlib import closing
from unittest.mock import Mock
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from scripts.demo_corpus_explorer import browsing_guards
from scripts.demo_evidence_export import offline_settings
from starlette.types import Message, Scope

from api import explorer, screening_explorer
from api.application import create_app
from storage.paper_collections import MAX_REVISION, CollectionError, PaperCollection
from storage.paper_screening import ScreeningSubmission
from tests.test_collection_explorer import detail_url
from tests.test_corpus_explorer import (
    HOSTILE,
    ExplorerAPI,
    ExplorerDOM,
    assert_headers,
    href,
    identities,
    values,
)
from tests.test_corpus_explorer import api as api
from tests.test_evidence_html import Node

ENDPOINT = "/explore/screening"
ORIGIN = {"origin": "http://testserver", "sec-fetch-site": "same-origin"}


def seed(api: ExplorerAPI, identifier: str = "exact") -> PaperCollection:
    api.seed(identifier, title=HOSTILE, source="https://external.invalid/not-followed")
    return api.container.paper_collections.create(
        name="Synthetic <screening>", document_ids=[identifier]
    )


def selection(collection: PaperCollection) -> dict[str, str | int]:
    return {
        "collection_id": collection.collection_id,
        "collection_revision": collection.revision,
        "document_id": collection.document_ids[0],
    }


def fields(collection: PaperCollection, **updates: str) -> dict[str, str]:
    return {
        "selection": json.dumps(selection(collection), ensure_ascii=True),
        "expected_decision_revision": "0",
        "decision": "include",
        "reason": "Explicit synthetic reason.",
        **updates,
    }


def failure(response: httpx.Response, status: int) -> None:
    assert response.status_code == status, response.text[:1000]
    assert_headers(response, referrer_policy="same-origin")
    root = ExplorerDOM(response.text).root
    assert not root.by_tag("form") and not root.by_class("history-reason")
    assert "PRIVATE_" not in response.text
    assert "location" not in response.headers and "set-cookie" not in response.headers


def controls(root: Node) -> dict[str, str]:
    form = root.by_id("screening-form")
    return {node.attrs["name"]: node.attrs.get("value", "") for node in form.by_tag("input")}


def test_native_decision_redirect_and_restart_preserve_reason_and_history(
    api: ExplorerAPI,
) -> None:
    identifier = "paper/../exact\r\nid\0?draft#v1"
    api.seed(identifier)
    collection = api.container.paper_collections.create(
        name="Synthetic human screening", document_ids=[identifier]
    )
    opened = api.client.get(
        ENDPOINT,
        params={
            "collection_id": collection.collection_id,
            "collection_revision": 1,
            "document_id": identifier,
        },
    )
    assert opened.status_code == 200
    assert_headers(opened, referrer_policy="same-origin")
    root = ExplorerDOM(opened.text).root
    assert root.by_id("screening-form").attrs["method"] == "post"
    assert root.by_id("screening-form").attrs["action"] == ENDPOINT
    assert root.by_tag("textarea")[0].text().strip() == ""
    assert not any("selected" in option.attrs for option in root.by_tag("option")[1:])
    fields = controls(root)
    assert json.loads(fields["selection"])["document_id"] == identifier
    assert fields["expected_decision_revision"] == "0"
    detail = api.page(detail_url(collection))
    assert urlsplit(href(detail, "screen-member-1")).path == ENDPOINT
    assert api.client.get(href(root, "screening-paper")).status_code == 200
    reason = '  Explicit human reason.\r\n<script>literal, not executed</script> & "quoted".  '
    response = api.client.post(
        ENDPOINT,
        data={**fields, "decision": "include", "reason": reason},
        headers=ORIGIN,
        follow_redirects=False,
    )
    assert response.status_code == 303
    target = urlsplit(response.headers["location"])
    assert target.path == ENDPOINT and not target.netloc and not target.scheme
    assert "reason" not in parse_qs(target.query)
    assert response.headers["cache-control"] == "no-store"
    persisted = api.client.get(response.headers["location"])
    assert persisted.status_code == 200
    assert_headers(persisted, referrer_policy="same-origin")
    saved = ExplorerDOM(persisted.text).root
    assert saved.by_id("screening-saved").text() == "Stored decision revision 1 confirmed."
    assert values(saved, "latest-review-reason") == values(saved, "history-reason") == [reason]
    assert controls(saved)["expected_decision_revision"] == "1"
    assert api.container.event_log.list_events() == []
    history = api.container.paper_screening.list_history(
        collection.collection_id, identifier, collection_revision=1
    )
    assert history.latest_review is not None and history.latest_review.reason == reason
    assert history.latest_review.decision == "include" and len(history.items) == 1
    restarted = create_app(offline_settings(api.path))
    with TestClient(restarted) as client:
        assert client.get(response.headers["location"]).content == persisted.content


def test_native_post_calls_the_existing_service_once(
    api: ExplorerAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    collection = seed(api)
    submit = Mock(wraps=api.container.paper_screening.submit)
    monkeypatch.setattr(api.container.paper_screening, "submit", submit)
    with browsing_guards(api.container) as guards:
        response = api.client.post(
            ENDPOINT, data=fields(collection), headers=ORIGIN, follow_redirects=False
        )
    assert response.status_code == 303
    submit.assert_called_once_with(
        collection.collection_id,
        "exact",
        ScreeningSubmission(
            collection_revision=1,
            expected_decision_revision=0,
            decision="include",
            reason="Explicit synthetic reason.",
        ),
    )
    assert not any(guard.call_count for guard in guards.values())


@pytest.mark.parametrize("decision", ["include", "exclude", "unsure"])
def test_explicit_decisions_are_not_presets_or_retrieval_filters(
    api: ExplorerAPI, decision: str
) -> None:
    collection = seed(api)
    before = api.path.read_bytes()
    with browsing_guards(api.container) as guards:
        root = api.page(ENDPOINT, referrer_policy="same-origin", **selection(collection))
        assert api.path.read_bytes() == before
        response = api.client.post(
            ENDPOINT,
            data={**controls(root), "decision": decision, "reason": "  Explicit reason.  "},
            headers=ORIGIN,
        )
        assert response.status_code == 200
        root = ExplorerDOM(response.text).root
        assert values(root, "latest-review-decision") == [decision]
        assert values(root, "latest-review-reason") == ["  Explicit reason.  "]
        assert root.by_tag("textarea")[0].text().strip() == ""
        assert root.by_id("decision").by_tag("option")[0].attrs["value"] == ""
        assert api.container.paper_collections.get(collection.collection_id) == collection
    assert not any(guard.call_count for guard in guards.values())


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"origin": "null"},
        {"origin": "https://external.invalid"},
        {"origin": "http://testserver.evil.invalid"},
        {"origin": "https://testserver"},
        {"origin": "http://testserver:81"},
        {"origin": "http://testserver/"},
        {"origin": "http://testserver@external.invalid"},
        {"origin": "http://testserver", "sec-fetch-site": "cross-site"},
        {"origin": "http://testserver", "sec-fetch-site": "same-site"},
        {"origin": "http://testserver", "sec-fetch-site": "none"},
        {"referer": "http://testserver/explore/screening"},
        {"origin": "https://external.invalid", "x-forwarded-host": "external.invalid"},
        [("origin", "http://testserver"), ("origin", "http://testserver")],
        [
            ("origin", "http://testserver"),
            ("sec-fetch-site", "same-origin"),
            ("sec-fetch-site", "same-origin"),
        ],
    ],
)
def test_cross_origin_or_ambiguous_browser_posts_never_write(
    api: ExplorerAPI, headers: dict[str, str] | list[tuple[str, str]]
) -> None:
    collection = seed(api)
    before = api.path.read_bytes()
    failure(
        api.client.post(ENDPOINT, data=fields(collection), headers=headers, follow_redirects=False),
        403,
    )
    assert api.path.read_bytes() == before


@pytest.mark.parametrize(
    "updates",
    [
        {"decision": ""},
        {"decision": "INCLUDE"},
        {"decision": "stale"},
        {"decision": "include,exclude"},
        {"reason": ""},
        {"reason": " \r\n\t"},
        {"reason": "PRIVATE_\0invalid"},
        {"reason": "x" * 1001},
        {"expected_decision_revision": "-1"},
        {"expected_decision_revision": "+0"},
        {"expected_decision_revision": "0.0"},
        {"expected_decision_revision": "false"},
        {"expected_decision_revision": str(MAX_REVISION + 1)},
        {"selection": "{}"},
        {"selection": "[]"},
        {"selection": "[" * 1100},
        {"selection": '{"document_id":"a","document_id":"b"}'},
        {"return_to": "https://external.invalid/PRIVATE_"},
    ],
)
def test_invalid_controls_fail_without_writes_or_echo(
    api: ExplorerAPI, updates: dict[str, str]
) -> None:
    collection = seed(api)
    before = api.path.read_bytes()
    failure(api.client.post(ENDPOINT, data=fields(collection, **updates), headers=ORIGIN), 422)
    assert api.path.read_bytes() == before


@pytest.mark.parametrize(
    "suffix",
    [
        "&reason=PRIVATE_duplicate",
        "&reason=%ff",
        "&reason=%ED%A0%80",
        "&x=%",
        "&x=%Q0",
        "&x",
        "&=x",
    ],
)
def test_malformed_urlencoded_bodies_are_rejected(api: ExplorerAPI, suffix: str) -> None:
    collection = seed(api)
    before = api.path.read_bytes()
    failure(
        api.client.post(
            ENDPOINT,
            content=(urlencode(fields(collection)) + suffix).encode(),
            headers={**ORIGIN, "content-type": "application/x-www-form-urlencoded"},
        ),
        422,
    )
    assert api.path.read_bytes() == before


def test_stale_decision_collection_and_removed_member_do_not_append(api: ExplorerAPI) -> None:
    collection = seed(api)
    initial = fields(collection)
    assert api.client.post(ENDPOINT, data=initial, headers=ORIGIN).status_code == 200
    before = api.path.read_bytes()
    failure(api.client.post(ENDPOINT, data=initial, headers=ORIGIN), 409)
    assert api.path.read_bytes() == before
    collection = api.container.paper_collections.replace(
        collection.collection_id, name="Renamed", document_ids=["exact"], expected_revision=1
    )
    before = api.path.read_bytes()
    failure(
        api.client.post(
            ENDPOINT, data={**initial, "expected_decision_revision": "1"}, headers=ORIGIN
        ),
        409,
    )
    assert api.path.read_bytes() == before
    root = api.page(ENDPOINT, referrer_policy="same-origin", **selection(collection))
    assert values(root, "current-status") == ["stale"]
    assert controls(root)["expected_decision_revision"] == "1"
    api.seed("replacement", ("other",))
    api.container.paper_collections.replace(
        collection.collection_id, name="Removed", document_ids=["replacement"], expected_revision=2
    )
    invalid = {**selection(collection), "collection_revision": 3}
    before = api.path.read_bytes()
    failure(
        api.client.post(
            ENDPOINT, data=fields(collection, selection=json.dumps(invalid)), headers=ORIGIN
        ),
        422,
    )
    assert api.path.read_bytes() == before


@pytest.mark.parametrize("prefix,mounted", [("/library", False), ("/prefix/library", True)])
def test_mounts_preserve_exact_local_forms_links_and_redirects(
    api: ExplorerAPI, prefix: str, mounted: bool
) -> None:
    collection = seed(api, "//external.invalid/paper\r\nid\0")
    application = api.app
    if mounted:
        application = FastAPI()
        application.mount("/library", api.app)
    with TestClient(application, root_path="/prefix" if mounted else prefix) as client:
        root = ExplorerDOM(client.get(prefix + detail_url(collection)).text).root
        opened = client.get(href(root, "screen-member-1"))
        assert opened.status_code == 200
        form = ExplorerDOM(opened.text).root
        response = client.post(
            form.by_id("screening-form").attrs["action"],
            data={**controls(form), "decision": "unsure", "reason": "Inspect synthetic evidence."},
            headers=ORIGIN,
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert response.headers["location"].startswith(prefix + ENDPOINT + "?")
        saved = client.get(response.headers["location"])
        assert saved.status_code == 200
        for root in (form, ExplorerDOM(saved.text).root):
            for link in root.by_tag("a"):
                parts = urlsplit(link.attrs["href"])
                assert not parts.netloc and not parts.scheme
                assert not parts.path or parts.path.startswith(prefix + "/")
        assert identities(form) == [collection.document_ids[0]]


def test_history_is_append_only_and_retained_boundary_is_honest(api: ExplorerAPI) -> None:
    collection = seed(api)
    for revision in range(7):
        assert (
            api.client.post(
                ENDPOINT,
                data=fields(
                    collection,
                    expected_decision_revision=str(revision),
                    reason=f"Reason {revision + 1}",
                ),
                headers=ORIGIN,
            ).status_code
            == 200
        )
    page = api.page(ENDPOINT, referrer_policy="same-origin", **selection(collection))
    assert values(page, "history-revision") == ["7", "6", "5", "4", "3"]
    history = api.page(href(page, "screening-history"))
    assert values(history, "history-revision") == ["7", "6", "5", "4", "3", "2", "1"]
    before = api.path.read_bytes()
    unconfirmed = api.page(
        ENDPOINT, referrer_policy="same-origin", **selection(collection), saved_revision=99
    )
    assert not any(node.attrs.get("id") == "screening-saved" for node in unconfirmed.walk())
    assert api.path.read_bytes() == before
    with closing(sqlite3.connect(api.path)) as connection, connection:
        connection.execute("DELETE FROM paper_screening_history")
    legacy = api.page(ENDPOINT, referrer_policy="same-origin", **selection(collection))
    assert legacy.by_id("unretained-revisions").text() == "7"
    assert "No old decisions were backfilled" in legacy.text()


@pytest.mark.parametrize("operation", ["submit", "list_history"])
def test_service_errors_are_sanitized_and_never_success(
    api: ExplorerAPI, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    collection = seed(api)
    before = api.path.read_bytes()
    monkeypatch.setattr(
        api.container.paper_screening,
        operation,
        Mock(side_effect=CollectionError("screening_storage_unavailable", "PRIVATE_database", 503)),
    )
    response = (
        api.client.post(ENDPOINT, data=fields(collection), headers=ORIGIN)
        if operation == "submit"
        else api.client.get(ENDPOINT, params=selection(collection))
    )
    failure(response, 503)
    assert api.path.read_bytes() == before


@pytest.mark.parametrize(
    "identifier",
    [
        "/",
        "..",
        "paper/../history?draft#v1",
        "//external.invalid/path",
        "quotes\"'?%#&+\\",
        "nul\0id",
        "line\r\nid",
        "cafe\u0301/\u202eRTL\u202c-\u7814",
        "\U0001f52c" * 128,
    ],
)
def test_native_hidden_selectors_preserve_opaque_identifiers(
    api: ExplorerAPI, identifier: str
) -> None:
    collection = seed(api, identifier)
    detail = api.page(detail_url(collection, limit=1, source="filter\r\nvalue"))
    root = api.page(href(detail, "screen-member-1"), referrer_policy="same-origin")
    assert identities(root) == [identifier]
    assert json.loads(controls(root)["selection"])["document_id"] == identifier
    response = api.client.post(
        ENDPOINT,
        data={**controls(root), "decision": "exclude", "reason": HOSTILE},
        headers=ORIGIN,
    )
    assert response.status_code == 200
    saved = ExplorerDOM(response.text).root
    assert values(saved, "latest-review-reason") == [HOSTILE]
    assert href(saved, "back-to-collection") == href(root, "back-to-collection")
    paper = api.page(href(saved, "screening-paper"))
    source = api.page(href(paper, "source-context-1"))
    assert identities(source) == [identifier]
    for page in (root, saved, paper, source):
        for tag in ("script", "img", "iframe", "svg", "object", "base"):
            assert not page.by_tag(tag)
        assert all(not any(key.startswith("on") for key in node.attrs) for node in page.walk())
    history = api.container.paper_screening.list_history(
        collection.collection_id, identifier, collection_revision=1
    )
    assert history.latest_review is not None and history.latest_review.reason == HOSTILE


@pytest.mark.parametrize(
    "updates",
    [
        {"collection_revision": "1.0"},
        {"collection_revision": "1" * 20},
        {"collection_revision": "true"},
        {"collection_revision": 0},
        {"document_id": ""},
        {"document_id": " missing "},
        {"document_id": "missing"},
        {"document_id": "x" * 129},
        {"saved_revision": 0},
        {"saved_revision": MAX_REVISION + 1},
        {"decision": "include"},
        {"reason": "PRIVATE_should_not_write"},
        {"return_to": "//external.invalid"},
        {"collection_return": "{}"},
    ],
)
def test_get_is_read_only_and_strict(api: ExplorerAPI, updates: dict[str, str | int]) -> None:
    collection = seed(api)
    before = api.path.read_bytes()
    failure(api.client.get(ENDPOINT, params={**selection(collection), **updates}), 422)
    assert api.path.read_bytes() == before


@pytest.mark.parametrize(
    "headers",
    [
        {"content-type": "application/json"},
        {"content-type": "text/plain"},
        {"content-type": "multipart/form-data; boundary=fixture"},
        {"content-type": "application/x-www-form-urlencoded; charset=latin-1"},
        {"content-type": "application/x-www-form-urlencoded; extra=ignored"},
        {"content-type": "application/x-www-form-urlencoded", "content-encoding": "gzip"},
    ],
)
def test_form_encoding_is_not_guessed(api: ExplorerAPI, headers: dict[str, str]) -> None:
    collection = seed(api)
    before = api.path.read_bytes()
    failure(
        api.client.post(
            ENDPOINT, content=urlencode(fields(collection)), headers={**ORIGIN, **headers}
        ),
        415,
    )
    assert api.path.read_bytes() == before


def test_bodies_queries_methods_and_byte_limits_fail_closed(
    api: ExplorerAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    collection = seed(api)
    raw = urlencode(fields(collection)).encode()
    headers = {**ORIGIN, "content-type": "application/x-www-form-urlencoded"}
    before = api.path.read_bytes()
    for extra in (
        [("content-length", "not-numeric")],
        [("content-length", "9" * 20)],
        [("content-length", "0")],
        [("content-length", str(len(raw) + 1))],
        [("content-length", str(len(raw))), ("content-length", str(len(raw)))],
    ):
        failure(api.client.post(ENDPOINT, content=raw, headers=[*headers.items(), *extra]), 422)
    failure(api.client.post(ENDPOINT + "?decision=include", content=raw, headers=headers), 422)
    failure(
        api.client.request("GET", ENDPOINT, params=selection(collection), content=b"reason=x"), 422
    )
    for suffix in ("&document_id=duplicate", "&x=%Q0", "&x=%ff", "&x", "&"):
        failure(api.client.get(ENDPOINT + "?" + urlencode(selection(collection)) + suffix), 422)
    failure(api.client.post(ENDPOINT, content=b"x" * 65_537, headers=headers), 413)
    failure(api.client.post(ENDPOINT, content=iter([b"x" * 32_768] * 3), headers=headers), 413)
    for method in ("PUT", "PATCH", "DELETE", "OPTIONS"):
        response = api.client.request(method, ENDPOINT)
        failure(response, 405)
        assert response.headers["allow"] == "GET, POST"
    response = api.client.head(ENDPOINT)
    assert response.status_code == 405 and not response.content
    assert response.headers["cache-control"] == "no-store"
    assert api.path.read_bytes() == before
    monkeypatch.setattr(screening_explorer, "MAX_FORM_BYTES", len(raw))
    assert api.client.post(ENDPOINT, content=raw, headers=headers).status_code == 200
    before = api.path.read_bytes()
    monkeypatch.setattr(screening_explorer, "MAX_FORM_BYTES", len(raw) - 1)
    failure(api.client.post(ENDPOINT, content=raw, headers=headers), 413)
    assert api.path.read_bytes() == before


@pytest.mark.parametrize("size,status", [(65_536, 422), (65_537, 413)])
async def test_query_byte_bound_at_asgi_boundary(api: ExplorerAPI, size: int, status: int) -> None:
    before = api.path.read_bytes()
    scope: Scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "path": ENDPOINT,
        "raw_path": ENDPOINT.encode(),
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
    response = httpx.Response(
        messages[0]["status"],
        headers=messages[0]["headers"],
        content=b"".join(message.get("body", b"") for message in messages[1:]),
    )
    failure(response, status)
    assert api.path.read_bytes() == before


def test_max_reason_and_html_bounds_are_measured(
    api: ExplorerAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    collection = seed(api)
    reason = "\U0001f52c" * 1000
    response = api.client.post(ENDPOINT, data=fields(collection, reason=reason), headers=ORIGIN)
    assert response.status_code == 200
    assert values(ExplorerDOM(response.text).root, "latest-review-reason") == [reason]
    before = api.path.read_bytes()
    failure(
        api.client.post(
            ENDPOINT,
            data=fields(collection, expected_decision_revision="1", reason=reason + "x"),
            headers=ORIGIN,
        ),
        422,
    )
    url = str(response.url)
    monkeypatch.setattr(explorer, "MAX_HTML_BYTES", len(response.content))
    assert api.client.get(url).content == response.content
    monkeypatch.setattr(explorer, "MAX_HTML_BYTES", len(response.content) - 1)
    failure(api.client.get(url), 413)
    assert api.path.read_bytes() == before


def test_failed_history_insert_rolls_back_latest_decision(api: ExplorerAPI) -> None:
    collection = seed(api)
    with closing(sqlite3.connect(api.path)) as connection, connection:
        connection.execute(
            """CREATE TRIGGER fail_screening BEFORE INSERT ON paper_screening_history
               BEGIN SELECT RAISE(ABORT, 'PRIVATE_synthetic_failure'); END"""
        )
    before = api.path.read_bytes()
    failure(api.client.post(ENDPOINT, data=fields(collection), headers=ORIGIN), 503)
    assert api.path.read_bytes() == before
    history = api.container.paper_screening.list_history(
        collection.collection_id, "exact", collection_revision=1
    )
    assert history.latest_review is None and history.items == []


def test_existing_json_put_does_not_gain_browser_origin_requirements(api: ExplorerAPI) -> None:
    collection = seed(api)
    response = api.client.put(
        f"/collections/{collection.collection_id}/screening/exact",
        json={
            "collection_revision": 1,
            "expected_decision_revision": 0,
            "decision": "include",
            "reason": "Explicit existing API call.",
        },
    )
    assert response.status_code == 200
    assert response.json()["revision"] == 1 and response.json()["decision"] == "include"
    assert "text/html" not in response.headers["content-type"]


def test_origin_checks_support_explicit_ports_tls_and_ipv6(api: ExplorerAPI) -> None:
    collection = seed(api)
    for revision, origin in enumerate(
        ("http://127.0.0.1:8765", "https://example.test:8443", "http://[::1]:8765")
    ):
        with TestClient(api.app, base_url=origin) as client:
            response = client.post(
                ENDPOINT,
                data=fields(collection, expected_decision_revision=str(revision)),
                headers={"origin": origin},
                follow_redirects=False,
            )
            assert response.status_code == 303


@pytest.mark.parametrize(
    "surface,status",
    [
        ("form", 200),
        ("redirect", 303),
        ("invalid-form", 422),
        ("method-error", 405),
        ("null-origin", 403),
        ("cross-origin", 403),
    ],
)
def test_screening_referrer_policy_allows_native_posts_only_on_screening_surfaces(
    api: ExplorerAPI, surface: str, status: int
) -> None:
    collection = seed(api)
    before = api.path.read_bytes()
    if surface == "form":
        response = api.client.get(ENDPOINT, params=selection(collection))
    elif surface == "invalid-form":
        response = api.client.get(ENDPOINT)
    elif surface == "method-error":
        response = api.client.put(ENDPOINT)
    else:
        origin = {
            "redirect": "http://testserver",
            "null-origin": "null",
            "cross-origin": "https://external.invalid",
        }[surface]
        response = api.client.post(
            ENDPOINT,
            data=fields(collection),
            headers={"origin": origin},
            follow_redirects=False,
        )
    assert response.status_code == status
    assert response.headers["referrer-policy"] == "same-origin"
    if surface != "redirect":
        assert api.path.read_bytes() == before
    else:
        redirected = api.client.get(response.headers["location"])
        assert redirected.status_code == 200
        assert redirected.headers["referrer-policy"] == "same-origin"
        assert "reason" not in parse_qs(urlsplit(response.headers["location"]).query)
    for url in (
        "/explore",
        "/explore/collections",
        detail_url(collection),
        "/explore/document?document_id=exact",
        "/explore/context?document_id=exact&chunk_id=c",
        "/explore/screening-history?" + urlencode(selection(collection)),
        "/explore/runs",
        "/explore/worksheet",
    ):
        read_only = api.client.get(url)
        assert read_only.status_code == 200
        assert read_only.headers["referrer-policy"] == "no-referrer"
    worksheet_error = api.client.post("/explore/worksheet", data={"questions": ""})
    assert worksheet_error.status_code == 422
    assert worksheet_error.headers["referrer-policy"] == "no-referrer"
