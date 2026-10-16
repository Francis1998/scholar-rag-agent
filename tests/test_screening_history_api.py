"""Exact query-selected history and native read-only navigation share one service."""

import json
import sqlite3
from contextlib import closing
from urllib.parse import parse_qs, urlencode, urlsplit

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from scripts.demo_corpus_explorer import browsing_guards
from scripts.demo_evidence_export import offline_settings
from starlette.types import Message

from api import collection_explorer, explorer
from api.application import create_app
from storage import paper_screening
from storage.paper_collections import MAX_REVISION, PaperCollection
from storage.paper_screening import ScreeningSubmission
from tests.test_collection_explorer import changed, detail_url, failure
from tests.test_corpus_explorer import (
    HOSTILE,
    ExplorerAPI,
    ExplorerDOM,
    href,
    identities,
    no_work,
    values,
)
from tests.test_corpus_explorer import api as api

HISTORY = "/explore/screening-history"


def seed(
    api: ExplorerAPI, identifier: str = "exact", reason: str = "Synthetic human opinion."
) -> PaperCollection:
    api.seed(identifier)
    collection = api.container.paper_collections.create(
        name="Synthetic <history>", document_ids=[identifier]
    )
    for revision, decision in enumerate(("include", "exclude", "unsure")):
        api.container.paper_screening.submit(
            collection.collection_id,
            identifier,
            ScreeningSubmission.model_validate(
                {
                    "collection_revision": 1,
                    "expected_decision_revision": revision,
                    "decision": decision,
                    "reason": reason,
                }
            ),
        )
    return collection


def api_url(collection: PaperCollection, **params: str | int | None) -> str:
    return changed(
        f"/collections/{collection.collection_id}/screening/history",
        document_id=collection.document_ids[0],
        collection_revision=collection.revision,
        **params,
    )


@pytest.mark.parametrize(
    "identifier",
    [
        "paper/../history?draft#v1",
        "//external.invalid/path",
        "quote\"'?%#&+\\",
        "nul\0id",
        "line\r\nid",
        "cafe\u0301/\u202eRTL\u202c-\u7814",
        "\U0001f52c" * 128,
    ],
)
def test_exact_history_selectors_and_native_links(api: ExplorerAPI, identifier: str) -> None:
    collection = seed(api, identifier)
    response = api.client.get(api_url(collection, limit=2))
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    page = api.container.paper_screening.list_history(
        collection.collection_id, identifier, collection_revision=1, limit=2
    )
    assert response.json() == page.model_dump(mode="json")
    detail = api.page(detail_url(collection, status="unsure", limit=1))
    history_url = changed(href(detail, "screening-history-1"), limit=2)
    assert urlsplit(history_url).path == HISTORY
    root = api.page(history_url)
    assert identities(root) == [identifier]
    assert values(root, "history-revision") == ["3", "2"]
    assert values(root, "current-status") == ["unsure"]
    assert parse_qs(urlsplit(href(root, "back-to-collection")).query) == parse_qs(
        urlsplit(
            detail_url(collection, status="unsure", limit=1, catalog_limit=20, collections_limit=20)
        ).query
    )
    older = api.page(href(root, "next-page"))
    assert values(older, "history-revision") == ["1"] and not older.by_class("next-page")
    json_page = api.client.get(href(older, "history-json")).json()
    assert [item["revision"] for item in json_page["items"]] == [1]
    assert href(older, "back-to-collection") == href(root, "back-to-collection")
    assert "Historical includes are not current inclusion" in root.text()
    assert not root.by_tag("form")


@pytest.mark.parametrize(
    "updates",
    [
        {"document_id": None},
        {"document_id": ""},
        {"document_id": "PRIVATE_NOT_MEMBER"},
        {"collection_revision": None},
        {"collection_revision": 0},
        {"collection_revision": MAX_REVISION + 1},
        {"collection_revision": "1.0"},
        {"limit": 0},
        {"limit": 101},
        {"limit": "true"},
        {"limit": "1e0"},
        {"cursor": 0},
        {"cursor": ""},
        {"cursor": "1.0"},
        {"cursor": 4},
        {"cursor": MAX_REVISION + 1},
        {"unknown": "PRIVATE_FIELD"},
    ],
)
def test_history_rejects_invalid_query_without_echoing(
    api: ExplorerAPI, updates: dict[str, str | int | None]
) -> None:
    collection = seed(api)
    response = api.client.get(changed(api_url(collection), **updates))
    assert response.status_code == 422
    assert response.headers["cache-control"] == "no-store"
    assert "PRIVATE_" not in response.text and "items" not in response.json()
    options = {
        "collection_id": collection.collection_id,
        "collection_revision": 1,
        "document_id": "exact",
        **updates,
    }
    failure(api.client.get(changed(HISTORY, **options)), 422)


@pytest.mark.parametrize(
    "suffix",
    [
        "&limit=1&limit=2",
        "&document_id=PRIVATE_DUPLICATE",
        "&document_id=%ff",
        "&document_id=%ED%A0%80",
    ],
)
def test_duplicate_non_unicode_and_oversized_queries_fail(api: ExplorerAPI, suffix: str) -> None:
    collection = seed(api)
    response = api.client.get(api_url(collection) + suffix)
    assert response.status_code == 422 and "PRIVATE_" not in response.text
    failure(
        api.client.get(
            changed(
                HISTORY,
                collection_id=collection.collection_id,
                collection_revision=1,
                document_id="exact",
            )
            + suffix
        ),
        422,
    )


@pytest.mark.parametrize("prefix,mounted", [("/library", False), ("/prefix/library", True)])
def test_history_root_prefix_and_error_return_navigation(
    api: ExplorerAPI, prefix: str, mounted: bool
) -> None:
    collection = seed(api)
    application = api.app
    if mounted:
        application = FastAPI()
        application.mount("/library", api.app)
    with TestClient(application, root_path="/prefix" if mounted else prefix) as client:
        detail = ExplorerDOM(client.get(prefix + detail_url(collection)).text).root
        url = changed(href(detail, "screening-history-1"), limit=1)
        root = ExplorerDOM(client.get(url).text).root
        for link in (
            "back-to-collection",
            "back-to-collections",
            "back-to-catalog",
            "next-page",
            "history-json",
        ):
            target = href(root, link)
            assert target.startswith(prefix + "/")
            assert client.get(target).status_code == 200
        api.container.paper_collections.replace(
            collection.collection_id,
            name="New revision",
            document_ids=collection.document_ids,
            expected_revision=1,
        )
        failed = failure(client.get(url), 409)
        assert href(failed, "back-to-collection") == href(root, "back-to-collection")
        assert href(failed, "back-to-collections").startswith(prefix + "/")
        assert client.get(href(root, "history-json")).status_code == 409


def test_literal_history_reasons_original_timestamps_and_html_bound(
    api: ExplorerAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    reason = "\n\r\t" + HOSTILE + "\r\nExact reason."
    collection = seed(api, reason=reason)
    detail = api.page(detail_url(collection))
    url = href(detail, "screening-history-1")
    response = api.client.get(url)
    root = ExplorerDOM(response.text).root
    assert values(root, "history-reason") == [reason] * 3
    assert values(root, "latest-review-reason") == [reason]
    assert values(root, "collection-name") == [collection.name]
    records = api.client.get(api_url(collection)).json()["items"]
    for record in records:
        assert record["updated_at"].replace("Z", "+00:00") in root.text()
    for tag in ("script", "iframe", "img", "base", "svg", "object"):
        assert not root.by_tag(tag)
    assert all(not any(key.startswith("on") for key in node.attrs) for node in root.walk())
    monkeypatch.setattr(explorer, "MAX_HTML_BYTES", len(response.content))
    assert api.client.get(url).content == response.content
    monkeypatch.setattr(explorer, "MAX_HTML_BYTES", len(response.content) - 1)
    failure(api.client.get(url), 413)


def test_legacy_boundary_stale_readded_member_and_private_errors(api: ExplorerAPI) -> None:
    collection = seed(api)
    with closing(sqlite3.connect(api.path)) as connection, connection:
        connection.execute("DELETE FROM paper_screening_history")
    legacy = api.page(
        changed(
            HISTORY,
            collection_id=collection.collection_id,
            collection_revision=1,
            document_id="exact",
        )
    )
    assert "legacy latest-only" in legacy.text()
    assert legacy.by_id("unretained-revisions").text() == "3"
    assert not legacy.by_tag("article")
    api.container.paper_screening.submit(
        collection.collection_id,
        "exact",
        ScreeningSubmission(
            collection_revision=1,
            expected_decision_revision=3,
            decision="include",
            reason="Newly retained opinion.",
        ),
    )
    collection = api.container.paper_collections.replace(
        collection.collection_id,
        name=collection.name,
        document_ids=collection.document_ids,
        expected_revision=1,
    )
    detail = api.page(detail_url(collection))
    url = href(detail, "screening-history-1")
    root = api.page(url)
    assert values(root, "current-status") == ["stale"]
    assert root.by_id("first-retained-revision").text() == "4"
    assert root.by_id("unretained-revisions").text() == "3"
    assert values(root, "history-decision") == ["include"]
    with closing(sqlite3.connect(api.path)) as connection, connection:
        connection.execute("UPDATE paper_screening_history SET record = 'PRIVATE_BAD_HISTORY'")
    failed = api.client.get(api_url(collection))
    assert failed.status_code == 409 and "PRIVATE_" not in failed.text
    failure(api.client.get(url), 409)
    with closing(sqlite3.connect(api.path)) as connection, connection:
        connection.execute("DROP TABLE paper_screening_history")
    assert api.client.get(api_url(collection)).status_code == 503
    failure(api.client.get(url), 503)


def test_history_reads_do_not_write_or_call_other_work_and_restart_identically(
    api: ExplorerAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    collection = seed(api)
    url = href(api.page(detail_url(collection)), "screening-history-1")
    before = api.path.read_bytes()
    outputs: dict[str, bytes] = {}
    for application in (api.app, create_app(offline_settings(api.path))):
        container = application.state.container
        with (
            TestClient(application) as client,
            browsing_guards(container) as guards,
            monkeypatch.context() as patch,
        ):
            for target, method in (
                (container.paper_screening, "submit"),
                (container.paper_screening, "export_results"),
                (container.event_log, "append_transition"),
                (container.worksheets, "build"),
            ):
                patch.setattr(target, method, no_work)
            for target_url in (url, api_url(collection)):
                response = client.get(target_url)
                assert response.status_code == 200
                assert outputs.setdefault(target_url, response.content) == response.content
            assert not any(guard.call_count for guard in guards.values())
        assert api.path.read_bytes() == before and container.event_log.list_events() == []


def test_api_and_browser_enforce_same_serialized_byte_limit(
    api: ExplorerAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    collection = seed(api, reason="\U0001f52c" * 1000)
    url = href(api.page(detail_url(collection)), "screening-history-1")
    response = api.client.get(api_url(collection))
    monkeypatch.setattr(paper_screening, "MAX_HISTORY_RESPONSE_BYTES", len(response.content))
    assert api.client.get(api_url(collection)).content == response.content
    api.page(url)
    monkeypatch.setattr(paper_screening, "MAX_HISTORY_RESPONSE_BYTES", len(response.content) - 1)
    assert api.client.get(api_url(collection)).status_code == 413
    failure(api.client.get(url), 413)


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE", "HEAD"])
def test_browser_cannot_write(api: ExplorerAPI, method: str) -> None:
    before = api.path.read_bytes()
    response = api.client.request(method, HISTORY)
    assert response.status_code == 405 and response.headers["allow"] == "GET"
    assert response.headers["cache-control"] == "no-store"
    assert "script-src 'none'" in response.headers["content-security-policy"]
    assert api.path.read_bytes() == before


def test_maximum_unicode_and_control_return_state_is_preserved(
    api: ExplorerAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    collection = seed(api, "\U0001f52c" * 128)
    detail = api.page(
        detail_url(
            collection,
            limit=100,
            source="\U0001f52c" * 512,
            title="\U0001f52c" * 300,
            catalog_cursor="line\r\nid",
        )
    )
    history_url = href(detail, "screening-history-1")
    size = len(urlsplit(history_url).query.encode())
    assert size < 65_536
    root = api.page(history_url)
    returned = parse_qs(urlsplit(href(root, "back-to-collection")).query)
    assert returned["catalog_cursor"] == ["line\r\nid"]
    assert returned["source"] == ["\U0001f52c" * 512]
    assert api.client.get(href(root, "history-json")).status_code == 200
    bad = json.dumps({"collection_id": "col_" + "0" * 32, "collection_revision": 1})
    failure(api.client.get(changed(history_url, collection_return=bad)), 422)
    failure(
        api.client.get(
            HISTORY
            + "?"
            + urlencode(
                {
                    "collection_id": collection.collection_id,
                    "collection_revision": 1,
                    "document_id": "exact",
                    "collection_return": '{"return_url":"https://PRIVATE.invalid"}',
                }
            )
        ),
        422,
    )
    monkeypatch.setattr(collection_explorer, "MAX_COLLECTION_QUERY_BYTES", size)
    api.page(history_url)
    monkeypatch.setattr(collection_explorer, "MAX_COLLECTION_QUERY_BYTES", size - 1)
    failure(api.client.get(history_url), 422)


@pytest.mark.parametrize("browser", [False, True])
async def test_raw_query_bound_before_parsing(api: ExplorerAPI, browser: bool) -> None:
    collection = seed(api)
    path = HISTORY if browser else urlsplit(api_url(collection)).path
    sent: list[Message] = []

    async def receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: Message) -> None:
        sent.append(message)

    await api.app(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"PRIVATE=" + b"x" * (65_536 if browser else 16_384),
            "root_path": "",
            "headers": [],
            "client": ("testclient", 1234),
            "server": ("testserver", 80),
        },
        receive,
        send,
    )
    assert sent[0]["status"] == 422
    assert b"PRIVATE" not in b"".join(message.get("body", b"") for message in sent)
