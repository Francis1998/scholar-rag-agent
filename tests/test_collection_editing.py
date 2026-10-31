"""Explicit native deltas preserve hidden membership and never erase research evidence."""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from unittest.mock import patch
from urllib.parse import parse_qs, urlencode, urlsplit

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from scripts.demo_corpus_explorer import browsing_guards
from scripts.demo_evidence_export import offline_settings

from api import collection_editing, explorer
from api.application import create_app
from storage.paper_collections import CollectionError, PaperCollection
from tests.test_collection_creation import ENCODING, ORIGIN, failure
from tests.test_corpus_explorer import HOSTILE, ExplorerAPI, ExplorerDOM, href, no_work
from tests.test_corpus_explorer import api as api
from tests.test_evidence_html import Node

ENDPOINT = "/explore/collections/edit"


def seed(api: ExplorerAPI) -> PaperCollection:
    for identifier in ("a", "b", "c", "hidden"):
        api.seed(identifier, (identifier + "-chunk",), title="Synthetic " + identifier)
    return api.container.paper_collections.create(name="Before", document_ids=["a", "hidden"])


def options(collection: PaperCollection) -> dict[str, str | int]:
    return {
        "collection_id": collection.collection_id,
        "expected_revision": collection.revision,
    }


def fields(collection: PaperCollection, **changes: str) -> dict[str, str]:
    return {
        "collection_id": collection.collection_id,
        "expected_revision": str(collection.revision),
        "name": collection.name,
        "action": "preview",
        **changes,
    }


def controls(root: Node, form_id: str) -> dict[str, str]:
    return {
        node.attrs["name"]: node.attrs.get("value", "")
        for node in root.by_id(form_id).by_tag("input")
        if node.attrs.get("type") != "checkbox"
    }


def test_native_editor_is_discoverable_and_get_never_changes_selection(api: ExplorerAPI) -> None:
    collection = seed(api)
    before = api.path.read_bytes()
    with browsing_guards(api.container) as guards:
        detail = api.page(
            "/explore/collection",
            collection_id=collection.collection_id,
            collection_revision=collection.revision,
            limit=1,
        )
        opened = api.page(
            href(detail, "edit-collection") + "&title=Synthetic+b",
            referrer_policy="same-origin",
        )
        listed = api.page("/explore/collections")
        assert urlsplit(href(listed, "edit-collection-1")).path == ENDPOINT
        form = opened.by_id("collection-edit")
        assert form.attrs["action"] == ENDPOINT and form.attrs["method"] == "post"
        assert controls(opened, "collection-edit") == fields(collection)
        choices = {
            node.attrs["name"]: json.loads(node.attrs["value"])
            for node in form.by_tag("input")
            if node.attrs.get("type") == "checkbox"
        }
        assert choices == {"remove_1": "a", "remove_2": "hidden", "add_1": "b"}
        assert not any("checked" in node.attrs for node in form.by_tag("input"))
        assert "unchecked" in opened.text() and "not carried" in opened.text()
    assert not any(guard.call_count for guard in guards.values())
    assert api.path.read_bytes() == before


def test_preview_then_save_preserves_hidden_members_rejects_replays_and_restarts(
    api: ExplorerAPI,
) -> None:
    collection = seed(api)
    store = api.container.paper_collections
    before = api.path.read_bytes()
    payload = fields(collection, name=" After ", remove_1='"a"', add_1='"b"')
    with (
        browsing_guards(api.container) as guards,
        patch.object(api.container.paper_screening, "submit", side_effect=no_work),
        patch.object(api.container.worksheets, "build", side_effect=no_work),
        patch.object(api.container.event_log, "append_transition", side_effect=no_work),
    ):
        preview = api.client.post(ENDPOINT, data=payload, headers=ORIGIN)
        assert preview.status_code == 200
        root = ExplorerDOM(preview.text).root
        assert api.path.read_bytes() == before
        assert root.by_id("before-count").text() == "2"
        assert root.by_id("after-count").text() == "2"
        assert root.by_id("retained-count").text() == "1"
        saved_fields = controls(root, "collection-save")
        assert saved_fields == {**payload, "name": "After", "action": "save"}
        saved = api.client.post(
            root.by_id("collection-save").attrs["action"],
            data=saved_fields,
            headers=ORIGIN,
            follow_redirects=False,
        )
        assert saved.status_code == 303
        assert saved.headers["cache-control"] == "no-store"
        current = store.get(collection.collection_id)
        assert current.document_ids == ("hidden", "b") and current.name == "After"
        assert current.revision == 2
        assert parse_qs(urlsplit(saved.headers["location"]).query) == {
            "collection_id": [collection.collection_id],
            "collection_revision": ["2"],
        }
        after = api.path.read_bytes()
        for action in ("save", "save", "preview"):
            failure(
                api.client.post(ENDPOINT, data={**saved_fields, "action": action}, headers=ORIGIN),
                409,
            )
            assert api.path.read_bytes() == after
    assert not any(guard.call_count for guard in guards.values())
    assert api.container.event_log.list_events() == []
    with TestClient(create_app(offline_settings(api.path))) as client:
        assert client.get(saved.headers["location"]).status_code == 200
        assert client.get("/collections/" + collection.collection_id).json() == current.model_dump(
            mode="json"
        )
    assert api.path.read_bytes() == after


def test_pages_filter_only_candidates_and_keep_exact_controls(api: ExplorerAPI) -> None:
    collection = seed(api)
    selected = 'control/../\r\nid\0?x#"\u7814'
    api.seed(selected, (), title=HOSTILE, source=HOSTILE)
    first = api.page(ENDPOINT, referrer_policy="same-origin", **options(collection), limit=1)
    second = api.page(href(first, "next-page"), referrer_policy="same-origin")
    for root in (first, second):
        removal = [
            json.loads(node.attrs["value"])
            for node in root.by_id("collection-edit").by_tag("input")
            if node.attrs.get("name", "").startswith("remove_")
        ]
        assert removal == ["a", "hidden"]
        filters = root.by_id("collection-candidates")
        assert filters.attrs["method"] == "get"
        assert controls(root, "collection-candidates")["collection_id"] == collection.collection_id
        assert "cursor" not in controls(root, "collection-candidates")
    exact = api.page(ENDPOINT, referrer_policy="same-origin", **options(collection), source=HOSTILE)
    for tag in ("script", "img", "iframe", "svg", "object", "base"):
        assert not exact.by_tag(tag)
    checkbox = next(
        node for node in exact.by_tag("input") if node.attrs.get("name", "").startswith("add_")
    )
    assert json.loads(checkbox.attrs["value"]) == selected
    preview = api.client.post(
        ENDPOINT,
        data=fields(collection, add_1=checkbox.attrs["value"], name='<draft> & "native"'),
        headers=ORIGIN,
    )
    assert preview.status_code == 200
    saved = api.client.post(
        ENDPOINT,
        data=controls(ExplorerDOM(preview.text).root, "collection-save"),
        headers=ORIGIN,
        follow_redirects=False,
    )
    assert saved.status_code == 303
    current = api.container.paper_collections.get(collection.collection_id)
    assert current.document_ids == ("a", "hidden", selected)


@pytest.mark.parametrize(
    "changes",
    [
        {"expected_revision": "0"},
        {"expected_revision": "true"},
        {"expected_revision": "1.0"},
        {"expected_revision": "9" * 20},
        {"action": "replace"},
        {"name": ""},
        {"name": "PRIVATE_\0name"},
        {"name": "x" * 121},
        {"add_1": '" a "'},
        {"add_1": '"a"'},
        {"add_1": '"b"', "add_2": '"b"'},
        {"add_1": '"b"', "remove_1": '"b"'},
        {"add_1": '"missing"'},
        {"add_1": '["b"]'},
        {"add_1": '"\\ud800"'},
        {"add_0": '"b"'},
        {"add_01": '"b"'},
        {"add_101": '"b"'},
        {"remove_1": '"b"'},
        {"remove_1": '"a"', "remove_2": '"a"'},
        {"remove_1": '"a"', "remove_2": '"hidden"'},
        {"document_ids": '["b"]'},
        {"return_to": "//external.invalid/PRIVATE_"},
    ],
)
def test_invalid_or_ambiguous_edits_do_not_write(api: ExplorerAPI, changes: dict[str, str]) -> None:
    collection = seed(api)
    before = api.path.read_bytes()
    for action in ("preview", "save"):
        failure(
            api.client.post(
                ENDPOINT,
                data=fields(collection, name="New", **{"action": action, **changes})
                if "name" not in changes
                else fields(collection, action=action, **changes),
                headers=ORIGIN,
            ),
            422,
        )
        assert api.path.read_bytes() == before


def test_missing_old_member_fails_even_when_explicitly_removed_and_api_can_recover(
    api: ExplorerAPI,
) -> None:
    collection = seed(api)
    with closing(sqlite3.connect(api.path)) as connection, connection:
        connection.execute("DELETE FROM documents WHERE document_id = 'hidden'")
    before = api.path.read_bytes()
    failure(api.client.get(ENDPOINT, params=options(collection)), 409)
    for action in ("preview", "save"):
        failure(
            api.client.post(
                ENDPOINT,
                data=fields(collection, action=action, remove_1='"hidden"'),
                headers=ORIGIN,
            ),
            409,
        )
        assert api.path.read_bytes() == before
    recovered = api.client.put(
        "/collections/" + collection.collection_id,
        json={"name": "Recovered", "document_ids": ["a"], "expected_revision": 1},
    )
    assert recovered.status_code == 200 and recovered.json()["document_ids"] == ["a"]


def test_noop_conflicting_name_and_concurrent_save_are_atomic(api: ExplorerAPI) -> None:
    collection = seed(api)
    api.container.paper_collections.create(name="Occupied", document_ids=["c"])
    before = api.path.read_bytes()
    for changes, status in (({}, 422), ({"name": "Occupied"}, 409)):
        failure(
            api.client.post(ENDPOINT, data=fields(collection, **changes), headers=ORIGIN), status
        )
        assert api.path.read_bytes() == before

    def save(_: int) -> int:
        return api.client.post(
            ENDPOINT,
            data=fields(collection, name="Renamed", action="save"),
            headers=ORIGIN,
            follow_redirects=False,
        ).status_code

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(save, range(2))) == [303, 409]
    current = api.container.paper_collections.get(collection.collection_id)
    assert current.revision == 2 and current.document_ids == collection.document_ids


def test_wire_validation_and_origin_precede_any_edit(api: ExplorerAPI) -> None:
    collection = seed(api)
    before = api.path.read_bytes()
    raw = urlencode(fields(collection, name="New", action="save"))
    headers = {**ORIGIN, "content-type": ENCODING}
    for origin in ({}, {"origin": "null"}, {"origin": "https://external.invalid"}):
        failure(
            api.client.post(ENDPOINT, content=raw, headers={**origin, "content-type": ENCODING}),
            403,
        )
    for extra in (
        [("origin", "http://testserver")],
        [("host", "testserver"), ("host", "testserver")],
        [("sec-fetch-site", "cross-site")],
    ):
        failure(api.client.post(ENDPOINT, content=raw, headers=[*headers.items(), *extra]), 403)
    for suffix in (
        "&name=PRIVATE_",
        "&add_1=%22b%22&add_1=%22c%22",
        "&expected_revision=1",
        "&action=save",
        "&unknown=x",
        "&x=%",
        "&x=%Q0",
        "&x=%ff",
        "&x",
        "&",
    ):
        failure(api.client.post(ENDPOINT, content=raw + suffix, headers=headers), 422)
    for encoding in ("application/json", "multipart/form-data", ENCODING + "; charset=latin-1"):
        failure(
            api.client.post(ENDPOINT, content=raw, headers={**ORIGIN, "content-type": encoding}),
            415,
        )
    failure(
        api.client.post(ENDPOINT, content=raw, headers={**headers, "content-encoding": "gzip"}), 415
    )
    failure(api.client.post(ENDPOINT + "?name=PRIVATE_", content=raw, headers=headers), 422)
    failure(api.client.request("GET", ENDPOINT, content=raw), 422)
    for query in (
        "",
        "x=PRIVATE_",
        urlencode(options(collection)) + "&limit=101",
        urlencode(options(collection)) + "&expected_revision=1",
    ):
        failure(api.client.get(ENDPOINT + "?" + query), 422)
    failure(api.client.get(ENDPOINT + "?" + "x" * 16_385), 413)
    failure(api.client.post(ENDPOINT, content=b"x" * 524_289, headers=headers), 413)
    for method in ("PUT", "DELETE", "OPTIONS"):
        response = api.client.request(method, ENDPOINT)
        failure(response, 405)
        assert response.headers["allow"] == "GET, POST"
    assert api.path.read_bytes() == before


@pytest.mark.parametrize("mounted", [False, True])
def test_root_path_and_mount_preview_save_and_navigation(api: ExplorerAPI, mounted: bool) -> None:
    collection = seed(api)
    application = api.app
    prefix = "/prefix"
    if mounted:
        application = FastAPI()
        application.mount("/library", api.app)
        prefix += "/library"
    with TestClient(application, root_path="/prefix") as client:
        opened = client.get(prefix + ENDPOINT, params=options(collection))
        assert opened.status_code == 200
        editor = ExplorerDOM(opened.text).root
        preview = client.post(
            editor.by_id("collection-edit").attrs["action"],
            data=fields(collection, name="Mounted"),
            headers=ORIGIN,
        )
        assert preview.status_code == 200
        reviewed = ExplorerDOM(preview.text).root
        saved = client.post(
            reviewed.by_id("collection-save").attrs["action"],
            data=controls(reviewed, "collection-save"),
            headers=ORIGIN,
            follow_redirects=False,
        )
        assert saved.status_code == 303
        assert saved.headers["location"].startswith(prefix + "/explore/collection?")
        assert client.get(saved.headers["location"]).status_code == 200
        for root in (editor, reviewed):
            for link in root.by_tag("a"):
                parts = urlsplit(link.attrs["href"])
                assert not parts.netloc and not parts.scheme
                assert not parts.path or parts.path.startswith(prefix + "/")
            for form in root.by_tag("form"):
                assert form.attrs["action"].startswith(prefix + "/")


def test_maximum_exact_deltas_and_measured_body_html_bounds(
    api: ExplorerAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    identifiers = tuple(f"{index:03d}" + "\U0001f52c" * 125 for index in range(200))
    for identifier in identifiers:
        api.seed(identifier, ())
    collection = api.container.paper_collections.create(
        name="Maximum", document_ids=identifiers[:100]
    )
    parameters = {**options(collection), "limit": 100, "cursor": identifiers[99]}
    response = api.client.get(ENDPOINT, params=parameters)
    assert response.status_code == 200
    root = ExplorerDOM(response.text).root
    payload = {
        **controls(root, "collection-edit"),
        "name": "\U0001f52c" * 120,
        **{
            node.attrs["name"]: node.attrs["value"]
            for node in root.by_id("collection-edit").by_tag("input")
            if node.attrs.get("type") == "checkbox"
        },
    }
    assert len(payload) == 204
    raw = urlencode(payload).encode()
    assert len(raw) < collection_editing.MAX_FORM_BYTES
    before = api.path.read_bytes()
    monkeypatch.setattr(explorer, "MAX_HTML_BYTES", len(response.content))
    assert api.client.get(ENDPOINT, params=parameters).status_code == 200
    monkeypatch.setattr(explorer, "MAX_HTML_BYTES", len(response.content) - 1)
    failure(api.client.get(ENDPOINT, params=parameters), 413)
    monkeypatch.setattr(explorer, "MAX_HTML_BYTES", 1_048_576)
    monkeypatch.setattr(collection_editing, "MAX_FORM_BYTES", len(raw) - 1)
    failure(api.client.post(ENDPOINT, data=payload, headers=ORIGIN), 413)
    monkeypatch.setattr(collection_editing, "MAX_FORM_BYTES", len(raw))
    preview = api.client.post(ENDPOINT, content=raw, headers={**ORIGIN, "content-type": ENCODING})
    assert preview.status_code == 200
    assert api.path.read_bytes() == before
    saved = api.client.post(
        ENDPOINT,
        data=controls(ExplorerDOM(preview.text).root, "collection-save"),
        headers=ORIGIN,
        follow_redirects=False,
    )
    assert saved.status_code == 303
    assert (
        api.container.paper_collections.get(collection.collection_id).document_ids
        == identifiers[100:]
    )


def test_result_limit_and_disappearing_additions_fail_before_any_write(api: ExplorerAPI) -> None:
    identifiers = [f"paper-{index}" for index in range(101)]
    for identifier in identifiers:
        api.seed(identifier, ())
    collection = api.container.paper_collections.create(name="Full", document_ids=identifiers[:100])
    payload = fields(collection, add_1=json.dumps(identifiers[100]))
    before = api.path.read_bytes()
    failure(api.client.post(ENDPOINT, data=payload, headers=ORIGIN), 422)
    assert api.path.read_bytes() == before
    payload["remove_1"] = json.dumps(identifiers[0])
    preview = api.client.post(ENDPOINT, data=payload, headers=ORIGIN)
    assert preview.status_code == 200
    with closing(sqlite3.connect(api.path)) as connection, connection:
        connection.execute("DELETE FROM documents WHERE document_id = ?", (identifiers[100],))
    before = api.path.read_bytes()
    failure(
        api.client.post(
            ENDPOINT,
            data=controls(ExplorerDOM(preview.text).root, "collection-save"),
            headers=ORIGIN,
        ),
        422,
    )
    assert api.path.read_bytes() == before


def test_storage_errors_labels_and_diagnostics_are_sanitized(
    api: ExplorerAPI, caplog: pytest.LogCaptureFixture
) -> None:
    collection = seed(api)
    before = api.path.read_bytes()
    with patch.object(
        api.container.paper_collections,
        "edit",
        side_effect=CollectionError("collection_storage_error", "PRIVATE_database", 503),
    ):
        failure(
            api.client.post(
                ENDPOINT,
                data=fields(collection, name="PRIVATE_name", action="save"),
                headers=ORIGIN,
            ),
            503,
        )
    with patch.object(
        api.container.document_catalog,
        "list_documents",
        side_effect=sqlite3.OperationalError("PRIVATE_path"),
    ):
        failure(api.client.get(ENDPOINT, params=options(collection)), 503)
    assert "collection_storage_error" in caplog.text and "PRIVATE_" not in caplog.text
    assert api.path.read_bytes() == before
    api.seed("bad-label", (), title="PRIVATE_\0invalid")
    failure(api.client.get(ENDPOINT, params=options(collection)), 409)
