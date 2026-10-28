"""Native collection creation selects exact IDs without running research or screening."""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from unittest.mock import patch
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from scripts.demo_corpus_explorer import browsing_guards
from scripts.demo_evidence_export import offline_settings

from api import collection_creation, explorer
from api.application import create_app
from storage.paper_collections import CollectionError
from tests.test_corpus_explorer import (
    HOSTILE,
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

ENDPOINT = "/explore/collections/new"
ORIGIN = {"origin": "http://testserver", "sec-fetch-site": "same-origin"}
ENCODING = "application/x-www-form-urlencoded"


def choices(root: Node) -> dict[str, str]:
    return {
        node.attrs["name"]: node.attrs["value"]
        for node in root.by_tag("input")
        if node.attrs.get("type") == "checkbox"
    }


def failure(response: httpx.Response, status: int) -> None:
    assert response.status_code == status, response.text[:1000]
    assert_headers(response, referrer_policy="same-origin")
    assert "location" not in response.headers and "set-cookie" not in response.headers
    assert not ExplorerDOM(response.text).root.by_tag("form")
    assert "PRIVATE_" not in response.text


def test_native_creation_form_is_discoverable_and_starts_unselected(api: ExplorerAPI) -> None:
    api.seed("paper/../exact\r\nid\0?draft#v1")
    before = api.path.read_bytes()
    opened = api.client.get(ENDPOINT)
    assert opened.status_code == 200
    assert_headers(opened, referrer_policy="same-origin")
    root = ExplorerDOM(opened.text).root
    form = root.by_id("collection-create")
    assert form.attrs["method"] == "post"
    assert form.attrs["action"] == ENDPOINT
    checkboxes = [node for node in form.by_tag("input") if node.attrs.get("type") == "checkbox"]
    assert len(checkboxes) == 1
    assert checkboxes[0].attrs["name"] == "document_1"
    assert json.loads(checkboxes[0].attrs["value"]) == "paper/../exact\r\nid\0?draft#v1"
    assert "checked" not in checkboxes[0].attrs
    assert root.by_id("collection-name").attrs.get("value", "") == ""
    assert "not carried" in root.text()
    for path in ("/explore", "/explore/collections"):
        assert urlsplit(href(api.page(path), "create-collection")).path == ENDPOINT
    assert api.path.read_bytes() == before


def test_native_create_redirects_to_persisted_exact_selection_and_reopens(
    api: ExplorerAPI,
) -> None:
    selected = "paper/../exact\r\nid\0?draft#v1"
    api.seed(selected)
    api.seed("unselected", ("other-chunk",))
    response = api.client.post(
        ENDPOINT,
        data={"name": "  Synthetic selection  ", "document_1": json.dumps(selected)},
        headers=ORIGIN,
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["cache-control"] == "no-store"
    location = response.headers["location"]
    target = urlsplit(location)
    assert target.path == "/explore/collection" and not target.netloc and not target.scheme
    parameters = parse_qs(target.query)
    collection = api.container.paper_collections.get(parameters["collection_id"][0])
    assert parameters["collection_revision"] == [str(collection.revision)]
    assert collection.document_ids == (selected,) and collection.name == "Synthetic selection"
    saved = api.page(location)
    assert identities(saved) == [selected]
    for link in ("screen-member-1", "search-collection", "worksheet-collection"):
        assert api.client.get(href(saved, link)).status_code == 200
    assert api.container.event_log.list_events() == []
    with TestClient(create_app(offline_settings(api.path))) as client:
        assert client.get(location).content == api.client.get(location).content


def test_empty_and_unknown_native_selections_are_atomic(api: ExplorerAPI) -> None:
    api.seed("known")
    before = api.path.read_bytes()
    for fields in (
        {"name": "No selection"},
        {"name": "Missing paper", "document_1": '"known"', "document_2": '"unknown"'},
    ):
        response = api.client.post(ENDPOINT, data=fields, headers=ORIGIN, follow_redirects=False)
        assert response.status_code == 422
        assert "location" not in response.headers
        assert_headers(response, referrer_policy="same-origin")
        assert api.path.read_bytes() == before
    assert api.container.paper_collections.list_collections().collections == []


def test_source_title_filters_pages_and_entry_links_keep_exact_catalog_scope(
    api: ExplorerAPI,
) -> None:
    for identifier, title, source in (
        ("a", "100%_Graph methods", "synthetic:chosen"),
        ("b\r\ncursor", "100%_GRAPH limitations", "synthetic:chosen"),
        ("c", "100%_Graph notes", "synthetic:chosen"),
        ("d", "Unrelated note", "synthetic:chosen"),
        ("e", "100%_Graph excluded source", "SYNTHETIC:CHOSEN"),
    ):
        api.seed(identifier, (), title=title, source=source)
    options = {"limit": 2, "source": "synthetic:chosen", "title": "%_graph"}
    catalog = api.page(**options)
    listed = api.page(href(catalog, "saved-collections"))
    assert href(listed, "create-collection") == href(catalog, "create-collection")
    before = api.path.read_bytes()
    with browsing_guards(api.container) as guards:
        first = api.page(href(catalog, "create-collection"), referrer_policy="same-origin")
        assert identities(first) == ["a", "b\r\ncursor"]
        form = first.by_tag("form")[0]
        assert form.attrs["method"] == "get" and form.attrs["action"] == ENDPOINT
        assert "cursor" not in {node.attrs["name"] for node in form.by_tag("input")}
        next_url = href(first, "next-page")
        assert parse_qs(urlsplit(next_url).query)["cursor"] == ["b\r\ncursor"]
        second = api.page(next_url, referrer_policy="same-origin")
        assert identities(second) == ["c"] and not second.by_class("next-page")
        assert len(choices(first)) == 2 and len(choices(second)) == 1
        for root in (first, second):
            assert root.by_id("collection-name").attrs.get("value", "") == ""
            assert not any("checked" in node.attrs for node in root.by_tag("input"))
            assert "not carried" in root.text() and "no hidden pages" in root.text()
        assert parse_qs(urlsplit(href(second, "back-to-catalog")).query) == parse_qs(
            urlsplit(next_url).query
        )
        assert "cursor" not in parse_qs(urlsplit(href(second, "first-page")).query)
        empty = api.page(ENDPOINT, referrer_policy="same-origin", title="no matching paper")
        assert not choices(empty) and len(empty.by_tag("form")) == 1
    assert api.path.read_bytes() == before and not any(g.call_count for g in guards.values())


def test_opaque_checkbox_values_and_labels_round_trip_without_normalization(
    api: ExplorerAPI,
) -> None:
    identifiers = (
        "/",
        "//external.invalid/paper/../?draft#v1",
        "quotes\"'?%#&+\\",
        "line\r\nid\0",
        "cafe\u0301/\u202eRTL\u202c-\u7814",
        "\U0001f52c" * 128,
    )
    for identifier in identifiers:
        api.seed(identifier, (), title=HOSTILE, source=HOSTILE)
    root = api.page(ENDPOINT, referrer_policy="same-origin")
    fields = choices(root)
    expected = tuple(sorted(identifiers))
    assert tuple(json.loads(value) for value in fields.values()) == expected
    assert values(root, "document-title") == [HOSTILE] * len(identifiers)
    for tag in ("script", "img", "iframe", "svg", "object", "base"):
        assert not root.by_tag(tag)
    assert not any(key.startswith("on") for node in root.walk() for key in node.attrs)
    labels = {node.attrs["for"] for node in root.by_tag("label")}
    assert all(node.attrs["id"] in labels for node in root.by_tag("input"))
    response = api.client.post(
        ENDPOINT,
        data={"name": '<draft> & "native"', **dict(reversed(list(fields.items())))},
        headers=ORIGIN,
        follow_redirects=False,
    )
    assert response.status_code == 303
    stored = api.container.paper_collections.list_collections().collections
    assert len(stored) == 1
    assert api.container.paper_collections.get(stored[0].collection_id).document_ids == expected
    assert set(parse_qs(urlsplit(response.headers["location"]).query)) == {
        "collection_id",
        "collection_revision",
    }


def test_only_explicit_metadata_creation_runs_the_existing_service_once(api: ExplorerAPI) -> None:
    api.seed("selected")
    before = api.path.read_bytes()
    with browsing_guards(api.container) as read_guards:
        api.page(ENDPOINT, referrer_policy="same-origin")
    assert api.path.read_bytes() == before
    assert not any(guard.call_count for guard in read_guards.values())
    with (
        browsing_guards(api.container, allow_collection_create=True) as guards,
        patch.object(
            api.container.paper_collections, "create", wraps=api.container.paper_collections.create
        ) as create,
        patch.object(api.container.paper_screening, "submit", side_effect=no_work),
        patch.object(api.container.worksheets, "build", side_effect=no_work),
        patch.object(api.container.event_log, "append_transition", side_effect=no_work),
    ):
        response = api.client.post(
            ENDPOINT,
            data={"name": "Explicit metadata", "document_1": '"selected"'},
            headers=ORIGIN,
            follow_redirects=False,
        )
        assert response.status_code == 303
        create.assert_called_once_with(name="Explicit metadata", document_ids=("selected",))
        assert not any(guard.call_count for guard in guards.values())
    assert api.container.event_log.list_events() == []


def test_duplicate_name_and_deleted_selection_never_partially_save(api: ExplorerAPI) -> None:
    api.seed("known", ())
    api.seed("later-deleted", ())
    initial = api.client.post("/collections", json={"name": "Existing", "document_ids": ["known"]})
    assert initial.status_code == 201
    assert initial.headers["location"] == "/collections/" + initial.json()["collection_id"]
    opened = api.page(ENDPOINT, referrer_policy="same-origin")
    with closing(sqlite3.connect(api.path)) as connection, connection:
        connection.execute("DELETE FROM documents WHERE document_id = ?", ("later-deleted",))
    before = api.path.read_bytes()
    for fields, status in (
        ({"name": " Existing ", "document_1": '"known"'}, 409),
        ({"name": "Deleted member", **choices(opened)}, 422),
    ):
        failure(api.client.post(ENDPOINT, data=fields, headers=ORIGIN), status)
        assert api.path.read_bytes() == before
    listed = api.container.paper_collections.list_collections().collections
    assert len(listed) == 1 and listed[0].name == "Existing"


def test_concurrent_same_name_submissions_save_only_one_selection(api: ExplorerAPI) -> None:
    api.seed("known", ())

    def submit(_: int) -> int:
        return api.client.post(
            ENDPOINT,
            data={"name": "One winner", "document_1": '"known"'},
            headers=ORIGIN,
            follow_redirects=False,
        ).status_code

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(submit, range(2))) == [303, 409]
    saved = api.container.paper_collections.list_collections().collections
    assert len(saved) == 1 and saved[0].revision == 1 and saved[0].document_count == 1


@pytest.mark.parametrize(
    "fields",
    [
        {"document_1": '"known"'},
        {"name": "", "document_1": '"known"'},
        {"name": "x" * 121, "document_1": '"known"'},
        {"name": "PRIVATE_\0name", "document_1": '"known"'},
        {"name": "PRIVATE_\nname", "document_1": '"known"'},
        {"name": "Native", "document_1": '"known"', "document_2": '"known"'},
        {"name": "Native", "document_0": '"known"'},
        {"name": "Native", "document_01": '"known"'},
        {"name": "Native", "document_101": '"known"'},
        {"name": "Native", "document_1": '" known "'},
        {"name": "Native", "document_1": json.dumps("x" * 129)},
        {"name": "Native", "document_1": '["known"]'},
        {"name": "Native", "document_1": "42"},
        {"name": "Native", "document_1": "null"},
        {"name": "Native", "document_1": '"\\ud800"'},
        {"name": "Native", "document_1": '"known"', "return_to": "//external.invalid/PRIVATE_"},
    ],
)
def test_invalid_controls_do_not_repair_or_write(api: ExplorerAPI, fields: dict[str, str]) -> None:
    api.seed("known")
    before = api.path.read_bytes()
    failure(api.client.post(ENDPOINT, data=fields, headers=ORIGIN), 422)
    assert api.path.read_bytes() == before


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"origin": "null"},
        {"origin": "https://testserver"},
        {"origin": "http://testserver:81"},
        {"origin": "http://testserver.evil.invalid"},
        {"origin": "http://testserver", "sec-fetch-site": "same-site"},
        {"origin": "http://testserver", "sec-fetch-site": "cross-site"},
        {"referer": "http://testserver/explore/collections/new"},
        [("origin", "http://testserver"), ("origin", "http://testserver")],
        [("origin", "http://testserver"), ("host", "testserver"), ("host", "testserver")],
        [
            ("origin", "http://testserver"),
            ("sec-fetch-site", "same-origin"),
            ("sec-fetch-site", "same-origin"),
        ],
    ],
)
def test_exact_origin_is_required_before_any_native_write(
    api: ExplorerAPI, headers: dict[str, str] | list[tuple[str, str]]
) -> None:
    api.seed("known")
    before = api.path.read_bytes()
    failure(
        api.client.post(
            ENDPOINT, data={"name": "No save", "document_1": '"known"'}, headers=headers
        ),
        403,
    )
    assert api.path.read_bytes() == before


def test_wire_encoding_unknown_duplicate_and_malformed_fields_fail_closed(
    api: ExplorerAPI,
) -> None:
    api.seed("known")
    before = api.path.read_bytes()
    raw = urlencode({"name": "Native", "document_1": '"known"'})
    headers = {**ORIGIN, "content-type": ENCODING}
    for suffix in (
        "&name=PRIVATE_duplicate",
        "&document_1=%22known%22",
        "&x=%",
        "&x=%Q0",
        "&x=%ff",
        "&x=%ED%A0%80",
        "&x",
        "&",
        "&=x",
        "&" + "k" * 33 + "=value",
        "&x=" + "a" * (collection_creation.MAX_FIELD_CHARACTERS + 1),
    ):
        failure(api.client.post(ENDPOINT, content=(raw + suffix).encode(), headers=headers), 422)
    failure(api.client.post(ENDPOINT, content=raw.encode() + b"&x=\xff", headers=headers), 422)
    for encoding in (
        "application/json",
        "multipart/form-data; boundary=fixture",
        "application/x-www-form-urlencoded; charset=latin-1",
    ):
        failure(
            api.client.post(ENDPOINT, content=raw, headers={**ORIGIN, "content-type": encoding}),
            415,
        )
    failure(
        api.client.post(ENDPOINT, content=raw, headers={**headers, "content-encoding": "gzip"}), 415
    )
    failure(
        api.client.post(
            ENDPOINT, content=raw, headers=[*headers.items(), ("content-type", ENCODING)]
        ),
        415,
    )
    assert api.path.read_bytes() == before


def test_query_body_method_and_length_boundaries(api: ExplorerAPI) -> None:
    api.seed("known")
    before = api.path.read_bytes()
    raw = urlencode({"name": "Native", "document_1": '"known"'}).encode()
    headers = {**ORIGIN, "content-type": ENCODING}
    for query in (
        "limit=0",
        "limit=101",
        "limit=true",
        "limit=1.0",
        "limit=" + "1" * 20,
        "source=" + "a" * 513,
        "title=" + "a" * 301,
        "cursor=",
        "name=PRIVATE_",
        "limit=1&limit=2",
        "source=%Q0",
        "source=%ff",
    ):
        failure(api.client.get(ENDPOINT + "?" + query), 422)
    for length in ("0", "not-numeric", "9" * 20, str(len(raw) + 1)):
        failure(
            api.client.post(ENDPOINT, content=raw, headers={**headers, "content-length": length}),
            422,
        )
    failure(
        api.client.post(
            ENDPOINT,
            content=raw,
            headers=[
                *headers.items(),
                ("content-length", str(len(raw))),
                ("content-length", str(len(raw))),
            ],
        ),
        422,
    )
    failure(api.client.request("GET", ENDPOINT, content=b"name=not-a-write"), 422)
    failure(api.client.post(ENDPOINT + "?name=PRIVATE_", content=raw, headers=headers), 422)
    for size, status in ((16_384, 422), (16_385, 413)):
        failure(api.client.get(ENDPOINT + "?" + "x" * size), status)
    failure(api.client.post(ENDPOINT, content=b"x" * 262_145, headers=headers), 413)
    failure(api.client.post(ENDPOINT, content=iter([b"x" * 131_073] * 2), headers=headers), 413)
    for method in ("PUT", "DELETE", "OPTIONS"):
        response = api.client.request(method, ENDPOINT)
        failure(response, 405)
        assert response.headers["allow"] == "GET, POST"
    assert api.client.head(ENDPOINT).status_code == 405
    assert api.path.read_bytes() == before


def test_maximum_selection_name_and_measured_html_and_body_bounds(
    api: ExplorerAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    identifiers = tuple(f"{index:03d}" + "\U0001f52c" * 125 for index in range(100))
    for identifier in identifiers:
        api.seed(identifier, ())
    response = api.client.get(ENDPOINT, params={"limit": 100})
    assert response.status_code == 200
    root = ExplorerDOM(response.text).root
    fields = {"name": "\U0001f52c" * 120, **choices(root)}
    assert len(fields) == 101
    raw = urlencode(fields).encode()
    assert len(raw) < collection_creation.MAX_FORM_BYTES
    before = api.path.read_bytes()
    monkeypatch.setattr(explorer, "MAX_HTML_BYTES", len(response.content))
    assert api.client.get(ENDPOINT, params={"limit": 100}).status_code == 200
    monkeypatch.setattr(explorer, "MAX_HTML_BYTES", len(response.content) - 1)
    failure(api.client.get(ENDPOINT, params={"limit": 100}), 413)
    assert api.path.read_bytes() == before
    monkeypatch.setattr(collection_creation, "MAX_FORM_BYTES", len(raw) - 1)
    failure(
        api.client.post(ENDPOINT, content=raw, headers={**ORIGIN, "content-type": ENCODING}), 413
    )
    assert api.path.read_bytes() == before
    monkeypatch.setattr(collection_creation, "MAX_FORM_BYTES", len(raw))
    saved = api.client.post(
        ENDPOINT,
        content=raw,
        headers={"origin": "http://testserver", "content-type": ENCODING + '; charset="UTF-8"'},
        follow_redirects=False,
    )
    assert saved.status_code == 303
    collection = api.container.paper_collections.list_collections().collections[0]
    assert collection.name == fields["name"]
    assert api.container.paper_collections.get(collection.collection_id).document_ids == identifiers


def test_decoded_field_limit_rejects_an_otherwise_valid_json_identity(api: ExplorerAPI) -> None:
    api.seed("known", ())
    maximum = '"known"'.ljust(collection_creation.MAX_FIELD_CHARACTERS)
    before = api.path.read_bytes()
    failure(
        api.client.post(
            ENDPOINT,
            data={"name": "Field boundary", "document_1": maximum + " "},
            headers=ORIGIN,
        ),
        422,
    )
    assert api.path.read_bytes() == before
    response = api.client.post(
        ENDPOINT,
        data={"name": "Field boundary", "document_1": maximum},
        headers=ORIGIN,
        follow_redirects=False,
    )
    assert response.status_code == 303
    record = api.container.paper_collections.list_collections().collections[0]
    assert api.container.paper_collections.get(record.collection_id).document_ids == ("known",)


def test_redirect_uses_the_service_returned_persisted_revision(api: ExplorerAPI) -> None:
    api.seed("known", ())
    store = api.container.paper_collections
    saved = store.create(name="Service result", document_ids=["known"])
    revised = store.replace(
        saved.collection_id, name=saved.name, document_ids=saved.document_ids, expected_revision=1
    )
    with patch.object(store, "create", return_value=revised):
        response = api.client.post(
            ENDPOINT,
            data={"name": revised.name, "document_1": '"known"'},
            headers=ORIGIN,
            follow_redirects=False,
        )
    assert response.status_code == 303
    params = parse_qs(urlsplit(response.headers["location"]).query)
    assert params == {"collection_id": [revised.collection_id], "collection_revision": ["2"]}
    assert api.client.get(response.headers["location"]).status_code == 200


@pytest.mark.parametrize("prefix,mounted", [("/library", False), ("/prefix/library", True)])
def test_proxy_and_mounted_forms_redirects_and_workflow_links_stay_local(
    api: ExplorerAPI, prefix: str, mounted: bool
) -> None:
    api.seed("//external.invalid/paper\r\nid\0")
    application = api.app
    if mounted:
        application = FastAPI()
        application.mount("/library", api.app)
    with TestClient(application, root_path="/prefix" if mounted else prefix) as client:
        catalog = ExplorerDOM(client.get(prefix + "/explore").text).root
        opened = client.get(href(catalog, "create-collection"))
        assert opened.status_code == 200
        form = ExplorerDOM(opened.text).root
        response = client.post(
            form.by_id("collection-create").attrs["action"],
            data={"name": "Mounted selection", **choices(form)},
            headers=ORIGIN,
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert response.headers["location"].startswith(prefix + "/explore/collection?")
        saved = client.get(response.headers["location"])
        assert saved.status_code == 200
        detail = ExplorerDOM(saved.text).root
        for root in (form, detail):
            for link in root.by_tag("a"):
                parts = urlsplit(link.attrs["href"])
                assert not parts.netloc and not parts.scheme
                assert not parts.path or parts.path.startswith(prefix + "/")
            for control in root.by_tag("form"):
                assert control.attrs["action"].startswith(prefix + "/")
        for link in ("screen-member-1", "search-collection", "worksheet-collection"):
            assert client.get(href(detail, link)).status_code == 200


def test_storage_errors_are_sanitized_and_unrepresentable_labels_show_no_partial_form(
    api: ExplorerAPI, caplog: pytest.LogCaptureFixture
) -> None:
    api.seed("known")
    before = api.path.read_bytes()
    for target, method, error in (
        (
            api.container.document_catalog,
            "list_documents",
            sqlite3.OperationalError("PRIVATE_path"),
        ),
        (
            api.container.paper_collections,
            "create",
            CollectionError("collection_storage_error", "PRIVATE_database", 503),
        ),
    ):
        with patch.object(target, method, side_effect=error):
            response = (
                api.client.get(ENDPOINT)
                if method == "list_documents"
                else api.client.post(
                    ENDPOINT, data={"name": "PRIVATE_name", "document_1": '"known"'}, headers=ORIGIN
                )
            )
            failure(response, 503)
            assert api.path.read_bytes() == before
    assert "collection_storage_error" in caplog.text and "PRIVATE_" not in caplog.text
    api.seed("bad-label", (), title="PRIVATE_\0invalid")
    before = api.path.read_bytes()
    failure(api.client.get(ENDPOINT), 409)
    assert api.path.read_bytes() == before
    with closing(sqlite3.connect(api.path)) as connection, connection:
        connection.execute("UPDATE documents SET title = X'FF' WHERE document_id = 'bad-label'")
    failure(api.client.get(ENDPOINT), 409)
