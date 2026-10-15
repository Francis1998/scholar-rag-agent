"""Native collection discovery and human-screening inspection over existing read services."""

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

from api import explorer
from api.application import create_app
from storage import paper_screening
from storage.paper_collections import MAX_REVISION, CollectionError, PaperCollection
from storage.paper_screening import ScreeningExport, ScreeningSubmission
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
from tests.test_worksheet_explorer import controls as worksheet_controls

LIST = "/explore/collections"
DETAIL = "/explore/collection"


def seed(api: ExplorerAPI) -> PaperCollection:
    identifiers = ("z-methods", "a-background", "n-unsure", "b-stale", "m-unread")
    for identifier in identifiers:
        api.seed(identifier, (identifier + "/chunk",))
    collection = api.container.paper_collections.create(
        name="Synthetic review <draft>", document_ids=identifiers
    )
    api.container.paper_screening.submit(
        collection.collection_id,
        "b-stale",
        ScreeningSubmission(
            collection_revision=1,
            expected_decision_revision=0,
            decision="include",
            reason="Earlier human opinion.",
        ),
    )
    collection = api.container.paper_collections.replace(
        collection.collection_id,
        name=collection.name,
        document_ids=identifiers,
        expected_revision=1,
    )
    for identifier, decision in zip(identifiers, ("include", "exclude", "unsure"), strict=False):
        api.container.paper_screening.submit(
            collection.collection_id,
            identifier,
            ScreeningSubmission.model_validate(
                {
                    "collection_revision": 2,
                    "expected_decision_revision": 0,
                    "decision": decision,
                    "reason": f"Human {decision} reason.",
                }
            ),
        )
    return collection


def test_empty_collection_browser_is_available_without_creating_a_collection(
    api: ExplorerAPI,
) -> None:
    before = api.path.read_bytes()
    response = api.client.get(LIST)
    assert response.status_code == 200
    assert "No saved collections" in response.text
    assert api.path.read_bytes() == before


def test_collection_detail_uses_existing_screening_snapshot(
    api: ExplorerAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    collection = seed(api)
    expected = ScreeningExport.model_validate_json(
        api.container.paper_screening.export_results(
            collection.collection_id, collection_revision=2
        ).content
    )
    export = Mock(wraps=api.container.paper_screening.export_results)
    monkeypatch.setattr(api.container.paper_screening, "export_results", export)
    root = api.page(DETAIL, collection_id=collection.collection_id, collection_revision=2)
    export.assert_called_once_with(collection.collection_id, collection_revision=2, format="json")
    assert identities(root) == list(collection.document_ids)
    assert values(root, "member-status") == [item.status for item in expected.items]
    assert values(root, "review-reason") == [
        item.review.reason for item in expected.items if item.review
    ]
    for status, count in expected.counts.model_dump().items():
        assert root.by_id(f"count-{status}").text() == str(count)
    assert identities(root, "included-document-id") == ["z-methods"]
    assert "not screened-in papers" in root.text()
    assert "Separate requests are not one snapshot" in root.text()
    assert "Earlier decision (stale)" in root.text()
    for output, path, format in (
        ("screening-json", "screening/export", "json"),
        ("screening-csv", "screening/export", "csv"),
        ("duplicates-json", "duplicates", "json"),
        ("duplicates-markdown", "duplicates", "markdown"),
    ):
        url = urlsplit(href(root, f"download-{output}"))
        assert url.path == f"/collections/{collection.collection_id}/{path}"
        assert parse_qs(url.query) == {"collection_revision": ["2"], "format": [format]}


def test_catalog_discovers_collections_without_calling_members_screened(
    api: ExplorerAPI,
) -> None:
    collection = seed(api)
    catalog = api.page(limit=1, title="Synthetic", source="synthetic:explorer", cursor="a")
    listed = api.page(href(catalog, "saved-collections"))
    assert identities(listed, "collection-id") == [collection.collection_id]
    assert values(listed, "collection-name") == [collection.name]
    assert values(listed, "member-count") == ["5"]
    assert "Membership counts are not screening decisions" in listed.text()
    detail = api.page(href(listed, "inspect-collection-1"))
    assert href(detail, "back-to-collections") == href(catalog, "saved-collections")
    assert json.loads(values(detail, "collection-id")[0]) == collection.collection_id


def changed(url: str, **updates: str | int | None) -> str:
    parts = urlsplit(url)
    params = {key: value[0] for key, value in parse_qs(parts.query).items()}
    for key, value in updates.items():
        if value is None:
            params.pop(key, None)
        else:
            params[key] = str(value)
    return parts.path + "?" + urlencode(params)


def controls(root: Node, form_id: str) -> dict[str, str]:
    form = root.by_id(form_id)
    params = {node.attrs["name"]: node.attrs.get("value", "") for node in form.by_tag("input")}
    for select in form.by_tag("select"):
        params[select.attrs["name"]] = next(
            node.attrs["value"] for node in select.by_tag("option") if "selected" in node.attrs
        )
    return params


def failure(response: httpx.Response, status: int) -> Node:
    assert response.status_code == status, response.text[:1000]
    assert_headers(response)
    root = ExplorerDOM(response.text).root
    assert not root.by_tag("article") and not root.by_class("member-status")
    assert "PRIVATE_" not in response.text
    assert "content-disposition" not in response.headers
    return root


def detail_url(collection: PaperCollection, **params: str | int) -> str:
    return changed(
        DETAIL,
        collection_id=collection.collection_id,
        collection_revision=collection.revision,
        **params,
    )


def test_list_paging_detail_filtering_and_return_state(api: ExplorerAPI) -> None:
    collection = seed(api)
    api.container.paper_collections.create(name="Other synthetic", document_ids=["m-unread"])
    catalog_url = "/explore?limit=1&source=synthetic%3Aexplorer&title=Synthetic&cursor=a"
    catalog = api.page(catalog_url)
    first = api.page(changed(href(catalog, "saved-collections"), limit=1))
    second_url = href(first, "next-page")
    second = api.page(second_url)
    assert not second.by_class("next-page")
    assert href(second, "back-to-catalog") == catalog_url
    assert len(set(identities(first, "collection-id") + identities(second, "collection-id"))) == 2
    opened = api.page(href(second, "inspect-collection-1"))
    assert href(opened, "back-to-collections") == second_url
    assert href(opened, "back-to-catalog") == catalog_url
    first = api.page(detail_url(collection, limit=2))
    assert identities(first) == ["z-methods", "a-background"]
    second = api.page(href(first, "next-page"))
    assert identities(second) == ["n-unsure", "b-stale"]
    third = api.page(href(second, "next-page"))
    assert identities(third) == ["m-unread"] and not third.by_class("next-page")
    form = third.by_id("member-filters")
    assert form.attrs["method"] == "get"
    fields = controls(third, "member-filters")
    assert "cursor" not in fields
    filtered = api.page(form.attrs["action"], **{**fields, "status": "stale", "limit": "1"})
    assert identities(filtered) == ["b-stale"]
    assert values(filtered, "member-status") == ["stale"]
    assert filtered.by_id("count-include").text() == "1"
    assert identities(api.page(href(filtered, "status-all"))) == ["z-methods"]
    labels = {node.attrs["for"] for node in form.by_tag("label")}
    for node in (*form.by_tag("input"), *form.by_tag("select")):
        if node.attrs.get("type") != "hidden":
            assert node.attrs["id"] in labels


@pytest.mark.parametrize(
    "identifier",
    [
        "../paper/?draft#v1",
        "//external.invalid/../x",
        "quotes\"'?%#&+\\",
        "cafe\u0301/\u202eRTL\u202c-\u7814",
        "line\r\nid",
        "nul\0id",
        "\U0001f52c" * 128,
    ],
)
def test_exact_member_paper_source_search_and_worksheet_return(
    api: ExplorerAPI, identifier: str
) -> None:
    api.seed(identifier, ("a", "b"), text="Synthetic literal phrase.")
    collection = api.container.paper_collections.create(
        name="Literal synthetic", document_ids=[identifier]
    )
    opened = api.page(detail_url(collection, limit=1, catalog_limit=1, source="synthetic:explorer"))
    canonical = href(api.page(href(opened, "inspect-member-1")), "back-to-collection")
    paper = api.page(href(opened, "inspect-member-1"))
    assert identities(paper) == [identifier]
    later = api.page(href(paper, "next-page"))
    assert href(later, "back-to-collection") == canonical
    context = api.page(href(later, "source-context-1"))
    assert identities(context) == [identifier]
    assert identities(context, "chunk-id") == ["a", "b"]
    assert href(context, "back-to-collection") == canonical
    assert href(api.page(href(context, "maximum-window")), "back-to-collection") == canonical
    assert href(api.page(href(context, "back-to-passages")), "back-to-collection") == canonical
    search = api.page(href(opened, "search-collection"))
    assert href(search, "back-to-collection") == canonical
    results = api.page(
        search.by_id("passage-search").attrs["action"],
        **{**controls(search, "passage-search"), "query": "literal"},
    )
    assert identities(results) == [identifier, identifier]
    selected = api.page(href(results, "source-context-1"))
    assert href(selected, "back-to-collection") == canonical
    assert href(api.page(href(selected, "back-to-search")), "back-to-collection") == canonical
    worksheet = api.page(href(opened, "worksheet-collection"))
    assert worksheet_controls(worksheet)["collection_id"] == collection.collection_id
    assert worksheet_controls(worksheet)["document_ids"] == ""
    assert href(worksheet, "back-to-collection") == canonical
    # Invalid submission never builds a worksheet, but retains validated return state.
    invalid = api.client.post(
        "/explore/worksheet", data={**worksheet_controls(worksheet), "questions": ""}
    )
    assert href(failure(invalid, 422), "back-to-collection") == canonical
    prefilled = api.client.get(href(paper, "worksheet-document"))
    if "\0" in identifier:
        failure(prefilled, 422)
    else:
        assert prefilled.status_code == 200
        assert href(ExplorerDOM(prefilled.text).root, "back-to-collection") == canonical


@pytest.mark.parametrize("prefix,mounted", [("/library", False), ("/prefix/library", True)])
def test_mounts_downloads_and_error_navigation(
    api: ExplorerAPI, prefix: str, mounted: bool
) -> None:
    seed(api)
    app = api.app
    root_path = prefix
    if mounted:
        app = FastAPI()
        app.mount("/library", api.app)
        root_path = "/prefix"
    with TestClient(app, root_path=root_path) as client:
        catalog = ExplorerDOM(client.get(prefix + "/explore").text).root
        listed = ExplorerDOM(client.get(href(catalog, "saved-collections")).text).root
        detail = ExplorerDOM(client.get(href(listed, "inspect-collection-1")).text).root
        for output in ("screening-json", "screening-csv", "duplicates-json", "duplicates-markdown"):
            downloaded = client.get(href(detail, f"download-{output}"))
            assert downloaded.status_code == 200
            assert downloaded.headers["content-disposition"].startswith("attachment;")
            assert downloaded.headers["cache-control"] == "no-store"
        invalid = failure(client.get(prefix + DETAIL), 422)
        for root in (catalog, listed, detail, invalid):
            for node in root.by_tag("a"):
                parts = urlsplit(node.attrs["href"])
                assert not parts.scheme and not parts.netloc
                assert not parts.path or parts.path.startswith(prefix + "/")
            for form in root.by_tag("form"):
                assert form.attrs["action"].startswith(prefix + "/")


def test_browsing_is_read_only_and_restarts_with_identical_output(
    api: ExplorerAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    collection = seed(api)
    list_url = changed(LIST, limit=1)
    selected_url = detail_url(collection, limit=2)
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
                (container.event_log, "append_transition"),
                (container.paper_screening, "submit"),
                (container.worksheets, "build"),
            ):
                patch.setattr(target, method, no_work)
            for url in (list_url, selected_url):
                response = client.get(url)
                assert response.status_code == 200
                assert outputs.setdefault(url, response.content) == response.content
            root = ExplorerDOM(outputs[selected_url].decode()).root
            for output, format in (("screening-json", "json"), ("screening-csv", "csv")):
                response = client.get(href(root, f"download-{output}"))
                expected = container.paper_screening.export_results(
                    collection.collection_id, collection_revision=2, format=format
                )
                assert response.content == expected.content
                assert outputs.setdefault(output, response.content) == response.content
            report = container.collection_duplicates.inspect(
                collection.collection_id, collection_revision=2
            )
            for output, expected_content in (
                ("duplicates-json", report.to_json()),
                ("duplicates-markdown", report.to_markdown()),
            ):
                response = client.get(href(root, f"download-{output}"))
                assert response.text == expected_content
                assert outputs.setdefault(output, response.content) == response.content
            client.get(href(root, "worksheet-collection")).raise_for_status()
            client.get(href(root, "search-collection")).raise_for_status()
            assert not any(guard.call_count for guard in guards.values())
        assert container.event_log.list_events() == []
        assert api.path.read_bytes() == before


def test_status_changes_do_not_claim_a_frozen_revision(api: ExplorerAPI) -> None:
    collection = seed(api)
    initial = api.page(detail_url(collection))
    api.container.paper_screening.submit(
        collection.collection_id,
        "m-unread",
        ScreeningSubmission(
            collection_revision=2,
            expected_decision_revision=0,
            decision="exclude",
            reason="Later opinion at the same collection revision.",
        ),
    )
    download = api.client.get(href(initial, "download-screening-json")).json()
    assert download["collection_revision"] == 2
    assert download["counts"]["exclude"] == 2
    assert initial.by_id("count-exclude").text() == "1"
    api.container.paper_collections.replace(
        collection.collection_id,
        name="New name",
        document_ids=collection.document_ids,
        expected_revision=2,
    )
    failure(api.client.get(detail_url(collection)), 409)
    for output in ("screening-json", "screening-csv", "duplicates-json", "duplicates-markdown"):
        response = api.client.get(href(initial, f"download-{output}"))
        assert response.status_code == 409
        assert response.json()["detail"]["code"] == "collection_revision_conflict"


@pytest.mark.parametrize(
    "suffix",
    [
        "?limit=0",
        "?limit=101",
        "?limit=1.0",
        "?limit=true",
        "?cursor=",
        "?cursor=PRIVATE_BAD_ID",
        "?limit=1&limit=2",
        "?unknown=PRIVATE_VALUE",
        "?source=%ff",
        "?source=%ED%A0%80",
        "?source=" + "x" * 16_384,
    ],
)
def test_list_rejects_invalid_ambiguous_or_oversized_queries(api: ExplorerAPI, suffix: str) -> None:
    failure(api.client.get(LIST + suffix), 422)


@pytest.mark.parametrize(
    "updates",
    [
        {"collection_id": "PRIVATE_BAD"},
        {"collection_revision": None},
        {"collection_revision": 0},
        {"collection_revision": MAX_REVISION + 1},
        {"collection_revision": "2.0"},
        {"collection_revision": "true"},
        {"status": "INCLUDE"},
        {"status": "PRIVATE_UNKNOWN"},
        {"cursor": ""},
        {"cursor": "PRIVATE_NONMEMBER"},
        {"limit": 101},
        {"collections_cursor": "PRIVATE_BAD"},
        {"catalog_limit": 0},
        {"unknown": "PRIVATE_VALUE"},
    ],
)
def test_detail_rejects_invalid_selection_without_partial_page(
    api: ExplorerAPI, updates: dict[str, str | int | None]
) -> None:
    collection = seed(api)
    failure(api.client.get(changed(detail_url(collection), **updates)), 422)


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE", "HEAD"])
@pytest.mark.parametrize("endpoint", [LIST, DETAIL])
def test_browser_rejects_writes_and_has_private_method_errors(
    api: ExplorerAPI, endpoint: str, method: str
) -> None:
    before = api.path.read_bytes()
    response = api.client.request(method, endpoint)
    assert response.status_code == 405
    assert response.headers["allow"] == "GET"
    assert response.headers["cache-control"] == "no-store"
    assert "script-src 'none'" in response.headers["content-security-policy"]
    assert api.path.read_bytes() == before


def test_missing_corrupt_and_unavailable_storage_are_explicit(
    api: ExplorerAPI, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    collection = seed(api)
    failure(
        api.client.get(
            DETAIL, params={"collection_id": "col_" + "0" * 32, "collection_revision": 1}
        ),
        404,
    )
    with closing(sqlite3.connect(api.path)) as connection, connection:
        connection.execute("UPDATE paper_screening_decisions SET record = ?", ("PRIVATE_INVALID",))
    failure(api.client.get(detail_url(collection, status="unscreened", limit=1)), 409)
    with closing(sqlite3.connect(api.path)) as connection, connection:
        connection.execute("DELETE FROM documents WHERE document_id = ?", ("m-unread",))
    missing = failure(api.client.get(detail_url(collection)), 409)
    assert "missing from the corpus" in missing.text()
    with monkeypatch.context() as patch:
        patch.setattr(
            api.container.paper_collections,
            "list_collections",
            Mock(side_effect=CollectionError("collection_storage_error", "PRIVATE_SQL", 503)),
        )
        failure(api.client.get(LIST), 503)
    assert "PRIVATE_" not in caplog.text
    moved = api.path.with_suffix(".offline")
    api.path.rename(moved)
    failure(api.client.get(LIST), 503)
    assert not api.path.exists()


def test_literal_labels_reasons_bounds_and_unrepresentable_text(
    api: ExplorerAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    api.seed("exact", ("chunk",), title=HOSTILE * 3, source=HOSTILE * 4)
    collection = api.container.paper_collections.create(
        name='Literal <script>& "name"', document_ids=["exact"]
    )
    reason = "\n\r\t" + HOSTILE + "\r\nEnd."
    api.container.paper_screening.submit(
        collection.collection_id,
        "exact",
        ScreeningSubmission(
            collection_revision=1,
            expected_decision_revision=0,
            decision="unsure",
            reason=reason,
        ),
    )
    response = api.client.get(detail_url(collection))
    root = ExplorerDOM(response.text).root
    assert values(root, "review-reason") == [reason]
    assert values(root, "member-title") == [(HOSTILE * 3)[:300]]
    assert values(root, "member-source") == [(HOSTILE * 4)[:512]]
    assert "Title truncated" in root.text() and "Source truncated" in root.text()
    for tag in ("script", "iframe", "img", "base", "svg", "object"):
        assert not root.by_tag(tag)
    for node in root.walk():
        assert not any(key.startswith("on") or key == "style" for key in node.attrs)
    size = len(response.content)
    with monkeypatch.context() as patch:
        patch.setattr(explorer, "MAX_HTML_BYTES", size)
        assert api.client.get(detail_url(collection)).content == response.content
        patch.setattr(explorer, "MAX_HTML_BYTES", size - 1)
        failure(api.client.get(detail_url(collection)), 413)
    snapshot_bytes = len(
        api.container.paper_screening.export_results(
            collection.collection_id, collection_revision=1
        ).content
    )
    with monkeypatch.context() as patch:
        patch.setattr(paper_screening, "MAX_EXPORT_BYTES", snapshot_bytes - 1)
        failure(api.client.get(detail_url(collection, limit=1, status="include")), 413)
    with closing(sqlite3.connect(api.path)) as connection, connection:
        connection.execute("UPDATE documents SET title = ?", ("PRIVATE_\0label",))
    failure(api.client.get(detail_url(collection)), 409)


@pytest.mark.parametrize(
    "token",
    [
        "{}",
        "null",
        "[]",
        '{"return_url":"https://external.invalid/PRIVATE_REDIRECT"}',
        '{"collection_id":"col_' + "0" * 32 + '","collection_revision":1,"limit":1,"limit":2}',
        "[" * 1500,
    ],
)
@pytest.mark.parametrize("endpoint", ["/explore/document", "/explore/search", "/explore/worksheet"])
def test_invalid_return_state_cannot_redirect_or_be_silently_ignored(
    api: ExplorerAPI, token: str, endpoint: str
) -> None:
    params = {"collection_return": token}
    if endpoint == "/explore/document":
        params["document_id"] = "exact"
    failure(api.client.get(endpoint, params=params), 422)


def test_maximum_unicode_navigation_and_native_controls_round_trip(
    api: ExplorerAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    identifier = "\U0001f52c" * 128
    prior = "\U0001f4da" * 128
    chunk = "\U0001f4da" * 256
    source = "\U0001f52c" * 512
    title = "\U0001f52c" * 300
    for document_id in (prior, identifier):
        api.seed(
            document_id,
            (chunk if document_id == identifier else "\U0001f9ea" * 256,),
            source=source,
            title=title,
            text="literal",
        )
    collection = api.container.paper_collections.create(
        name="Maximum Unicode navigation", document_ids=[prior, identifier]
    )
    url = detail_url(
        collection,
        limit=100,
        cursor=prior,
        status="unscreened",
        collections_limit=100,
        collections_cursor="col_" + "0" * 32,
        catalog_limit=100,
        catalog_cursor=prior,
        source=source,
        title=title,
    )
    root = api.page(url)
    member_url = href(root, "inspect-member-1")
    assert 16_384 < len(urlsplit(member_url).query.encode()) < 65_536
    paper = api.page(member_url)
    canonical = href(paper, "back-to-collection")
    assert parse_qs(urlsplit(canonical).query) == parse_qs(urlsplit(url).query)
    context_url = href(paper, "source-context-1")
    context = api.page(context_url)
    assert href(context, "back-to-collection") == canonical
    window = api.page(
        context.by_id("context-window").attrs["action"],
        **controls(context, "context-window"),
    )
    assert href(window, "back-to-collection") == canonical
    for anchor in root.by_tag("a"):
        if urlsplit(anchor.attrs["href"]).path:
            api.client.get(anchor.attrs["href"]).raise_for_status()
    search = api.page(href(root, "search-collection"))
    matches = api.page(
        search.by_id("passage-search").attrs["action"],
        **{**controls(search, "passage-search"), "query": "literal"},
    )
    assert href(matches, "back-to-collection") == canonical
    assert href(api.page(href(matches, "source-context-1")), "back-to-collection") == canonical
    worksheet = api.page(href(root, "worksheet-collection"))
    submitted = api.client.post(
        "/explore/worksheet",
        data={**worksheet_controls(worksheet), "questions": "literal"},
    )
    assert submitted.status_code == 200, submitted.text
    assert href(ExplorerDOM(submitted.text).root, "back-to-collection") == canonical
    assert href(api.page(href(paper, "worksheet-document")), "back-to-collection") == canonical
    query_bytes = len(urlsplit(context_url).query.encode())
    monkeypatch.setattr(explorer, "MAX_COLLECTION_QUERY_BYTES", query_bytes)
    api.page(context_url)
    monkeypatch.setattr(explorer, "MAX_COLLECTION_QUERY_BYTES", query_bytes - 1)
    failure(api.client.get(context_url), 422)
