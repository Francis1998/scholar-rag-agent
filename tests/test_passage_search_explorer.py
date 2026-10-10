"""Native browser search-to-source acceptance over the existing SQLite readers."""

import json
import sqlite3
from contextlib import closing
from unittest.mock import Mock
from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi.testclient import TestClient
from scripts.demo_corpus_explorer import browsing_guards
from scripts.demo_evidence_export import offline_settings

from api.application import create_app
from storage.literal_search import MAX_TEXT_BYTES, LiteralSearchError, LiteralSearchRequest
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


def test_native_phrase_search_paginate_read_context_and_return(api: ExplorerAPI) -> None:
    document_id = "paper/../\u7814?draft#v1"
    phrase = "\U0001f52c cafe\u0301"
    text = "Synthetic padding. " * 60 + phrase + " then " + phrase
    api.seed(document_id, ("chunk-2", "chunk-10", "chunk-9"), text=text)
    api.seed("a-excluded", ("excluded",), text=phrase + " EXCLUDED_EVIDENCE")
    params: dict[str, str | int] = {
        "query": phrase,
        "scope": "document",
        "scope_id": document_id,
        "limit": 1,
        "catalog_limit": 2,
        "source": "synthetic:explorer",
        "title": "graph",
        "catalog_cursor": "a-excluded",
    }
    before = api.path.read_bytes()
    with browsing_guards(api.container) as guards:
        response = api.client.get("/explore/search", params=params)
        assert response.status_code == 200, response.text
        assert_headers(response)
        first = ExplorerDOM(response.text).root
        assert identities(first) == [document_id]
        assert identities(first, "chunk-id") == ["chunk-10"]
        assert [mark.text() for mark in first.by_tag("mark")] == [phrase]
        assert "EXCLUDED_EVIDENCE" not in first.text()
        assert "Unicode code-point" in first.text()
        assert "not relevance" in first.text()
        second_url = href(first, "next-page")
        second = api.page(second_url)
        assert identities(second, "chunk-id") == ["chunk-2"]
        third = api.page(href(second, "next-page"))
        assert identities(third, "chunk-id") == ["chunk-9"]
        assert not third.by_class("next-page")
        assert "cursor" not in parse_qs(urlsplit(href(third, "first-page")).query)
        context = api.page(href(second, "source-context-1"))
        assert identities(context.by_id("context-anchor"), "chunk-id") == ["chunk-2"]
        assert href(context, "back-to-search") == second_url
        recentered = api.page(href(context, "center-context-2"))
        assert href(recentered, "back-to-search") == second_url
        assert href(recentered, "back-to-catalog") == href(first, "back-to-catalog")
        assert identities(api.page(href(recentered, "back-to-search")), "chunk-id") == ["chunk-2"]
        form = second.by_id("passage-search")
        assert form.attrs["method"] == "get"
        assert form.attrs["action"] == "/explore/search"
        assert all(node.attrs["name"] != "cursor" for node in form.by_tag("input"))
        reset = api.page(href(second, "reset-search"))
        assert not reset.by_tag("article")
        reset_inputs = {node.attrs["name"]: node.attrs["value"] for node in reset.by_tag("input")}
        assert reset_inputs["scope_id"] == document_id
        assert reset_inputs["query"] == ""
        assert not any(guard.call_count for guard in guards.values())
    assert api.path.read_bytes() == before
    assert api.container.event_log.list_events() == []


@pytest.mark.parametrize("missing", ["search_query", "search_scope", "search_limit"])
def test_partial_search_return_state_fails_before_source_read(
    api: ExplorerAPI, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    params = {
        "document_id": "selected",
        "chunk_id": "a",
        "search_query": "PRIVATE_PHRASE",
        "search_scope": "all",
        "search_limit": "1",
    }
    del params[missing]
    monkeypatch.setattr(api.container.source_context, "read", no_work)
    response = api.client.get("/explore/context", params=params)
    assert response.status_code == 422
    assert_headers(response)
    assert "Invalid passage search request" in response.text
    assert "PRIVATE_PHRASE" not in response.text
    assert not ExplorerDOM(response.text).root.by_tag("article")


@pytest.mark.parametrize("encoded", ["%FF", "%C0%AF", "%ED%A0%80"])
def test_invalid_utf8_query_is_not_replaced_before_search(
    api: ExplorerAPI, monkeypatch: pytest.MonkeyPatch, encoded: str
) -> None:
    monkeypatch.setattr(api.container.literal_search, "search", no_work)
    response = api.client.get("/explore/search?query=" + encoded)
    assert response.status_code == 422
    assert_headers(response)
    assert "Invalid passage search request" in response.text


def test_catalog_entries_and_native_submission_keep_exact_scope_and_return_state(
    api: ExplorerAPI,
) -> None:
    api.seed("a", ("a1",), source="excluded", text="exact phrase")
    api.seed("b", ("b1", "b2"), text="exact phrase")
    catalog = api.page(limit=1, cursor="a", title="graph", source="synthetic:explorer")
    all_form = api.page(href(catalog, "search-passages"))
    assert "Whole current corpus" in all_form.text()
    selected = api.page(href(catalog, "search-document-1"))
    form = selected.by_id("passage-search")
    inputs = {node.attrs["name"]: node for node in form.by_tag("input")}
    params = {name: node.attrs["value"] for name, node in inputs.items()}
    params["query"] = "exact phrase"
    scope = form.by_tag("select")[0]
    params["scope"] = next(
        option.attrs["value"] for option in scope.by_tag("option") if "selected" in option.attrs
    )
    assert params["scope"] == "document" and params["scope_id"] == "b"
    labels = {node.attrs["for"] for node in form.by_tag("label")}
    for node in (*form.by_tag("input"), *form.by_tag("select")):
        if node.attrs.get("type") != "hidden":
            assert node.attrs["id"] in labels
    result = api.page(form.attrs["action"], **params)
    assert identities(result) == ["b", "b"]
    returned = api.page(href(result, "back-to-catalog"))
    assert identities(returned) == ["b"]
    assert parse_qs(urlsplit(href(result, "back-to-catalog")).query) == {
        "limit": ["1"],
        "cursor": ["a"],
        "title": ["graph"],
        "source": ["synthetic:explorer"],
    }
    changed = api.page(href(result, "reset-scope"))
    assert values(changed, "scope-id") == []
    assert parse_qs(urlsplit(href(changed, "back-to-catalog")).query)["cursor"] == ["a"]


def test_form_only_and_clear_phrase_never_scan_storage(
    api: ExplorerAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api.container.literal_search, "search", no_work)
    monkeypatch.setattr(api.container.document_catalog, "list_documents", no_work)
    for params in ({}, {"scope": "document", "scope_id": "missing"}):
        root = api.page("/explore/search", **params)
        assert "No corpus scan has run" in root.text()
        assert root.by_id("passage-search").tag == "form"
        assert not root.by_tag("article")


@pytest.mark.parametrize(
    "phrase",
    [
        "exact phrase",
        "\U0001f52c cafe\u0301",
        "100%_'\"\\",
        "' OR 1=1 --",
        "<mark>&amp;</mark>",
        " phrase ",
        "line\r\nbreak\r",
        "\nfirst line",
        "\u202eRTL\u202c",
    ],
)
@pytest.mark.parametrize("prefix", ["", "\n\r\t\u7814\U0001f52c" * 300])
def test_first_match_highlight_and_unicode_offsets_are_exact_literal_slices(
    api: ExplorerAPI, phrase: str, prefix: str
) -> None:
    text = prefix + phrase + " another occurrence: " + phrase + "z" * 850
    api.seed("paper", ("chunk",), text=text, title=HOSTILE, source=HOSTILE)
    response = api.client.get("/explore/search", params={"query": phrase})
    assert response.status_code == 200
    assert_headers(response)
    root = ExplorerDOM(response.text).root
    result = api.container.literal_search.search(LiteralSearchRequest(query=phrase))
    match = result.matches[0]
    assert match.match_start == text.index(phrase) == len(prefix)
    assert match.match_end == len(prefix) + len(phrase)
    assert values(root, "search-excerpt") == [text[match.excerpt_start : match.excerpt_end]]
    assert [mark.text() for mark in root.by_tag("mark")] == [phrase]
    assert values(root, "chunk-title") == values(root, "chunk-source") == [HOSTILE]
    for css, start, end in (
        ("match-offsets", match.match_start, match.match_end),
        ("excerpt-offsets", match.excerpt_start, match.excerpt_end),
    ):
        node = root.by_class(css)[0]
        assert (int(node.attrs["data-start"]), int(node.attrs["data-end"])) == (start, end)
    assert len(match.excerpt) == 800
    assert "Later chunk text is omitted" in root.text()
    assert ("Earlier chunk text is omitted" in root.text()) == (match.excerpt_start > 0)
    assert not any(root.by_tag(tag) for tag in ("script", "img", "iframe", "object", "base"))
    for node in root.walk():
        assert not any(key.startswith("on") or key == "style" for key in node.attrs)
    if "\r" in phrase or "\n" in phrase:
        assert not root.by_tag("form")
        assert "native text inputs cannot preserve" in root.text()
    assert json.loads(values(root, "search-query")[0]) == phrase
    context = api.page(href(root, "source-context-1"))
    assert identities(context.by_id("context-anchor"), "chunk-id") == ["chunk"]


@pytest.mark.parametrize(
    "phrase", ["Exact phrase", "exact phrase ", "cafe\u0301", "%", "phrase across chunks"]
)
def test_no_results_do_not_expand_case_normalization_or_chunk_boundaries(
    api: ExplorerAPI, phrase: str
) -> None:
    api.seed(text="exact phrase;caf\u00e9")
    root = api.page("/explore/search", query=phrase)
    assert not root.by_tag("article") and not root.by_tag("mark")
    assert "No matching passages in this exact scope" in root.text()
    assert "No scope was widened" in root.text()


def test_empty_unknown_paper_and_exhausted_pages_never_widen(api: ExplorerAPI) -> None:
    assert "No matching passages" in api.page("/explore/search", query="exact phrase").text()
    api.seed("excluded", ("excluded",), text="exact phrase")
    api.seed("empty", ())
    for identifier in ("missing", "empty"):
        root = api.page(
            "/explore/search", query="exact phrase", scope="document", scope_id=identifier
        )
        assert not root.by_tag("article")
        assert json.loads(values(root, "scope-id")[0]) == identifier
    api.seed(text="exact phrase")
    first = api.page(
        "/explore/search", query="exact phrase", scope="document", scope_id="selected", limit=1
    )
    with closing(sqlite3.connect(api.path)) as connection, connection:
        connection.execute("DELETE FROM chunks WHERE document_id = 'selected'")
    exhausted = api.page(href(first, "next-page"))
    assert "No matching passages remain after this cursor" in exhausted.text()
    assert not exhausted.by_tag("article")


def test_collection_pagination_revision_rejection_and_restart(api: ExplorerAPI) -> None:
    api.seed("selected", ("a", "b", "c"), text="exact phrase")
    api.seed("excluded", ("excluded",), text="exact phrase EXCLUDED_EVIDENCE")
    collection = api.container.paper_collections.create(name="Selection", document_ids=["selected"])
    params = {
        "query": "exact phrase",
        "scope": "collection",
        "scope_id": collection.collection_id,
        "limit": 1,
    }
    first = api.page("/explore/search", **params)
    assert identities(first) == ["selected"]
    assert "Resolved collection revision: 1" in first.text()
    assert values(first, "scope-document-id") == ['"selected"']
    following = href(first, "next-page")
    assert identities(api.page(following), "chunk-id") == ["b"]
    api.container.paper_collections.replace(
        collection.collection_id, name="Renamed", document_ids=["selected"], expected_revision=1
    )
    stale = api.client.get(following)
    assert stale.status_code == 422
    assert_headers(stale)
    root = ExplorerDOM(stale.text).root
    assert "Invalid or stale search cursor" in root.text()
    assert not root.by_tag("article")
    restarted = api.page(href(root, "first-page"))
    assert "Resolved collection revision: 2" in restarted.text()
    assert identities(restarted, "chunk-id") == ["a"]
    with closing(sqlite3.connect(api.path)) as connection, connection:
        connection.execute("DELETE FROM documents WHERE document_id = 'selected'")
    missing_member = api.client.get("/explore/search", params=params)
    assert missing_member.status_code == 409
    assert "Collection unavailable" in missing_member.text
    with closing(sqlite3.connect(api.path)) as connection, connection:
        connection.execute("UPDATE paper_collections SET document_ids = '[]'")
    empty_scope = api.client.get("/explore/search", params=params)
    assert empty_scope.status_code == 409
    assert not ExplorerDOM(empty_scope.text).root.by_tag("article")
    with closing(sqlite3.connect(api.path)) as connection, connection:
        connection.execute("DELETE FROM paper_collections")
    missing_collection = api.client.get("/explore/search", params=params)
    assert missing_collection.status_code == 404
    assert "Collection not found" in missing_collection.text


@pytest.mark.parametrize(
    "changed",
    [
        {"query": "Exact phrase"},
        {"query": "exact phrase "},
        {"scope": "all", "scope_id": ""},
        {"scope_id": "other"},
    ],
)
def test_search_cursor_is_bound_to_exact_query_and_scope(
    api: ExplorerAPI, changed: dict[str, str]
) -> None:
    api.seed(text="exact phrase")
    api.seed("other", ("other-a", "other-b"), text="exact phrase")
    params: dict[str, str | int] = {
        "query": "exact phrase",
        "scope": "document",
        "scope_id": "selected",
        "limit": 1,
    }
    first = api.page("/explore/search", **params)
    cursor = parse_qs(urlsplit(href(first, "next-page")).query)["cursor"][0]
    invalid = api.client.get("/explore/search", params={**params, "cursor": cursor, **changed})
    assert invalid.status_code == 422
    assert_headers(invalid)
    assert "Invalid or stale search cursor" in invalid.text
    assert cursor not in invalid.text
    assert not ExplorerDOM(invalid.text).root.by_tag("article")
    resized = api.page("/explore/search", **{**params, "cursor": cursor, "limit": 50})
    assert identities(resized, "chunk-id") == ["b", "c"]


@pytest.mark.parametrize(
    "change",
    [
        {"query": ""},
        {"query": None},
        {"query": " \t\n"},
        {"query": "x" * 201},
        {"scope": ""},
        {"scope": None},
        {"scope": "null"},
        {"scope": "documents"},
        {"scope": "document"},
        {"scope": "document", "scope_id": ""},
        {"scope": "document", "scope_id": None},
        {"scope": "document", "scope_id": " selected "},
        {"scope": "document", "scope_id": "x" * 129},
        {"scope": "collection", "scope_id": ""},
        {"scope": "collection", "scope_id": "null"},
        {"scope_id": "selected"},
        {"limit": ""},
        {"limit": None},
        {"limit": 0},
        {"limit": 51},
        {"limit": "1.0"},
        {"limit": "true"},
        {"limit": "1e0"},
        {"limit": "1" * 5000},
        {"cursor": ""},
        {"cursor": None},
        {"cursor": "x" * 4097},
        {"document_ids": "selected"},
        {"collection_id": "col_" + "0" * 32},
        {"PRIVATE_FIELD": "PRIVATE_INPUT"},
        {"catalog_limit": 101},
        {"catalog_limit": ""},
        {"catalog_cursor": ""},
        {"catalog_cursor": " selected"},
        {"source": "x" * 513},
        {"title": "x" * 301},
    ],
)
def test_invalid_form_values_fail_before_reader_without_echoing_inputs(
    api: ExplorerAPI,
    monkeypatch: pytest.MonkeyPatch,
    change: dict[str, str | int | None],
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(api.container.literal_search, "search", no_work)
    response = api.client.get("/explore/search", params={"query": "PRIVATE_PHRASE", **change})
    assert response.status_code == 422
    assert_headers(response)
    assert "Invalid passage search request" in response.text
    assert "PRIVATE_" not in response.text
    assert "PRIVATE_" not in "\n".join(
        record.message for record in caplog.records if record.name == "api.explorer"
    )


@pytest.mark.parametrize("name", ["query", "scope", "scope_id", "cursor", "limit", "source"])
def test_duplicate_form_fields_are_never_last_value_wins(
    api: ExplorerAPI, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    monkeypatch.setattr(api.container.literal_search, "search", no_work)
    response = api.client.get(
        "/explore/search", params=[("query", "exact phrase"), (name, ""), (name, "PRIVATE")]
    )
    assert response.status_code == 422
    assert_headers(response)
    assert "PRIVATE" not in response.text


@pytest.mark.parametrize("scope", ["document", "collection"])
def test_empty_return_scope_cannot_become_all_corpus(
    api: ExplorerAPI, monkeypatch: pytest.MonkeyPatch, scope: str
) -> None:
    monkeypatch.setattr(api.container.source_context, "read", no_work)
    response = api.client.get(
        "/explore/context",
        params={
            "document_id": "selected",
            "chunk_id": "a",
            "search_query": "exact phrase",
            "search_scope": scope,
            "search_scope_id": "",
            "search_limit": 1,
        },
    )
    assert response.status_code == 422
    assert_headers(response)
    assert not ExplorerDOM(response.text).root.by_tag("article")


@pytest.mark.parametrize(
    "identifier", [".", "..", "paper/../\u7814?%#&+\\", "nul\0id", "line\r\nid", "\U0001f52c" * 128]
)
def test_special_id_scope_and_control_navigation_are_lossless(
    api: ExplorerAPI, identifier: str
) -> None:
    chunks = ("\U0001f52c" * 256, "line\r\nchunk", "nul\0chunk")
    api.seed(identifier, chunks, text="exact phrase")
    first = api.page(
        "/explore/search", query="exact phrase", scope="document", scope_id=identifier, limit=1
    )
    second = api.page(href(first, "next-page"))
    assert identities(first) == identities(second) == [identifier]
    context = api.page(href(second, "source-context-1"))
    assert identities(context.by_id("context-anchor"), "chunk-id") == identities(second, "chunk-id")
    assert href(context, "back-to-search") == href(first, "next-page")
    for root in (first, second, context):
        for anchor in root.by_tag("a"):
            parts = urlsplit(anchor.attrs["href"])
            assert not parts.scheme and not parts.netloc
            assert parts.path == "" or parts.path.startswith("/explore")
    if any(character in identifier for character in "\0\r\n"):
        assert not first.by_tag("form")
    else:
        assert first.by_id("scope_id").attrs["value"] == identifier
        assert "maxlength" not in first.by_id("scope_id").attrs


def test_unicode_query_maximum_page_limits_and_exact_excerpt_boundary(api: ExplorerAPI) -> None:
    phrase = "\U0001f52c" * 200
    api.seed(chunk_ids=tuple(f"c-{index:02}" for index in range(51)), text=phrase + "z" * 600)
    default = api.page("/explore/search", query=phrase)
    assert len(default.by_tag("article")) == 20
    assert "maxlength" not in default.by_id("query").attrs
    first = api.page("/explore/search", query=phrase, limit=50)
    assert len(first.by_tag("article")) == 50
    assert len(api.page(href(first, "next-page")).by_tag("article")) == 1
    assert all(mark.text() == phrase for mark in first.by_tag("mark"))
    assert "Later chunk text is omitted" not in first.text()
    assert (
        api.client.get("/explore/search", params={"query": phrase + "\U0001f52c"}).status_code
        == 422
    )


def test_search_does_not_require_ordering_but_source_context_does(api: ExplorerAPI) -> None:
    api.seed(text="exact phrase")
    with closing(sqlite3.connect(api.path)) as connection, connection:
        connection.execute("UPDATE chunks SET metadata = '{}'")
    first = api.page("/explore/search", query="exact phrase", limit=1)
    context = api.client.get(href(first, "source-context-1"))
    assert context.status_code == 409
    assert_headers(context)
    root = ExplorerDOM(context.text).root
    assert "Source context unavailable" in root.text()
    assert not root.by_tag("article")
    assert identities(api.page(href(root, "back-to-search")), "chunk-id") == ["a"]


def test_missing_result_context_keeps_search_return_but_never_guesses_anchor(
    api: ExplorerAPI,
) -> None:
    api.seed(text="exact phrase")
    first = api.page("/explore/search", query="exact phrase", limit=1)
    with closing(sqlite3.connect(api.path)) as connection, connection:
        connection.execute("DELETE FROM chunks WHERE chunk_id = 'a'")
    response = api.client.get(href(first, "source-context-1"))
    assert response.status_code == 404
    root = ExplorerDOM(response.text).root
    assert "Passage not found" in root.text()
    assert not root.by_tag("article")
    assert identities(api.page(href(root, "back-to-search")), "chunk-id") == ["b"]


@pytest.mark.parametrize("field", ["title", "source", "text"])
def test_corrupt_or_unrepresentable_projected_text_fails_without_partial_matches(
    api: ExplorerAPI, field: str
) -> None:
    api.seed(chunk_ids=("a", "b"), **{"text": "exact phrase", field: "exact phrase\0PRIVATE"})
    response = api.client.get("/explore/search", params={"query": "exact phrase", "limit": 1})
    assert response.status_code == 409
    assert_headers(response)
    assert "PRIVATE" not in response.text
    assert not ExplorerDOM(response.text).root.by_tag("article")


def test_corrupt_lookahead_storage_and_deadline_errors_are_not_empty_results(
    api: ExplorerAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    import storage.literal_search as search_module

    api.seed(chunk_ids=tuple(f"c-{index:03}" for index in range(100)), text="exact phrase")
    with closing(sqlite3.connect(api.path)) as connection, connection:
        connection.execute("UPDATE chunks SET text = zeroblob(3) WHERE chunk_id = 'c-001'")
    corrupt = api.client.get("/explore/search", params={"query": "exact phrase", "limit": 1})
    assert corrupt.status_code == 409 and "Stored data cannot be displayed" in corrupt.text
    with monkeypatch.context() as patch:
        patch.setattr(search_module, "SEARCH_TIMEOUT_SECONDS", -1)
        timeout = api.client.get("/explore/search", params={"query": "absent"})
    assert timeout.status_code == 504 and "Passage search timed out" in timeout.text
    with closing(sqlite3.connect(api.path)) as connection, connection:
        connection.execute("DROP TABLE chunks")
    unavailable = api.client.get("/explore/search", params={"query": "exact phrase"})
    assert unavailable.status_code == 503 and "Corpus storage unavailable" in unavailable.text
    for response in (corrupt, timeout, unavailable):
        assert_headers(response)
        assert not ExplorerDOM(response.text).root.by_tag("article")


def test_actual_search_read_and_response_caps_preserve_all_or_nothing(api: ExplorerAPI) -> None:
    api.seed("huge", ("huge",), text="z" * (MAX_TEXT_BYTES + 1))
    oversized = api.client.get("/explore/search", params={"query": "absent"})
    assert (
        oversized.status_code == 413 and "Passage exceeds the search read limit" in oversized.text
    )
    api.seed(
        "selected",
        tuple(f"c-{index:02}" for index in range(50)),
        text="exact phrase" + "\U0001f52c" * 1000,
        title="\U0001f52c" * 301,
        source="\U0001f52c" * 513,
    )
    params: dict[str, str | int] = {
        "query": "exact phrase",
        "scope": "document",
        "scope_id": "selected",
        "limit": 50,
    }
    too_large = api.client.get("/explore/search", params=params)
    assert (
        too_large.status_code == 413 and "Search exceeds the response size limit" in too_large.text
    )
    root = api.page("/explore/search", **{**params, "limit": 1})
    assert len(values(root, "chunk-title")[0]) == 300
    assert len(values(root, "chunk-source")[0]) == 512
    assert "Title truncated" in root.text() and "Source truncated" in root.text()
    for response in (oversized, too_large):
        assert_headers(response)
        assert not ExplorerDOM(response.text).root.by_tag("article")


def test_exact_escaped_html_budget_rejects_one_byte_more_and_preserves_recovery(
    api: ExplorerAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    from api import explorer

    api.seed(text="exact phrase " + '"' * 1000)
    params = {"query": "exact phrase", "scope": "document", "scope_id": "selected"}
    complete = api.client.get("/explore/search", params=params)
    assert complete.status_code == 200
    monkeypatch.setattr(explorer, "MAX_HTML_BYTES", len(complete.content))
    assert api.client.get("/explore/search", params=params).content == complete.content
    monkeypatch.setattr(explorer, "MAX_HTML_BYTES", len(complete.content) - 1)
    too_large = api.client.get("/explore/search", params=params)
    assert too_large.status_code == 413 and "Page exceeds the HTML size limit" in too_large.text
    root = ExplorerDOM(too_large.text).root
    assert not root.by_tag("article")
    assert parse_qs(urlsplit(href(root, "first-page")).query)["scope_id"] == ["selected"]


@pytest.mark.parametrize(
    "error",
    [
        LiteralSearchError("search_storage_unavailable", "PRIVATE SQL", 503),
        CollectionError("collection_not_found", "PRIVATE SQL", 404),
    ],
)
def test_search_errors_never_render_reader_exception_messages(
    api: ExplorerAPI,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(api.container.literal_search, "search", Mock(side_effect=error))
    response = api.client.get("/explore/search", params={"query": "exact phrase"})
    assert response.status_code in {404, 503}
    assert_headers(response)
    assert "PRIVATE SQL" not in response.text + caplog.text


def test_one_readonly_search_snapshot_python_api_parity_restart_and_no_work(
    api: ExplorerAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    api.seed(text="exact phrase")
    collection = api.container.paper_collections.create(name="Selection", document_ids=["selected"])
    payload = {"query": "exact phrase", "collection_id": collection.collection_id, "limit": 1}
    expected = api.container.literal_search.search(LiteralSearchRequest.model_validate(payload))
    api_before = api.client.post("/research/search", json=payload).content
    before = api.path.read_bytes()
    original_connect = sqlite3.connect
    connections: list[str] = []
    statements: list[str] = []

    def connect(database: str, *, uri: bool, timeout: float = 5) -> sqlite3.Connection:
        assert uri and database.endswith("?mode=ro")
        connections.append(database)
        connection = original_connect(database, uri=uri, timeout=timeout)
        connection.set_trace_callback(statements.append)
        return connection

    search = Mock(wraps=api.container.literal_search.search)
    params = {
        "query": "exact phrase",
        "scope": "collection",
        "scope_id": collection.collection_id,
        "limit": 1,
    }
    with monkeypatch.context() as patch, browsing_guards(api.container) as guards:
        patch.setattr(sqlite3, "connect", connect)
        patch.setattr(api.container.literal_search, "search", search)
        response = api.client.get("/explore/search", params=params)
        assert response.status_code == 200
        assert len(connections) == 1
        search.assert_called_once_with(LiteralSearchRequest.model_validate(payload))
        root = ExplorerDOM(response.text).root
        assert identities(root, "chunk-id") == [match.chunk_id for match in expected.matches]
        api.page(href(root, "source-context-1"))
        assert len(connections) == 2
        assert not any(guard.call_count for guard in guards.values())
    assert all(
        statement.lstrip().split()[0] in {"BEGIN", "PRAGMA", "SELECT", "WITH"}
        for statement in statements
    )
    restarted = create_app(offline_settings(api.path))
    with TestClient(restarted) as client, browsing_guards(restarted.state.container):
        assert client.get("/explore/search", params=params).content == response.content
        assert client.post("/research/search", json=payload).content == api_before
    assert api.path.read_bytes() == before
    assert api.container.event_log.list_events() == []


def test_mounted_search_links_window_form_return_and_error_recovery(api: ExplorerAPI) -> None:
    api.seed(text="exact phrase")
    with TestClient(api.app, root_path="/library/nested") as client:
        catalog = ExplorerDOM(client.get("/library/nested/explore").text).root
        form = ExplorerDOM(client.get(href(catalog, "search-document-1")).text).root
        action = form.by_id("passage-search").attrs["action"]
        first = ExplorerDOM(
            client.get(action, params={"query": "exact phrase", "limit": 1}).text
        ).root
        following = href(first, "next-page")
        second = ExplorerDOM(client.get(following).text).root
        context = ExplorerDOM(client.get(href(second, "source-context-1")).text).root
        window = context.by_id("context-window")
        params = {node.attrs["name"]: node.attrs["value"] for node in window.by_tag("input")}
        params.update(before="0", after="0")
        selected = ExplorerDOM(client.get(window.attrs["action"], params=params).text).root
        assert identities(selected, "chunk-id") == ["b"]
        assert href(selected, "back-to-search") == following
        invalid = client.get(action, params={"query": "exact phrase", "cursor": "bad"})
        assert invalid.status_code == 422
        for root in (form, first, context, selected, ExplorerDOM(invalid.text).root):
            for anchor in root.by_tag("a"):
                path = urlsplit(anchor.attrs["href"]).path
                assert path == "" or path.startswith("/library/nested/explore")
            for native_form in root.by_tag("form"):
                assert native_form.attrs["action"].startswith("/library/nested/explore")


def test_openapi_describes_native_search_bounds_and_return_parameters(api: ExplorerAPI) -> None:
    schema = api.client.get("/openapi.json").json()
    operation = schema["paths"]["/explore/search"]["get"]
    assert {"200", "404", "409", "413", "422", "503", "504"} <= set(operation["responses"])
    fields = {field["name"]: field["schema"] for field in operation["parameters"]}
    query = next(item for item in fields["query"]["anyOf"] if item.get("type") == "string")
    assert query["minLength"] == 1 and query["maxLength"] == 200
    assert fields["limit"]["minimum"] == 1 and fields["limit"]["maximum"] == 50
    assert fields["scope"]["enum"] == ["all", "document", "collection"]
    context = schema["paths"]["/explore/context"]["get"]
    assert {"search_query", "search_scope", "search_scope_id", "search_limit", "search_cursor"} <= {
        field["name"] for field in context["parameters"]
    }
