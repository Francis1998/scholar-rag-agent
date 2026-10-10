"""Browser-shaped acceptance tests over real, bounded, read-only corpus readers."""

import base64
import hashlib
import inspect
import json
import socket
import sqlite3
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn
from unittest.mock import Mock
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from scripts.demo_evidence_export import offline_settings

from api.application import create_app
from api.dependencies import AppContainer
from retrieval.models import Chunk, Document
from storage.document_catalog import DocumentCatalogError
from storage.document_chunks import DocumentChunksError
from tests.test_evidence_html import Node, ReaderDOM

HOSTILE = (
    '"><script>alert("fixture")</script><img src="//example.invalid/pixel" onerror="x">'
    '<base href="//example.invalid/"> & [link](javascript:alert(1))'
)
PASSAGE = "\n\n  Synthetic evidence.\r\n\tLiteral &lt;b&gt; and cafe\u0301.\r" + HOSTILE


class ExplorerDOM(ReaderDOM):
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        super().handle_starttag(tag, attrs)
        if tag == "input":
            self.stack.pop()


def no_work(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("Exploring must not generate, retrieve, mutate, or access the network.")


@dataclass
class ExplorerAPI:
    app: FastAPI
    client: TestClient
    container: AppContainer
    path: Path

    def seed(
        self,
        document_id: str = "selected",
        chunk_ids: tuple[str, ...] = ("c", "a", "b"),
        *,
        title: str = "Synthetic Graph methods",
        source: str = "synthetic:explorer",
        text: str = "Synthetic stored evidence.",
    ) -> None:
        self.container.document_store.add_documents(
            [
                Document(
                    document_id=document_id,
                    title=title,
                    source=source,
                    text="UNEXPOSED_DOCUMENT_BODY",
                    metadata={"private": "UNEXPOSED_METADATA"},
                )
            ],
            [
                Chunk(
                    chunk_id=identifier,
                    document_id=document_id,
                    title=title,
                    source=source,
                    text=text,
                    metadata={"chunk_index": str(index), "private": "UNEXPOSED_METADATA"},
                )
                for index, identifier in enumerate(chunk_ids)
            ],
        )

    def page(self, path: str = "/explore", **params: str | int) -> Node:
        response = self.client.get(path, params=params) if params else self.client.get(path)
        assert response.status_code == 200, response.text[:1000]
        assert_headers(response)
        assert "UNEXPOSED_" not in response.text
        return ExplorerDOM(response.text).root


@pytest.fixture
def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[ExplorerAPI]:
    for target, method in (
        (httpx.HTTPTransport, "handle_request"),
        (httpx.AsyncHTTPTransport, "handle_async_request"),
        (socket, "create_connection"),
        (socket.socket, "connect"),
    ):
        monkeypatch.setattr(target, method, no_work)
    path = tmp_path / "corpus.sqlite3"
    app = create_app(offline_settings(path))
    with TestClient(app) as client:
        yield ExplorerAPI(app, client, app.state.container, path)


def assert_headers(response: httpx.Response) -> None:
    assert response.headers["content-type"] == "text/html; charset=utf-8"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["referrer-policy"] == "no-referrer"
    csp = response.headers["content-security-policy"]
    for directive in (
        "default-src 'none'",
        "script-src 'none'",
        "base-uri 'none'",
        "object-src 'none'",
        "frame-src 'none'",
        "frame-ancestors 'none'",
        "form-action 'self'",
    ):
        assert directive in csp
    assert "'unsafe-inline'" not in csp and "'unsafe-eval'" not in csp
    root = ExplorerDOM(response.text).root
    style = root.by_tag("style")
    assert len(style) == 1
    digest = base64.b64encode(hashlib.sha256(style[0].text().encode()).digest()).decode()
    assert f"style-src 'sha256-{digest}'" in csp
    assert len(response.content) <= 1_048_576


def href(root: Node, identifier: str) -> str:
    return root.by_id(identifier).attrs["href"]


def values(root: Node, css_class: str) -> list[str]:
    return [node.text() for node in root.by_class(css_class)]


def identities(root: Node, css_class: str = "document-id") -> list[str]:
    return [json.loads(value) for value in values(root, css_class)]


def test_mounted_explorer_preserves_root_path_in_forms_links_and_errors(api: ExplorerAPI) -> None:
    api.seed()
    with TestClient(api.app, root_path="/library") as client:
        response = client.get("/library/explore", params={"limit": 1})
        assert response.status_code == 200
        catalog = ExplorerDOM(response.text).root
        assert catalog.by_tag("form")[0].attrs["action"] == "/library/explore"
        document = ExplorerDOM(client.get(href(catalog, "inspect-document-1")).text).root
        context = ExplorerDOM(client.get(href(document, "source-context-1")).text).root
        for root in (catalog, document, context):
            for anchor in root.by_tag("a"):
                path = urlsplit(anchor.attrs["href"]).path
                assert path == "" or path.startswith("/library/explore")
            for form in root.by_tag("form"):
                assert form.attrs["action"].startswith("/library/explore")
        invalid = client.get("/library/explore/context")
        assert invalid.status_code == 422
        assert "Invalid source context request" in invalid.text
        assert '/library/explore"' in invalid.text


def test_catalog_filters_and_pagination_preserve_browser_state(api: ExplorerAPI) -> None:
    for document_id, title, source in (
        ("a", "100%_Graph excluded source", "wrong"),
        ("b", "Other title", "selected"),
        ("c", "100%_Graph methods", "selected"),
        ("d", "100%_GRAPH limitations", "selected"),
        ("e", "100%_Graph case-sensitive source", "SELECTED"),
    ):
        api.seed(document_id, (f"{document_id}-chunk",), title=title, source=source)
    first = api.page(limit=1, source="selected", title="%_graph")
    assert identities(first) == ["c"]
    next_url = href(first, "next-page")
    assert parse_qs(urlsplit(next_url).query) == {
        "limit": ["1"],
        "source": ["selected"],
        "title": ["%_graph"],
        "cursor": ["c"],
    }
    second = api.page(next_url)
    assert identities(second) == ["d"]
    assert not second.by_class("next-page")
    assert href(second, "reset-filters") == "/explore"
    assert "cursor" not in parse_qs(urlsplit(href(second, "first-page")).query)
    document = api.page(href(second, "inspect-document-1"))
    assert href(document, "back-to-catalog") == next_url
    assert values(document, "passage-text") == ["Synthetic stored evidence."]
    assert identities(api.page(href(document, "back-to-catalog"))) == ["d"]


def test_passages_paginate_by_chunk_id_not_ordinal(api: ExplorerAPI) -> None:
    api.seed(chunk_ids=("f", "d", "c", "b", "a"))
    api.seed("excluded", ("aa", "bb"), text="EXCLUDED_EVIDENCE")
    first = api.page("/explore/document", document_id="selected", limit=2)
    second = api.page(href(first, "next-page"))
    third = api.page(href(second, "next-page"))
    assert [identities(root, "chunk-id") for root in (first, second, third)] == [
        ["a", "b"],
        ["c", "d"],
        ["f"],
    ]
    assert values(first, "chunk-index") == ["4", "3"]
    assert all(identities(root) == ["selected"] for root in (first, second, third))
    assert "ascending chunk ID" in first.text()
    assert "not chunk_index" in first.text()
    assert "EXCLUDED_EVIDENCE" not in "".join(root.text() for root in (first, second, third))
    assert not third.by_class("next-page")
    assert "cursor" not in parse_qs(urlsplit(href(third, "first-page")).query)


@pytest.mark.parametrize(
    "document_id",
    [
        ".",
        "..",
        "/",
        "paper/../other",
        "./paper",
        "//example.invalid/a/..",
        "https://example.invalid/a/../b",
        "quotes'\"?%#&+\\",
        "paper-\u03b2/\u7814\u7a76",
        "cafe\u0301/\u202eRTL\u202c",
        "nul\x00id",
        "line\r\nid",
        "\U0001f52c" * 128,
    ],
)
def test_query_document_identity_survives_links_and_cursor_exactly(
    api: ExplorerAPI, document_id: str
) -> None:
    api.seed(document_id, ("chunk-\u7814" * 20, "z-last"))
    api.seed("excluded", ("excluded-chunk",), text="EXCLUDED_EVIDENCE")
    catalog = api.page(source="synthetic:explorer")
    index = identities(catalog).index(document_id) + 1
    url = href(catalog, f"inspect-document-{index}")
    assert urlsplit(url).path == "/explore/document"
    assert parse_qs(urlsplit(url).query)["document_id"] == [document_id]
    first = api.page("/explore/document", document_id=document_id, limit=1)
    second = api.page(href(first, "next-page"))
    assert identities(first) == identities(second) == [document_id]
    assert len(identities(first, "chunk-id") + identities(second, "chunk-id")) == 2
    assert "EXCLUDED_EVIDENCE" not in first.text() + second.text()
    for root in (catalog, first, second):
        for anchor in root.by_tag("a"):
            parts = urlsplit(anchor.attrs["href"])
            assert not parts.scheme and not parts.netloc
            assert parts.path in {
                "",
                "/explore",
                "/explore/document",
                "/explore/context",
                "/explore/search",
            }


def test_empty_form_filters_are_omitted_but_nonempty_values_are_not_changed(
    api: ExplorerAPI,
) -> None:
    api.seed("a", ("a1",), title="Graph", source=" selected ")
    api.seed("b", ("b1",), title=" Graph ", source="selected")
    assert identities(api.page(source="", title="")) == ["a", "b"]
    assert identities(api.page(source="", title=" ")) == ["b"]
    assert identities(api.page(source=" selected ", title="")) == ["a"]
    assert identities(api.page(source="selected", title=" graph ")) == ["b"]
    assert identities(api.page(source="SELECTED", title="")) == []
    assert api.client.get("/documents", params={"source": ""}).status_code == 422
    assert api.client.get("/documents", params={"title": ""}).status_code == 422
    root = api.page(source="", title="")
    form = root.by_tag("form")[0]
    assert form.attrs["method"].lower() == "get"
    assert form.attrs["action"] == "/explore"
    inputs = {node.attrs["name"]: node for node in form.by_tag("input")}
    assert inputs["source"].attrs["value"] == inputs["title"].attrs["value"] == ""
    assert inputs["limit"].attrs["min"] == "1" and inputs["limit"].attrs["max"] == "100"
    assert "cursor" not in inputs
    labels = {node.attrs["for"] for node in form.by_tag("label")}
    assert all(node.attrs["id"] in labels for node in inputs.values())


@pytest.mark.parametrize(
    ("field", "limit", "symbol"),
    [("title", 300, "\U0001f52c"), ("source", 512, "\U0001f4da")],
)
def test_astral_filter_bounds_match_native_inputs_and_exact_pagination(
    api: ExplorerAPI, field: str, limit: int, symbol: str
) -> None:
    filters = {"title": "\U0001f52c" * 300, "source": "\U0001f4da" * 512}
    for identifier in ("a", "b"):
        api.seed(identifier, (f"{identifier}-chunk",), **filters)
    before = api.path.read_bytes()
    first = api.page(limit=1, **filters)
    assert identities(first) == ["a"]
    inputs = {node.attrs["name"]: node for node in first.by_tag("input")}
    assert len(filters[field]) == limit
    assert len(filters[field].encode("utf-16-le")) // 2 == limit * 2
    assert "maxlength" not in inputs[field].attrs
    for name, value in filters.items():
        assert inputs[name].attrs["value"] == value
    next_url = href(first, "next-page")
    query = parse_qs(urlsplit(next_url).query)
    for name, value in filters.items():
        assert query[name] == [value]
    second = api.page(next_url)
    assert identities(second) == ["b"]
    document = api.page(href(second, "inspect-document-1"))
    assert href(document, "back-to-catalog") == next_url
    assert identities(api.page(href(document, "back-to-catalog"))) == ["b"]
    response = api.client.get("/explore", params={**filters, field: filters[field] + symbol})
    assert response.status_code == 422
    assert_headers(response)
    assert filters[field] not in response.text
    assert "Invalid explorer request" in response.text
    assert api.client.get("/documents", params=filters).status_code == 200
    assert api.path.read_bytes() == before


@pytest.mark.parametrize("title", [" ", " \t\r\n", "\u2003", "\u00a0\u3000", " \t\n\u2003"])
def test_whitespace_title_labels_have_a_usable_inspection_link(
    api: ExplorerAPI, title: str
) -> None:
    api.seed(title=title)
    before = api.path.read_bytes()
    catalog = api.page()
    link = catalog.by_id("inspect-document-1")
    assert link.text() == "Untitled paper"
    assert identities(api.page(link.attrs["href"])) == ["selected"]
    assert api.client.get("/documents").json()["documents"][0]["title"] == title
    assert api.path.read_bytes() == before


@pytest.mark.parametrize("title", ["  Graph methods  ", "\u2003\t\nGraph \U0001f52c\u00a0"])
def test_nonblank_title_labels_preserve_original_padding(api: ExplorerAPI, title: str) -> None:
    api.seed(title=title)
    assert api.page().by_id("inspect-document-1").text() == title
    assert api.client.get("/documents").json()["documents"][0]["title"] == title


def test_control_character_filters_are_not_silently_rewritten_by_html_inputs(
    api: ExplorerAPI,
) -> None:
    for identifier in ("a", "b"):
        api.seed(identifier, (), title="Line\r\nGraph", source="synthetic")
    first = api.page(title="\r\n", limit=1)
    assert identities(first) == ["a"]
    assert "native text inputs cannot preserve" in first.text()
    assert not first.by_tag("form")
    assert parse_qs(urlsplit(href(first, "next-page")).query)["title"] == ["\r\n"]
    assert identities(api.page(href(first, "next-page"))) == ["b"]


def test_literal_fields_attribute_escaping_and_navigation_never_execute_text(
    api: ExplorerAPI,
) -> None:
    identifier = 'paper/"?x=<>&+'
    api.seed(identifier, ('chunk-"><script>', "last"), title=HOSTILE, source=HOSTILE, text=PASSAGE)
    catalog = api.page(source=HOSTILE, title=HOSTILE[:40], limit=1)
    document = api.page(href(catalog, "inspect-document-1"))
    assert values(catalog, "document-title") == [HOSTILE]
    assert values(document, "passage-text") == [PASSAGE]
    assert values(document, "chunk-source") == [HOSTILE]
    assert identities(document) == [identifier]
    assert identities(document, "chunk-id") == ['chunk-"><script>']
    inputs = {node.attrs["name"]: node.attrs["value"] for node in catalog.by_tag("input")}
    assert inputs["source"] == HOSTILE and inputs["title"] == HOSTILE[:40]
    for root in (catalog, document):
        assert not any(root.by_tag(tag) for tag in ("script", "img", "iframe", "object", "base"))
        assert len(root.by_tag("h1")) == 1
        assert root.by_tag("html")[0].attrs["lang"] == "en"
        assert root.by_id("main").tag == "main"
        for node in root.walk():
            assert not any(key.startswith("on") or key == "style" for key in node.attrs)
        for anchor in root.by_tag("a"):
            parsed = urlsplit(anchor.attrs["href"])
            assert not parsed.netloc and not parsed.scheme
            assert "example.invalid" not in parsed.path


def test_empty_states_are_truthful_and_distinct(api: ExplorerAPI) -> None:
    assert "The local corpus is empty" in api.page().text()
    api.seed("empty", ())
    assert "No papers match these filters" in api.page(title="no match").text()
    assert "No papers after this cursor" in api.page(cursor="z").text()
    assert (
        "This document has no stored passages"
        in api.page("/explore/document", document_id="empty").text()
    )
    response = api.client.get("/explore/document", params={"document_id": "missing-secret"})
    assert response.status_code == 404
    assert_headers(response)
    assert "Document not found" in response.text and "missing-secret" not in response.text
    api.seed("selected", ("a", "b"))
    first = api.page("/explore/document", document_id="selected", limit=1)
    with sqlite3.connect(api.path) as connection:
        connection.execute("DELETE FROM chunks WHERE document_id = 'selected'")
    assert "No passages remain after this cursor" in api.page(href(first, "next-page")).text()


def test_explicit_prefix_limits_empty_text_and_missing_ordinals(api: ExplorerAPI) -> None:
    api.seed(title="t" * 301, source="s" * 513, text="\U0001f52c" * 4001)
    catalog = api.page()
    assert values(catalog, "document-title") == ["t" * 300]
    assert values(catalog, "document-source") == ["s" * 512]
    assert "Title truncated" in catalog.text() and "Source truncated" in catalog.text()
    doc = api.page("/explore/document", document_id="selected", limit=1)
    assert values(doc, "passage-text") == ["\U0001f52c" * 4000]
    assert "Passage truncated" in doc.text()
    assert "not the rest of this passage" in doc.text()
    assert "Title truncated" in doc.text() and "Source truncated" in doc.text()
    api.seed("empty-text", ("empty-text-chunk",), text="")
    with sqlite3.connect(api.path) as connection:
        connection.execute("UPDATE chunks SET metadata = '{}' WHERE document_id = 'empty-text'")
    empty = api.page("/explore/document", document_id="empty-text")
    assert values(empty, "chunk-index") == ["Not recorded"]
    assert "This stored passage is empty" in empty.text()


@pytest.mark.parametrize(
    ("path", "params"),
    [
        ("/explore", {"limit": "0"}),
        ("/explore", {"limit": "101"}),
        ("/explore", {"limit": "1.5"}),
        ("/explore", {"limit": ""}),
        ("/explore", {"cursor": ""}),
        ("/explore", {"cursor": " secret"}),
        ("/explore", {"cursor": "s" * 129}),
        ("/explore", {"source": "s" * 513}),
        ("/explore", {"title": "s" * 301}),
        ("/explore/document", {}),
        ("/explore/document", {"document_id": ""}),
        ("/explore/document", {"document_id": " secret"}),
        ("/explore/document", {"document_id": "s" * 129}),
        ("/explore/document", {"document_id": "selected", "cursor": ""}),
        ("/explore/document", {"document_id": "selected", "cursor": "not-a-cursor-secret"}),
        ("/explore/document", {"document_id": "selected", "cursor": "s" * 4097}),
        ("/explore/document", {"document_id": "selected", "catalog_cursor": " secret"}),
        ("/explore/document", {"document_id": "selected", "limit": "101"}),
    ],
)
def test_validation_errors_are_html_and_never_echo_input(
    api: ExplorerAPI, path: str, params: dict[str, str], caplog: pytest.LogCaptureFixture
) -> None:
    api.seed()
    response = api.client.get(path, params=params)
    assert response.status_code == 422
    assert_headers(response)
    assert "secret" not in response.text
    assert "Traceback" not in response.text
    assert "Invalid" in response.text
    assert any(record.name == "api.explorer" for record in caplog.records)
    assert "secret" not in "\n".join(
        record.message for record in caplog.records if record.name == "api.explorer"
    )


def test_document_bound_cursor_cannot_read_another_paper(api: ExplorerAPI) -> None:
    api.seed()
    api.seed("other", ("other-a", "other-b"))
    first = api.page("/explore/document", document_id="selected", limit=1)
    cursor = parse_qs(urlsplit(href(first, "next-page")).query)["cursor"][0]
    response = api.client.get(
        "/explore/document", params={"document_id": "other", "cursor": cursor}
    )
    assert response.status_code == 422
    assert_headers(response)
    assert "exact document" in response.text and cursor not in response.text


@pytest.mark.parametrize("reader", ["catalog", "chunks"])
def test_corrupt_projected_lookahead_rejects_the_whole_page(api: ExplorerAPI, reader: str) -> None:
    api.seed("a", ("a1", "a2"))
    api.seed("b", ())
    with sqlite3.connect(api.path) as connection:
        if reader == "catalog":
            connection.execute("UPDATE documents SET title = zeroblob(3) WHERE document_id = 'b'")
        else:
            connection.execute("UPDATE chunks SET metadata = 'secret-not-json' WHERE chunk_id='a2'")
    path = "/explore" if reader == "catalog" else "/explore/document"
    params: dict[str, str | int] = {"limit": 1}
    if reader == "chunks":
        params["document_id"] = "a"
    response = api.client.get(path, params=params)
    assert response.status_code == 409
    assert_headers(response)
    assert "Stored data cannot be displayed" in response.text
    assert "secret" not in response.text
    assert not ExplorerDOM(response.text).root.by_tag("article")


@pytest.mark.parametrize("reader", ["catalog", "chunks"])
def test_storage_failure_is_sanitized_unavailable_not_empty_success(
    api: ExplorerAPI,
    reader: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    api.seed()
    target = (
        api.container.document_catalog if reader == "catalog" else api.container.document_chunks
    )
    method = "list_documents" if reader == "catalog" else "list_chunks"
    monkeypatch.setattr(
        target,
        method,
        Mock(
            side_effect=sqlite3.OperationalError("SELECT secret FROM /private/credential.sqlite3")
        ),
    )
    response = api.client.get(
        "/explore" if reader == "catalog" else "/explore/document",
        params={} if reader == "catalog" else {"document_id": "selected"},
    )
    assert response.status_code == 503
    assert_headers(response)
    assert "Corpus storage unavailable" in response.text
    assert "SELECT secret" not in response.text + caplog.text
    assert "/private/" not in response.text + caplog.text
    assert "credential" not in response.text + caplog.text
    assert "storage_unavailable" in caplog.text


@pytest.mark.parametrize("error", [DocumentCatalogError(), DocumentChunksError()])
def test_reader_error_messages_are_not_trusted_for_html(
    api: ExplorerAPI, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    error.args = ("private-secret <script>SELECT * FROM corpus</script>",)
    monkeypatch.setattr(api.container.document_catalog, "list_documents", Mock(side_effect=error))
    response = api.client.get("/explore", params={"source": "input-secret"})
    assert response.status_code == 409
    assert_headers(response)
    assert "secret" not in response.text and "SELECT *" not in response.text


def test_nul_passage_fails_honestly_but_json_remains_readable(api: ExplorerAPI) -> None:
    api.seed(text="before\x00after")
    response = api.client.get("/explore/document", params={"document_id": "selected"})
    assert response.status_code == 409
    assert_headers(response)
    assert "cannot be represented faithfully" in response.text
    assert (
        api.client.get("/documents/selected/chunks").json()["chunks"][0]["text"]
        == "before\x00after"
    )


def test_bounded_response_fails_instead_of_silently_dropping_passages(api: ExplorerAPI) -> None:
    api.seed(chunk_ids=tuple(f"c-{index:03}" for index in range(100)), text='"' * 4000)
    response = api.client.get("/explore/document", params={"document_id": "selected", "limit": 100})
    assert response.status_code == 413
    assert_headers(response)
    assert "Page exceeds the HTML size limit" in response.text
    assert "smaller limit" in response.text
    assert not ExplorerDOM(response.text).root.by_class("passage-text")
    assert (
        len(api.page("/explore/document", document_id="selected", limit=1).by_tag("article")) == 1
    )


def test_html_byte_budget_accepts_exact_boundary_and_rejects_one_byte_more(
    api: ExplorerAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from api import explorer

    api.seed(text="Exact Unicode boundary \U0001f52c " * 100)
    url = "/explore/document?document_id=selected&limit=1"
    full = api.client.get(url)
    assert full.status_code == 200
    monkeypatch.setattr(explorer, "MAX_HTML_BYTES", len(full.content))
    assert api.client.get(url).content == full.content
    monkeypatch.setattr(explorer, "MAX_HTML_BYTES", len(full.content) - 1)
    oversized = api.client.get(url)
    assert oversized.status_code == 413
    assert len(oversized.content) < len(full.content)


def test_catalog_and_passages_respect_default_and_maximum_row_limits(api: ExplorerAPI) -> None:
    api.seed(chunk_ids=tuple(f"chunk-{index:03}" for index in range(101)))
    assert len(api.page("/explore/document", document_id="selected").by_tag("article")) == 20
    maximum = api.page("/explore/document", document_id="selected", limit=100)
    assert len(maximum.by_tag("article")) == 100
    assert identities(api.page(href(maximum, "next-page")), "chunk-id") == ["chunk-100"]
    for index in range(100):
        api.seed(f"doc-{index:03}", ())
    assert len(api.page().by_tag("article")) == 20
    maximum = api.page(limit=100)
    assert len(maximum.by_tag("article")) == 100
    assert identities(api.page(href(maximum, "next-page"))) == ["selected"]


def test_page_reads_only_one_bounded_page_from_the_existing_readers(
    api: ExplorerAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    api.seed()
    catalog = Mock(wraps=api.container.document_catalog.list_documents)
    chunks = Mock(wraps=api.container.document_chunks.list_chunks)
    monkeypatch.setattr(api.container.document_catalog, "list_documents", catalog)
    monkeypatch.setattr(api.container.document_chunks, "list_chunks", chunks)
    before = api.path.read_bytes()
    events = api.container.event_log.list_events()
    for target, method in (
        (api.container.document_store, "list_chunks"),
        (api.container.document_store, "add_documents"),
        (api.container.hybrid_retriever, "retrieve"),
        (api.container.hybrid_retriever, "add_chunks"),
        (api.container.runner._executor._reranker, "rerank"),
        (api.container.llm, "generate"),
        (api.container.runner, "run"),
        (api.container.runner, "preview"),
        (api.container.event_log, "append_event"),
        (api.container.ingestion_pipeline, "ingest_documents"),
        (api.container.paper_collections, "create"),
    ):
        monkeypatch.setattr(target, method, no_work)
    api.page(source="", title="Graph", limit=1)
    catalog.assert_called_once_with(limit=1, cursor=None, source=None, title="Graph")
    chunks.assert_not_called()
    api.page("/explore/document", document_id="selected", limit=2)
    chunks.assert_called_once_with("selected", limit=2, cursor=None)
    catalog.assert_called_once()
    assert api.path.read_bytes() == before
    assert api.container.event_log.list_events() == events


def test_restart_and_concurrent_requests_keep_exact_scope_and_json_contract(
    api: ExplorerAPI,
) -> None:
    api.seed("a", ("a1", "a2"))
    api.seed("b", ("b1", "b2"))
    catalog_json = api.client.get("/documents").content
    chunks_json = api.client.get("/documents/a/chunks", params={"limit": 1}).content
    first = api.client.get("/explore", params={"limit": 1}).content
    api.app.state.container = AppContainer(offline_settings(api.path))
    assert api.client.get("/explore", params={"limit": 1}).content == first
    before = api.path.read_bytes()

    def read(document_id: str) -> list[str]:
        return identities(api.page("/explore/document", document_id=document_id, limit=1))

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(read, ["a", "b"] * 4)) == [["a"], ["b"]] * 4
    assert api.client.get("/documents").content == catalog_json
    assert api.client.get("/documents/a/chunks", params={"limit": 1}).content == chunks_json
    assert api.path.read_bytes() == before
    assert api.app.state.container.event_log.list_events() == []


def test_explorer_routes_are_sync_get_only_and_described_as_html(api: ExplorerAPI) -> None:
    from api.explorer import router

    routes = [
        route
        for route in router.routes
        if isinstance(route, APIRoute) and route.path.startswith("/explore")
    ]
    assert {route.path for route in routes} == {
        "/explore",
        "/explore/document",
        "/explore/context",
        "/explore/search",
    }
    for route in routes:
        assert route.methods == {"GET"}
        assert not inspect.iscoroutinefunction(route.endpoint)
        assert api.client.post(route.path).status_code == 405
    schema = api.client.get("/openapi.json").json()
    for path in ("/explore", "/explore/document", "/explore/context", "/explore/search"):
        responses = schema["paths"][path]["get"]["responses"]
        assert "text/html" in responses["200"]["content"]
        assert {"409", "413", "422", "503"} <= set(responses)
