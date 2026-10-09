"""Follow native explorer links into exact, source-ordered passage context."""

from urllib.parse import parse_qs, urlsplit

import pytest

from tests.test_corpus_explorer import (
    HOSTILE,
    PASSAGE,
    ExplorerAPI,
    ExplorerDOM,
    assert_headers,
    href,
    identities,
    values,
)
from tests.test_corpus_explorer import api as api


def test_displayed_passage_links_to_context_and_returns_to_exact_catalog_state(
    api: ExplorerAPI,
) -> None:
    api.seed("a-excluded", ())
    api.seed(chunk_ids=tuple(f"chunk-{index}" for index in range(14)))
    catalog = api.page(limit=2, source="synthetic:explorer", title="graph", cursor="a-excluded")
    document_url = href(catalog, "inspect-document-1")
    document = api.page(document_url)
    second_url = href(document, "next-page")
    second = api.page(second_url)
    assert identities(second, "chunk-id") == ["chunk-10", "chunk-11"]
    context_url = href(second, "source-context-1")
    root = api.page(context_url)
    assert "CURRENT CORPUS" in root.text() and "not frozen" in root.text()
    assert identities(root, "chunk-id") == [
        "chunk-8",
        "chunk-9",
        "chunk-10",
        "chunk-11",
        "chunk-12",
    ]
    assert root.by_id("context-anchor").by_tag("h3")[0].text() == "Selected passage"
    assert "ascending chunk_index" in root.text()
    assert href(root, "back-to-passages") == second_url
    assert href(root, "back-to-catalog") == href(second, "back-to-catalog")
    assert identities(api.page(href(root, "back-to-passages")), "chunk-id") == [
        "chunk-10",
        "chunk-11",
    ]
    recentered = api.page(href(root, "center-context-4"))
    assert identities(recentered.by_id("context-anchor"), "chunk-id") == ["chunk-11"]
    assert href(recentered, "back-to-passages") == second_url


def test_window_count_form_is_native_bounded_and_preserves_selection(api: ExplorerAPI) -> None:
    document_id = "paper/../\u7814?x#v1"
    chunk_ids = tuple(f"chunk-{index}" for index in range(14))
    api.seed(document_id, chunk_ids)
    root = api.page(
        "/explore/context",
        document_id=document_id,
        chunk_id="chunk-10",
        before=1,
        after=3,
        source="synthetic:explorer",
        title="graph",
        limit=2,
    )
    form = root.by_id("context-window")
    assert form.attrs["method"] == "get" and form.attrs["action"] == "/explore/context"
    inputs = {node.attrs["name"]: node for node in form.by_tag("input")}
    assert inputs["document_id"].attrs["value"] == document_id
    assert inputs["chunk_id"].attrs["value"] == "chunk-10"
    for name in ("before", "after"):
        assert inputs[name].attrs["type"] == "number"
        assert inputs[name].attrs["min"] == "0" and inputs[name].attrs["max"] == "5"
        assert form.by_tag("label")
    params = {name: node.attrs["value"] for name, node in inputs.items()}
    params.update(before="0", after="0")
    selected = api.page(form.attrs["action"], **params)
    assert identities(selected, "chunk-id") == ["chunk-10"]
    assert href(selected, "back-to-passages") == href(root, "back-to-passages")
    assert "More stored passages exist before" in selected.text()
    assert "More stored passages exist after" in selected.text()


@pytest.mark.parametrize(
    "document_id",
    [".", "..", "paper/../other", "x?%#&+\\", "nul\x00id", "line\r\nid", "\U0001f52c" * 128],
)
def test_context_links_and_recenter_preserve_special_ids(
    api: ExplorerAPI, document_id: str
) -> None:
    chunk_ids = ("chunk/../a?x#\u7814", "chunk\x00b", "chunk\r\nc")
    api.seed(document_id, chunk_ids)
    document = api.page("/explore/document", document_id=document_id)
    root = api.page(href(document, "source-context-1"))
    assert identities(root) == [document_id]
    assert identities(root, "chunk-id") == list(chunk_ids)
    assert identities(
        api.page(href(root, "center-context-2")).by_id("context-anchor"), "chunk-id"
    ) == ["chunk\x00b"]
    for anchor in root.by_tag("a"):
        parts = urlsplit(anchor.attrs["href"])
        assert not parts.scheme and not parts.netloc
        if "document_id" in parse_qs(parts.query):
            assert parse_qs(parts.query)["document_id"] == [document_id]


def test_context_markup_is_literal_and_uses_existing_csp_and_html_limits(api: ExplorerAPI) -> None:
    api.seed(
        'paper/"?x=<>&+', ('chunk-"><script>', "last"), title=HOSTILE, source=HOSTILE, text=PASSAGE
    )
    response = api.client.get(
        "/explore/context",
        params={"document_id": 'paper/"?x=<>&+', "chunk_id": 'chunk-"><script>'},
    )
    assert response.status_code == 200
    assert_headers(response)
    root = ExplorerDOM(response.text).root
    assert values(root, "passage-text") == [PASSAGE, PASSAGE]
    assert values(root, "chunk-source") == [HOSTILE, HOSTILE]
    assert identities(root.by_id("context-anchor"), "chunk-id") == ['chunk-"><script>']
    assert not any(root.by_tag(tag) for tag in ("script", "img", "iframe", "object", "base"))
    for node in root.walk():
        assert not any(key.startswith("on") or key == "style" for key in node.attrs)


def test_control_identity_disables_lossy_native_form_not_exact_navigation(api: ExplorerAPI) -> None:
    api.seed("nul\x00id", ("line\r\nchunk", "last"))
    root = api.page("/explore/context", document_id="nul\x00id", chunk_id="line\r\nchunk")
    assert not root.by_tag("form")
    assert "native form" in root.text()
    assert identities(api.page(href(root, "anchor-only")), "chunk-id") == ["line\r\nchunk"]


def test_context_truncation_warns_that_neighbors_do_not_complete_passage(api: ExplorerAPI) -> None:
    api.seed(text="\U0001f52c" * 4001)
    root = api.page("/explore/context", document_id="selected", chunk_id="a", before=0, after=0)
    assert values(root, "passage-text") == ["\U0001f52c" * 4000]
    assert "Passage truncated" in root.text()
    assert "not the rest of this passage" in root.text()


@pytest.mark.parametrize(
    ("params", "status", "label"),
    [
        ({}, 422, "Invalid source context request"),
        (
            {"document_id": "selected", "chunk_id": "a", "before": "6"},
            422,
            "Invalid source context request",
        ),
        ({"document_id": "selected", "chunk_id": "SECRET-missing"}, 404, "Passage not found"),
        ({"document_id": "SECRET-missing", "chunk_id": "a"}, 404, "Document not found"),
    ],
)
def test_context_errors_are_sanitized_html(
    api: ExplorerAPI, params: dict[str, str], status: int, label: str
) -> None:
    api.seed()
    response = api.client.get("/explore/context", params=params)
    assert response.status_code == status
    assert_headers(response)
    assert label in response.text and "SECRET" not in response.text
    assert not ExplorerDOM(response.text).root.by_tag("article")
