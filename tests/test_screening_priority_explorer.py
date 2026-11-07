"""Opt-in most-likely-relevant-first browser queue beside the existing decision forms."""

import json
import sqlite3
from unittest.mock import Mock
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from scripts.demo_corpus_explorer import browsing_guards
from starlette.types import Message, Scope

from api import explorer, screening_explorer
from api.explorer import _decode_collection_return
from retrieval.models import Document
from storage.paper_collections import CollectionError, PaperCollection
from storage.paper_screening import ScreeningSubmission
from storage.screening_priority import ScreeningPriority
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

PRIORITY = "/explore/screening-priority"
SAVED_ORDER = ("z-battery", "seed-include", "hostile", "seed-exclude", "grounded", "unsure")


def seed(api: ExplorerAPI, *, decide: bool = True) -> PaperCollection:
    for identifier, title in (
        ("z-battery", "Battery cathode fade"),
        ("seed-include", "Citation grounding for evidence answers"),
        ("hostile", "Citation evidence " + HOSTILE),
        ("seed-exclude", "Battery cathode coatings"),
        ("grounded", "Grounded citation evidence answers"),
        ("unsure", "Evidence for battery citation"),
    ):
        api.seed(identifier, (identifier + "/chunk",), title=title)
    collection = api.container.paper_collections.create(
        name="Synthetic <priority> review", document_ids=SAVED_ORDER
    )
    if decide:
        for identifier, decision in (
            ("seed-include", "include"),
            ("seed-exclude", "exclude"),
            ("unsure", "unsure"),
        ):
            api.container.paper_screening.submit(
                collection.collection_id,
                identifier,
                ScreeningSubmission.model_validate(
                    {
                        "collection_revision": 1,
                        "expected_decision_revision": 0,
                        "decision": decision,
                        "reason": "Synthetic human opinion.",
                    }
                ),
            )
    return collection


def queue_url(collection: PaperCollection, **params: str | int) -> str:
    return (
        PRIORITY
        + "?"
        + urlencode(
            {
                "collection_id": collection.collection_id,
                "collection_revision": collection.revision,
                **params,
            }
        )
    )


def failure(response: httpx.Response, status: int, message: str) -> Node:
    assert response.status_code == status, response.text[:1000]
    assert_headers(response)
    root = ExplorerDOM(response.text).root
    assert not root.by_class("priority-item") and not root.by_tag("form")
    assert f"HTTP {status}. {message}" in root.text()
    assert "READ-ONLY / SCREENING PRIORITIZATION" in root.text()
    assert "PRIVATE_" not in response.text and "UNEXPOSED_" not in response.text
    assert "location" not in response.headers
    return root


def test_opt_in_queue_ranks_pending_papers_and_escapes_titles(api: ExplorerAPI) -> None:
    collection = seed(api)
    detail = api.page(detail_url(collection))
    assert identities(detail) == list(SAVED_ORDER)
    assert not detail.by_class("priority-score")
    link = href(detail, "screening-priority")
    assert urlsplit(link).path == PRIORITY
    before = api.path.read_bytes()
    with browsing_guards(api.container) as guards:
        root = api.page(link)
    assert not any(guard.call_count for guard in guards.values())
    assert api.path.read_bytes() == before
    expected = api.client.get(
        f"/collections/{collection.collection_id}/screening-priority",
        params={"collection_revision": 1},
    ).json()
    ranked = [item["document_id"] for item in expected["items"]]
    assert ranked[0] in {"grounded", "hostile"} and ranked[-1] == "z-battery"
    assert identities(root)[: len(ranked)] == ranked
    assert identities(root)[len(ranked) :] == ["unsure"]
    assert values(root, "priority-rank") == ["1", "2", "3", "1"]
    assert values(root, "priority-score") == [
        f"{item['score']:.3f}" for item in expected["items"]
    ] + [f"{expected['unsure_items'][0]['score']:.3f}"]
    assert values(root, "model-status") == ["ready"]
    hostile_index = ranked.index("hostile")
    assert values(root, "member-title")[hostile_index] == "Citation evidence " + HOSTILE
    response = api.client.get(link)
    assert "<script>alert" not in response.text and "<img src" not in response.text
    assert "&lt;script&gt;alert" in response.text
    top_terms = [term["term"] for term in expected["items"][0]["top_terms"]]
    assert values(root, "top-term")[: len(top_terms)] == top_terms
    assert "Ordering only" in root.by_id("priority-guarantee").text()
    assert "No stopping signal is shown" in root.by_id("stopping-signal").text()
    assert "READ-ONLY / SCREENING PRIORITIZATION" in root.text()
    assert not root.by_tag("form")


def test_cards_open_the_existing_form_and_the_form_returns_to_the_queue(
    api: ExplorerAPI,
) -> None:
    collection = seed(api)
    detail = api.page(detail_url(collection, limit=2, status="unscreened"))
    queue = api.page(href(detail, "screening-priority"))
    assert urlsplit(href(queue, "back-to-collection")).path == "/explore/collection"
    returned = parse_qs(urlsplit(href(queue, "back-to-collection")).query)
    assert returned["status"] == ["unscreened"] and returned["limit"] == ["2"]
    first = identities(queue)[0]
    form = api.page(href(queue, "screen-member-1"), referrer_policy="same-origin")
    selection = json.loads(
        next(
            node.attrs["value"]
            for node in form.by_id("screening-form").by_tag("input")
            if node.attrs["name"] == "selection"
        )
    )
    assert selection["document_id"] == first
    assert _decode_collection_return(selection["collection_return"]).status == "unscreened"
    assert form.by_tag("textarea")[0].text().strip() == ""
    assert form.by_id("decision").by_tag("option")[0].attrs["value"] == ""
    again = api.page(href(form, "screening-priority"))
    assert identities(again) == identities(queue)
    paper = api.page(href(queue, "read-member-1"))
    assert identities(paper)[0] == first
    unsure = api.page(href(queue, "unsure-screen-member-1"), referrer_policy="same-origin")
    assert identities(unsure) == ["unsure"]


def test_default_collection_order_and_form_are_unchanged_by_prioritization(
    api: ExplorerAPI,
) -> None:
    collection = seed(api)
    detail = api.page(detail_url(collection))
    assert identities(detail) == list(SAVED_ORDER)
    assert values(detail, "member-status") == [
        "unscreened",
        "include",
        "unscreened",
        "exclude",
        "unscreened",
        "unsure",
    ]
    assert "in saved membership order" in detail.text()
    form = api.page(href(detail, "screen-member-1"), referrer_policy="same-origin")
    assert not form.by_class("priority-score") and not form.by_class("priority-rank")
    assert urlsplit(href(form, "screening-priority")).path == PRIORITY


def test_cold_start_keeps_saved_order_without_scores(api: ExplorerAPI) -> None:
    collection = seed(api, decide=False)
    root = api.page(queue_url(collection))
    assert identities(root) == list(SAVED_ORDER)
    assert values(root, "model-status") == ["needs_include_and_exclude"]
    assert not root.by_class("priority-score") and not root.by_class("top-term")
    assert len(root.by_class("priority-unscored")) == len(SAVED_ORDER)
    assert "saved collection order" in root.text()


@pytest.mark.parametrize("prefix,mounted", [("/library", False), ("/prefix/library", True)])
def test_mounted_queue_keeps_local_links(api: ExplorerAPI, prefix: str, mounted: bool) -> None:
    collection = seed(api)
    application = api.app
    if mounted:
        application = FastAPI()
        application.mount("/library", api.app)
    with TestClient(application, root_path="/prefix" if mounted else prefix) as client:
        detail = ExplorerDOM(client.get(prefix + detail_url(collection)).text).root
        response = client.get(href(detail, "screening-priority"))
        assert response.status_code == 200
        root = ExplorerDOM(response.text).root
        for link in root.by_tag("a"):
            parts = urlsplit(link.attrs["href"])
            assert not parts.netloc and not parts.scheme
            assert not parts.path or parts.path.startswith(prefix + "/")
        assert client.get(href(root, "screen-member-1")).status_code == 200


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"collection_revision": "1"},
        {"collection_id": "col_bad", "collection_revision": "1"},
        {"collection_id": "COLLECTION", "collection_revision": "1.0"},
        {"collection_id": "COLLECTION", "collection_revision": "+1"},
        {"collection_id": "COLLECTION", "collection_revision": "0"},
        {"collection_id": "COLLECTION", "collection_revision": "1", "limit": "5"},
        {"collection_id": "COLLECTION", "collection_revision": "1", "collection_return": "{}"},
        {
            "collection_id": "COLLECTION",
            "collection_revision": "1",
            "collection_return": "[" * 2000,
        },
    ],
)
def test_invalid_selection_fails_without_writes(api: ExplorerAPI, params: dict[str, str]) -> None:
    collection = seed(api)
    query = {
        key: collection.collection_id if value == "COLLECTION" else value
        for key, value in params.items()
    }
    before = api.path.read_bytes()
    failure(api.client.get(PRIORITY, params=query), 422, "Use an exact collection_id")
    assert api.path.read_bytes() == before


def test_wire_level_ambiguity_and_bounds_fail(api: ExplorerAPI) -> None:
    collection = seed(api)
    url = queue_url(collection)
    failure(api.client.get(url + "&collection_revision=1"), 422, "Use an exact collection_id")
    failure(api.client.get(url + "&x=%ff"), 422, "Use an exact collection_id")
    failure(api.client.get(url + "&x=%"), 422, "Use an exact collection_id")
    failure(api.client.request("GET", url, content=b"body"), 422, "Use an exact collection_id")
    other = api.container.paper_collections.create(name="Other", document_ids=["grounded"])
    mismatched = queue_url(
        collection,
        collection_return=json.dumps(
            {"collection_id": other.collection_id, "collection_revision": 1}
        ),
    )
    failure(api.client.get(mismatched), 422, "Use an exact collection_id")


def test_missing_stale_and_unavailable_collections_fail_explicitly(
    api: ExplorerAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    collection = seed(api)
    root = failure(
        api.client.get(
            PRIORITY, params={"collection_id": "col_" + "0" * 32, "collection_revision": 1}
        ),
        404,
        "The exact collection no longer exists",
    )
    assert href(root, "back-to-collections") == "/explore/collections"
    failure(
        api.client.get(queue_url(collection, collection_revision=2)),
        409,
        "The collection revision changed",
    )
    monkeypatch.setattr(
        api.container.paper_screening,
        "_snapshot",
        Mock(side_effect=CollectionError("screening_storage_unavailable", "PRIVATE_", 503)),
    )
    failure(api.client.get(queue_url(collection)), 503, "Screening storage is unavailable")


@pytest.mark.parametrize("method", ["POST", "PUT", "DELETE"])
def test_queue_is_get_only_and_never_records(api: ExplorerAPI, method: str) -> None:
    collection = seed(api)
    before = api.path.read_bytes()
    response = api.client.request(
        method,
        queue_url(collection),
        data={"decision": "include", "reason": "Never recorded."},
        headers={"origin": "http://testserver", "sec-fetch-site": "same-origin"},
    )
    failure(response, 405, "The prioritized queue is read with GET")
    assert response.headers["allow"] == "GET"
    assert api.path.read_bytes() == before
    form = api.client.put("/explore/screening")
    assert form.status_code == 405 and form.headers["allow"] == "GET, POST"


@pytest.mark.parametrize("size,status", [(65_536, 422), (65_537, 413)])
async def test_query_byte_bound_at_asgi_boundary(api: ExplorerAPI, size: int, status: int) -> None:
    before = api.path.read_bytes()
    scope: Scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "path": PRIORITY,
        "raw_path": PRIORITY.encode(),
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
    failure(
        response,
        status,
        "Use an exact collection_id" if status == 422 else "The URL exceeds 65536 bytes",
    )
    assert api.path.read_bytes() == before


def test_no_training_terms_and_fully_decided_queues_are_explicit(api: ExplorerAPI) -> None:
    api.container.document_store.add_documents(
        [
            Document(document_id=identifier, title=title, text="", source="synthetic:edge")
            for identifier, title in (("a", "The of and"), ("b", "Is it 2024"), ("c", "Grounded"))
        ],
        [],
    )
    collection = api.container.paper_collections.create(name="Edge", document_ids=["a", "b", "c"])
    for identifier, decision in (("a", "include"), ("b", "exclude")):
        api.container.paper_screening.submit(
            collection.collection_id,
            identifier,
            ScreeningSubmission(
                collection_revision=1,
                expected_decision_revision=0,
                decision=decision,  # type: ignore[arg-type]
                reason="Synthetic human opinion.",
            ),
        )
    root = api.page(queue_url(collection))
    assert values(root, "model-status") == ["no_training_terms"]
    assert identities(root) == ["c"] and not root.by_class("priority-score")
    api.container.paper_screening.submit(
        collection.collection_id,
        "c",
        ScreeningSubmission(
            collection_revision=1,
            expected_decision_revision=0,
            decision="include",
            reason="Synthetic human opinion.",
        ),
    )
    done = api.page(queue_url(collection))
    assert not done.by_class("priority-item")
    assert "No unscreened or stale members remain" in done.text()


def test_storage_html_size_and_unrepresentable_text_fail_closed(
    api: ExplorerAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    collection = seed(api)
    url = queue_url(collection)
    response = api.client.get(url)
    monkeypatch.setattr(explorer, "MAX_HTML_BYTES", len(response.content))
    assert api.client.get(url).content == response.content
    monkeypatch.setattr(explorer, "MAX_HTML_BYTES", len(response.content) - 1)
    failure(api.client.get(url), 413, "The complete escaped page exceeds 1 MiB")
    monkeypatch.undo()
    real = ScreeningPriority.model_validate_json(
        api.client.get(
            f"/collections/{collection.collection_id}/screening-priority",
            params={"collection_revision": 1},
        ).content
    )
    item = real.items[0].model_copy(update={"title": "nul\0title"})
    monkeypatch.setattr(
        screening_explorer,
        "prioritize_screening",
        Mock(return_value=real.model_copy(update={"items": (item, *real.items[1:])})),
    )
    failure(api.client.get(url), 409, "Saved text cannot be displayed faithfully")
    monkeypatch.setattr(
        screening_explorer,
        "prioritize_screening",
        Mock(side_effect=sqlite3.OperationalError("PRIVATE_database_detail")),
    )
    failure(api.client.get(url), 503, "Screening storage is unavailable")
