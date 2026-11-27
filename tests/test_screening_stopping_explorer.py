"""Native read-only stopping-rules page beside the existing screening and priority pages."""

import json
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx
import pytest
from scripts.demo_corpus_explorer import browsing_guards

from api.screening_explorer import STOPPING_PATH
from retrieval.models import Document
from storage.paper_collections import PaperCollection
from storage.paper_screening import ScreeningSubmission
from tests.test_collection_explorer import detail_url
from tests.test_corpus_explorer import ExplorerAPI, ExplorerDOM, assert_headers, href
from tests.test_corpus_explorer import api as api
from tests.test_evidence_html import Node

HOSTILE = '"><script>alert("id")</script><img src=x onerror="y"> & [x](javascript:1)'
MEMBERS = ("seed-include", "seed-exclude", HOSTILE, "pending-1", "pending-2", "pending-3")


def seed(api: ExplorerAPI) -> PaperCollection:
    api.container.document_store.add_documents(
        [
            Document(document_id=identifier, title=f"Title {index}", text="x", source="s")
            for index, identifier in enumerate(MEMBERS)
        ],
        [],
    )
    collection = api.container.paper_collections.create(
        name="Synthetic <stopping> review", document_ids=MEMBERS
    )
    for identifier, decision in (("seed-include", "include"), ("seed-exclude", "exclude")):
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


def page_url(collection: PaperCollection, **params: str | int) -> str:
    return (
        STOPPING_PATH
        + "?"
        + urlencode(
            {
                "collection_id": collection.collection_id,
                "collection_revision": collection.revision,
                **params,
            }
        )
    )


def form_fields(root: Node, form_id: str) -> dict[str, str]:
    form = root.by_id(form_id)
    fields = {
        node.attrs["name"]: node.attrs.get("value") or ""
        for node in form.walk()
        if node.tag == "input" and node.attrs.get("name")
    }
    fields.update({node.attrs["name"]: node.text() for node in form.by_tag("textarea")})
    return fields


def test_collection_workbench_and_priority_pages_link_to_the_stopping_rules(
    api: ExplorerAPI,
) -> None:
    collection = seed(api)
    detail = api.page(detail_url(collection))
    link = href(detail, "screening-stopping")
    assert urlsplit(link).path == STOPPING_PATH
    root = api.page(link)
    assert root.by_id("back-to-collection")
    priority = api.page(href(root, "stopping-priority"))
    assert urlsplit(href(priority, "priority-stopping")).path == STOPPING_PATH
    signal = priority.by_id("stopping-signal")
    assert "No stopping signal is shown" in signal.text()
    assert "advisory stopping rules" in signal.text()
    workbench = api.page(
        "/explore/screening",
        referrer_policy="same-origin",
        collection_id=collection.collection_id,
        collection_revision=1,
        document_id="pending-1",
    )
    assert urlsplit(href(workbench, "screening-stopping")).path == STOPPING_PATH


def test_page_matches_the_json_assessment_and_escapes_saved_text(api: ExplorerAPI) -> None:
    collection = seed(api)
    before = api.path.read_bytes()
    with browsing_guards(api.container) as guards:
        response = api.client.get(page_url(collection))
    assert response.status_code == 200
    assert_headers(response)
    assert not any(guard.call_count for guard in guards.values())
    assert api.path.read_bytes() == before
    assert "<script>alert" not in response.text and "&lt;stopping&gt;" in response.text
    root = ExplorerDOM(response.text).root
    assert "READ-ONLY / SCREENING STOPPING RULES" in root.text()
    assert "Advisory only" in root.by_id("stopping-guarantee").text()
    expected = api.client.get(urlsplit(href(root, "stopping-json")).path + "?collection_revision=1")
    body = expected.json()
    assert root.by_id("consecutive-status").text() == body["consecutive_irrelevant"]["status"]
    assert root.by_id("consecutive-run").text() == "1"
    assert root.by_id("safe-status").text() == body["safe_checkpoint"]["status"] == "incomplete"
    assert root.by_id("recall-status").text() == "not_started"
    assert root.by_id("count-screened").text() == "2"
    assert root.by_id("order-reliable").text() == "yes"
    assert [node.text() for node in root.by_class("criterion-status")] == [
        criterion["status"] for criterion in body["safe_checkpoint"]["criteria"]
    ]
    assert len(root.by_id("assumptions").by_tag("li")) == len(body["assumptions"])
    assert len(root.by_id("limitations").by_tag("li")) == len(body["limitations"])
    assert not root.by_class("next-member")


def test_the_seeded_sample_round_trips_through_native_get_forms(api: ExplorerAPI) -> None:
    collection = seed(api)
    root = api.page(page_url(collection))
    fields = form_fields(root, "start-sample")
    assert fields["sample_after"] == "2" and len(fields["sample_urn_sha256"]) == 64
    assert fields["sample_seed"] == ""
    fields["sample_seed"] = "20261124"
    started = api.page(STOPPING_PATH + "?" + urlencode(fields))
    assert started.by_id("recall-status").text() == "not_met"
    identity = json.loads(started.by_class("sample-identity")[0].text())
    assert identity == {
        "sample_seed": 20261124,
        "sample_after": 2,
        "sample_urn_sha256": fields["sample_urn_sha256"],
    }
    body = api.client.get(href(started, "stopping-json")).json()["recall_test"]
    shown = [json.loads(node.by_tag("code")[0].text()) for node in started.by_class("next-member")]
    assert shown == body["next_document_ids"] and sorted(shown) == sorted(MEMBERS[2:])
    first = parse_qs(urlsplit(href(started, "screen-next-1")).query)
    assert urlsplit(href(started, "screen-next-1")).path == "/explore/screening"
    assert first["document_id"] == [shown[0]] and "collection_return" in first
    assert started.by_id("p-value").text() == "1"
    assert started.by_id("additional-needed").text() == str(body["additional_irrelevant_needed"])
    reload = parse_qs(urlsplit(href(started, "stopping-reload")).query)
    assert reload["sample_seed"] == ["20261124"] and reload["sample_after"] == ["2"]


def test_parameter_form_keeps_exact_key_papers_and_choices(api: ExplorerAPI) -> None:
    collection = seed(api)
    root = api.page(
        page_url(
            collection,
            key_document_ids=f"seed-include\r\n\r\n{HOSTILE}",
            irrelevant_run=2,
            confidence="0.9",
            estimated_relevant=1,
        )
    )
    fields = form_fields(root, "stopping-parameters")
    assert fields["key_document_ids"] == f"seed-include\n{HOSTILE}"
    assert (fields["irrelevant_run"], fields["irrelevant_run_percent"]) == ("2", "")
    assert (fields["confidence"], fields["target_recall"]) == ("0.9", "0.95")
    query = parse_qs(urlsplit(href(root, "stopping-json")).query)
    assert query["key_document_id"] == ["seed-include", HOSTILE]
    assert query["irrelevant_run"] == ["2"] and query["confidence"] == ["0.9"]
    missing = [node.by_tag("code")[0].text() for node in root.by_class("missing-key")]
    assert missing == [json.dumps(HOSTILE)]
    assert root.by_id("consecutive-threshold").text() == "2"


@pytest.mark.parametrize(
    ("params", "status", "fragment"),
    [
        ({"collection_revision": "x"}, 422, "decimals such as 0.95"),
        ({"unknown": "1"}, 422, "decimals such as 0.95"),
        ({"irrelevant_run": "2", "irrelevant_run_percent": "5"}, 422, "decimals such as 0.95"),
        ({"target_recall": "1e-1"}, 422, "decimals such as 0.95"),
        ({"sample_seed": "1"}, 422, "decimals such as 0.95"),
        ({"key_document_ids": "missing"}, 422, "not a member"),
        ({"estimated_relevant": "99"}, 422, "exceeds the collection"),
        ({"collection_revision": "2"}, 409, "HTTP 409"),
    ],
)
def test_failures_render_bounded_escaped_error_pages(
    api: ExplorerAPI, params: dict[str, str], status: int, fragment: str
) -> None:
    collection = seed(api)
    response = api.client.get(page_url(collection, **params))
    assert response.status_code == status, response.text[:500]
    assert_headers(response)
    root = ExplorerDOM(response.text).root
    assert fragment in root.text() and "READ-ONLY / SCREENING STOPPING RULES" in root.text()
    assert not root.by_tag("form") and "missing" not in response.text.split("<main")[0]


def test_missing_collection_bodies_and_writes_are_refused(api: ExplorerAPI) -> None:
    collection = seed(api)
    missing: httpx.Response = api.client.get(
        STOPPING_PATH
        + "?"
        + urlencode({"collection_id": "col_" + "0" * 32, "collection_revision": 1})
    )
    assert missing.status_code == 404
    with_body = api.client.request("GET", page_url(collection), content=b"x=1")
    assert with_body.status_code == 422
    before = api.path.read_bytes()
    posted = api.client.post(page_url(collection), data={"decision": "exclude"})
    assert posted.status_code == 405 and posted.headers["allow"] == "GET"
    assert "read with GET" in ExplorerDOM(posted.text).root.text()
    assert api.path.read_bytes() == before
