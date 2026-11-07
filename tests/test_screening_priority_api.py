"""GET-only JSON ordering of the human screening queue; never a decision or suggestion."""

import json
import socket
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import Mock
from urllib.parse import quote

import httpx
import pytest
from fastapi.testclient import TestClient
from scripts.demo_corpus_explorer import browsing_guards
from scripts.demo_evidence_export import offline_settings

from api import screening_priority as priority_api
from api.application import create_app
from api.dependencies import AppContainer
from retrieval.models import Document
from storage.paper_collections import MAX_REVISION, CollectionError, PaperCollection
from storage.paper_screening import ScreeningSubmission
from storage.screening_priority import ScreeningPriority, prioritize_screening

PAPERS = (
    ("seed-include", "Citation grounding for scientific answers", "Evidence passages cited."),
    ("seed-exclude", "Battery cathode coatings", "Lithium cells degrade."),
    ("candidate-battery", "Cathode fade in battery cells", "Charge cycles."),
    ("candidate-grounded", "Grounded citation answers", "Evidence passages support answers."),
    ("candidate/slash?#", "Evidence grounding <script>alert(1)</script>", "Citation checks."),
)


@dataclass
class PriorityAPI:
    client: TestClient
    container: AppContainer
    database: Path
    collection: PaperCollection

    @property
    def path(self) -> str:
        return f"/collections/{self.collection.collection_id}/screening-priority"

    def decide(self, identifier: str, decision: str, expected: int = 0) -> None:
        self.container.paper_screening.submit(
            self.collection.collection_id,
            identifier,
            ScreeningSubmission.model_validate(
                {
                    "collection_revision": self.collection.revision,
                    "expected_decision_revision": expected,
                    "decision": decision,
                    "reason": "Synthetic human opinion for an API test.",
                }
            ),
        )


def denied(*args: object, **kwargs: object) -> None:
    raise AssertionError("Prioritization must not use the network.")


@pytest.fixture
def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[PriorityAPI]:
    for target, method in (
        (httpx.HTTPTransport, "handle_request"),
        (httpx.AsyncHTTPTransport, "handle_async_request"),
        (socket, "create_connection"),
        (socket.socket, "connect"),
    ):
        monkeypatch.setattr(target, method, denied)
    database = tmp_path / "priority-api.sqlite3"
    app = create_app(offline_settings(database))
    with TestClient(app) as client:
        container: AppContainer = app.state.container
        container.document_store.add_documents(
            [
                Document(document_id=i, title=t, text=x, source="synthetic:priority-api")
                for i, t, x in PAPERS
            ],
            [],
        )
        collection = container.paper_collections.create(
            name="Synthetic API review", document_ids=[paper[0] for paper in PAPERS]
        )
        yield PriorityAPI(client, container, database, collection)


def read(api: PriorityAPI, **params: object) -> httpx.Response:
    response = api.client.get(
        api.path, params={"collection_revision": api.collection.revision, **params}
    )
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "content-disposition" not in response.headers
    return response


def failure(response: httpx.Response, status: int, code: str) -> None:
    assert response.status_code == status, response.text
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.json()["detail"]["code"] == code
    assert set(response.json()["detail"]) <= {"code", "message"}
    assert "items" not in response.json()


def test_cold_start_returns_saved_order_counts_and_no_scores(api: PriorityAPI) -> None:
    response = read(api)
    assert response.status_code == 200
    body = response.json()
    assert body["model_status"] == "needs_include_and_exclude"
    assert body["ordering"] == "collection_order"
    assert [item["document_id"] for item in body["items"]] == [paper[0] for paper in PAPERS]
    assert all(item["score"] is None and item["top_terms"] == [] for item in body["items"])
    assert body["counts"] == {"unscreened": 5, "include": 0, "exclude": 0, "unsure": 0, "stale": 0}
    assert body["training_documents"] == body["feature_terms"] == 0
    api.decide("seed-include", "include")
    body = read(api).json()
    assert body["model_status"] == "needs_include_and_exclude"
    assert (body["training_relevant"], body["training_irrelevant"]) == (1, 0)


def test_ready_ordering_matches_python_and_explains_terms_without_writes(
    api: PriorityAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    api.decide("seed-include", "include")
    api.decide("seed-exclude", "exclude")
    api.decide("candidate-battery", "unsure")
    before = api.database.read_bytes()
    with closing(sqlite3.connect(api.database)) as connection:
        revisions = connection.execute(
            "SELECT document_id, record FROM paper_screening_decisions ORDER BY document_id"
        ).fetchall()
    submit = Mock(side_effect=AssertionError("Prioritization must not record decisions."))
    monkeypatch.setattr(api.container.paper_screening, "submit", submit)
    with browsing_guards(api.container) as guards:
        response = read(api)
        repeated = read(api)
    assert response.status_code == 200 and response.content == repeated.content
    assert not any(guard.call_count for guard in guards.values()) and not submit.called
    assert api.database.read_bytes() == before
    with closing(sqlite3.connect(api.database)) as connection:
        assert (
            connection.execute(
                "SELECT document_id, record FROM paper_screening_decisions ORDER BY document_id"
            ).fetchall()
            == revisions
        )
    assert api.container.event_log.list_events() == []
    result = ScreeningPriority.model_validate_json(response.content)
    assert result == prioritize_screening(
        api.container.paper_screening, api.collection.collection_id, collection_revision=1
    )
    body = response.json()
    assert body["model_status"] == "ready" and body["ordering"] == "most_likely_relevant_first"
    assert body["training_documents"] == 2 and body["feature_terms"] > 0
    assert [item["document_id"] for item in body["items"]] == [
        "candidate-grounded",
        "candidate/slash?#",
    ]
    assert [item["rank"] for item in body["items"]] == [1, 2]
    assert body["items"][1]["title"] == PAPERS[4][1]
    assert body["unsure_items"][0]["document_id"] == "candidate-battery"
    top = body["items"][0]
    assert 0.5 < top["score"] <= 1 and top["log_odds"] > 0
    assert {"grounded", "citation", "answers"} & {term["term"] for term in top["top_terms"]}
    assert body["notes"][0].startswith("Ordering only: no decision is recorded")
    assert any("No stopping signal" in note for note in body["notes"])
    assert body["method"]["name"] == "tfidf-multinomial-naive-bayes-v1"
    assert body["method"]["alpha"] == 1.0 and body["method"]["max_text_characters"] == 4000
    assert "decision" not in json.dumps(body["items"])


@pytest.mark.parametrize(
    "query",
    [
        "",
        "collection_revision=",
        "collection_revision=0",
        "collection_revision=-1",
        "collection_revision=%2B1",
        "collection_revision=1.0",
        "collection_revision=%EF%BC%91",
        "collection_revision=1&collection_revision=1",
        "collection_revision=1&limit=5",
        "collection_revision=1&format=json",
        f"collection_revision={MAX_REVISION + 1}",
        "collection_revision=" + "1" * 20,
        "collection_revision=1&x=%00",
        "collection_revision=1&x=%ff",
        "collection_revision=1&pad=" + "x" * 16_400,
    ],
)
def test_invalid_requests_use_the_stable_envelope(api: PriorityAPI, query: str) -> None:
    failure(api.client.get(api.path + "?" + query), 422, "invalid_screening_priority_request")


@pytest.mark.parametrize("collection_id", ["col_bad", "COL_" + "0" * 32, "col_" + "g" * 32])
def test_invalid_collection_ids_are_rejected(api: PriorityAPI, collection_id: str) -> None:
    failure(
        api.client.get(
            f"/collections/{collection_id}/screening-priority", params={"collection_revision": 1}
        ),
        422,
        "invalid_screening_priority_request",
    )


def test_missing_collection_stale_revision_and_invalid_saved_data_fail(api: PriorityAPI) -> None:
    failure(
        api.client.get(
            f"/collections/col_{'0' * 32}/screening-priority", params={"collection_revision": 1}
        ),
        404,
        "collection_not_found",
    )
    failure(read(api, collection_revision=2), 409, "collection_revision_conflict")
    with closing(sqlite3.connect(api.database)) as connection, connection:
        connection.execute(
            "INSERT INTO paper_screening_decisions VALUES (?, ?, ?)",
            (api.collection.collection_id, "seed-include", "{not json"),
        )
    failure(read(api), 409, "invalid_screening_record")


def test_storage_failures_and_size_limits_keep_the_collection_envelope(
    api: PriorityAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    for code, status in (
        ("collection_storage_error", 503),
        ("screening_priority_too_large", 413),
        ("collection_documents_missing", 409),
    ):
        monkeypatch.setattr(
            priority_api,
            "prioritize_screening",
            Mock(side_effect=CollectionError(code, "Synthetic public diagnostic.", status)),
        )
        response = read(api)
        failure(response, status, code)
        assert response.json()["detail"]["message"] == "Synthetic public diagnostic."


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
def test_only_get_is_allowed_and_put_never_records_a_decision(
    api: PriorityAPI, method: str
) -> None:
    before = api.database.read_bytes()
    response = api.client.request(
        method,
        api.path,
        params={"collection_revision": 1},
        json={
            "collection_revision": 1,
            "expected_decision_revision": 0,
            "decision": "include",
            "reason": "Must never be recorded.",
        },
    )
    failure(response, 405, "screening_priority_method_not_allowed")
    assert response.headers["allow"] == "GET"
    assert api.database.read_bytes() == before


def test_route_sits_outside_the_screening_put_wildcard(api: PriorityAPI) -> None:
    api.container.document_store.add_documents(
        [Document(document_id="priority", title="A paper named priority", text="", source="s")],
        [],
    )
    collection = api.container.paper_collections.replace(
        api.collection.collection_id,
        name=api.collection.name,
        document_ids=[*api.collection.document_ids, "priority"],
        expected_revision=1,
    )
    path = f"/collections/{collection.collection_id}/screening-priority"
    assert api.client.put(path, params={"collection_revision": 2}).status_code == 405
    queue = api.client.get(
        f"/collections/{collection.collection_id}/screening",
        params={"collection_revision": 2, "status": "unscreened", "limit": 100},
    ).json()
    assert queue["counts"]["unscreened"] == 6
    response = api.client.put(
        f"/collections/{collection.collection_id}/screening/{quote('priority', safe='')}",
        json={
            "collection_revision": 2,
            "expected_decision_revision": 0,
            "decision": "unsure",
            "reason": "Explicit human opinion through the existing route.",
        },
    )
    assert response.status_code == 200
    body = api.client.get(path, params={"collection_revision": 2}).json()
    assert [item["document_id"] for item in body["unsure_items"]] == ["priority"]


def test_openapi_documents_the_read_only_contract(api: PriorityAPI) -> None:
    operation = api.client.get("/openapi.json").json()["paths"][
        "/collections/{collection_id}/screening-priority"
    ]
    assert set(operation) == {"get"}
    responses = operation["get"]["responses"]
    assert {"200", "404", "405", "409", "413", "422", "503"} <= set(responses)
    schema = responses["200"]["content"]["application/json"]["schema"]
    assert schema["$ref"].endswith("/ScreeningPriority")
    assert "never record or suggest" in operation["get"]["description"]
