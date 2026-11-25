"""GET-only JSON stopping-rule assessment; strict parameters, never a decision or stop."""

import socket
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import Mock
from urllib.parse import quote, urlencode

import httpx
import pytest
from fastapi.testclient import TestClient
from scripts.demo_corpus_explorer import browsing_guards
from scripts.demo_evidence_export import offline_settings

from api import screening_stopping as stopping_api
from api.application import create_app
from api.dependencies import AppContainer
from retrieval.models import Document
from storage.paper_collections import MAX_REVISION, CollectionError, PaperCollection
from storage.paper_screening import ScreeningSubmission
from storage.screening_stopping import ScreeningStoppingAssessment, assess_screening_stopping

MEMBERS = ("seed-include", "seed-exclude", "candidate/slash?#", "candidate-4", "candidate-5")


@dataclass
class StoppingAPI:
    client: TestClient
    container: AppContainer
    database: Path
    collection: PaperCollection

    @property
    def path(self) -> str:
        return f"/collections/{self.collection.collection_id}/screening-stopping"

    def decide(self, identifier: str, decision: str) -> None:
        self.container.paper_screening.submit(
            self.collection.collection_id,
            identifier,
            ScreeningSubmission.model_validate(
                {
                    "collection_revision": 1,
                    "expected_decision_revision": 0,
                    "decision": decision,
                    "reason": "Synthetic human opinion for an API test.",
                }
            ),
        )


def denied(*args: object, **kwargs: object) -> None:
    raise AssertionError("Stopping rules must not use the network.")


@pytest.fixture
def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[StoppingAPI]:
    for target, method in (
        (httpx.HTTPTransport, "handle_request"),
        (httpx.AsyncHTTPTransport, "handle_async_request"),
        (socket, "create_connection"),
        (socket.socket, "connect"),
    ):
        monkeypatch.setattr(target, method, denied)
    database = tmp_path / "stopping-api.sqlite3"
    app = create_app(offline_settings(database))
    with TestClient(app) as client:
        container: AppContainer = app.state.container
        container.document_store.add_documents(
            [
                Document(document_id=identifier, title=identifier, text="x", source="synthetic")
                for identifier in MEMBERS
            ],
            [],
        )
        collection = container.paper_collections.create(
            name="Synthetic stopping review", document_ids=list(MEMBERS)
        )
        yield StoppingAPI(client, container, database, collection)


def read(api: StoppingAPI, query: str = "collection_revision=1") -> httpx.Response:
    response = api.client.get(api.path + "?" + query)
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    return response


def failure(response: httpx.Response, status: int, code: str) -> None:
    assert response.status_code == status, response.text
    assert response.headers["cache-control"] == "no-store"
    assert response.json()["detail"]["code"] == code
    assert set(response.json()["detail"]) <= {"code", "message"}


def test_assessment_matches_python_and_never_writes(
    api: StoppingAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    api.decide("seed-include", "include")
    api.decide("seed-exclude", "exclude")
    before = api.database.read_bytes()
    submit = Mock(side_effect=AssertionError("Stopping rules must not record decisions."))
    monkeypatch.setattr(api.container.paper_screening, "submit", submit)
    with browsing_guards(api.container) as guards:
        response = read(api)
        repeated = read(api)
    assert response.status_code == 200 and response.content == repeated.content
    assert not any(guard.call_count for guard in guards.values()) and not submit.called
    assert api.database.read_bytes() == before
    assert api.container.event_log.list_events() == []
    result = ScreeningStoppingAssessment.model_validate_json(response.content)
    assert result == assess_screening_stopping(
        api.container.paper_screening, api.collection.collection_id, collection_revision=1
    )
    body = response.json()
    assert body["advisory_only"] is True and body["schema_version"] == "1.0"
    assert [item["document_id"] for item in body["timeline"]["items"]] == list(MEMBERS[:2])
    assert body["consecutive_irrelevant"]["status"] == "met"
    assert body["recall_test"]["status"] == "not_started"
    assert body["recall_test"]["method"] == "buscar-hypergeometric-v1"
    assert body["assumptions"] and body["limitations"]


def test_every_parameter_reaches_the_assessment(api: StoppingAPI) -> None:
    api.decide("seed-include", "include")
    api.decide("seed-exclude", "exclude")
    urn = read(api).json()["recall_test"]["urn_sha256"]
    query = urlencode(
        [
            ("collection_revision", "1"),
            ("target_recall", "0.9"),
            ("confidence", "0.8"),
            ("irrelevant_run_percent", "40"),
            ("min_screened_percent", "20"),
            ("estimated_relevant", "1"),
            ("key_document_id", "seed-include"),
            ("key_document_id", "candidate/slash?#"),
            ("sample_seed", "20261124"),
            ("sample_after", "2"),
            ("sample_urn_sha256", urn),
        ]
    )
    body = read(api, query).json()
    parameters = body["parameters"]
    assert (parameters["target_recall"], parameters["confidence"]) == (0.9, 0.8)
    assert (parameters["irrelevant_run"], parameters["irrelevant_run_basis"]) == (
        2,
        "percent_of_collection",
    )
    assert parameters["key_document_ids"] == ["seed-include", "candidate/slash?#"]
    assert body["safe_checkpoint"]["key_documents_not_included"] == ["candidate/slash?#"]
    test = body["recall_test"]
    assert (test["sample_seed"], test["sample_after"], test["urn_size"]) == (20261124, 2, 3)
    assert test["alpha"] == pytest.approx(0.2) and test["status"] == "not_met"
    assert sorted(test["next_document_ids"]) == sorted(MEMBERS[2:])


@pytest.mark.parametrize(
    "query",
    [
        "",
        "collection_revision=",
        "collection_revision=0",
        "collection_revision=%2B1",
        "collection_revision=%EF%BC%91",
        "collection_revision=1&collection_revision=1",
        f"collection_revision={MAX_REVISION + 1}",
        "collection_revision=1&limit=5",
        "collection_revision=1&target_recall=1e-1",
        "collection_revision=1&target_recall=.95",
        "collection_revision=1&target_recall=0.9500001",
        "collection_revision=1&target_recall=0",
        "collection_revision=1&confidence=1",
        "collection_revision=1&confidence=0.9&confidence=0.9",
        "collection_revision=1&irrelevant_run=-1",
        "collection_revision=1&irrelevant_run=2&irrelevant_run_percent=5",
        "collection_revision=1&min_screened_percent=0",
        "collection_revision=1&sample_seed=1",
        "collection_revision=1&sample_after=0",
        "collection_revision=1&sample_seed=1&sample_after=0&sample_urn_sha256=xyz",
        "collection_revision=1&key_document_id=seed-include&key_document_id=seed-include",
        "collection_revision=1&key_document_id=%20seed-include",
        "collection_revision=1&key_document_id=" + "x" * 129,
        "collection_revision=1&" + "&".join(f"key_document_id=k{index}" for index in range(101)),
        "collection_revision=1&x=%00",
        "collection_revision=1&key_document_id=%ff",
        "collection_revision=1&pad=" + "x" * 33_000,
    ],
)
def test_invalid_requests_use_the_stable_envelope(api: StoppingAPI, query: str) -> None:
    failure(read(api, query), 422, "invalid_screening_stopping_request")


@pytest.mark.parametrize("collection_id", ["col_bad", "COL_" + "0" * 32, "col_" + "g" * 32])
def test_invalid_collection_ids_are_rejected(api: StoppingAPI, collection_id: str) -> None:
    failure(
        api.client.get(f"/collections/{collection_id}/screening-stopping?collection_revision=1"),
        422,
        "invalid_screening_stopping_request",
    )


def test_snapshot_dependent_failures_keep_the_collection_envelope(
    api: StoppingAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    failure(
        api.client.get(f"/collections/col_{'0' * 32}/screening-stopping?collection_revision=1"),
        404,
        "collection_not_found",
    )
    failure(read(api, "collection_revision=2"), 409, "collection_revision_conflict")
    failure(
        read(api, "collection_revision=1&key_document_id=" + quote("not-a-member")),
        422,
        "stopping_key_document_not_member",
    )
    failure(
        read(api, "collection_revision=1&estimated_relevant=6"),
        422,
        "stopping_parameter_out_of_range",
    )
    failure(
        read(api, "collection_revision=1&sample_seed=1&sample_after=1"),
        422,
        "stopping_parameter_out_of_range",
    )
    for code, status in (
        ("stopping_sample_boundary_tied", 409),
        ("stopping_sample_urn_changed", 409),
        ("screening_stopping_too_large", 413),
        ("screening_storage_unavailable", 503),
    ):
        monkeypatch.setattr(
            stopping_api,
            "assess_screening_stopping",
            Mock(side_effect=CollectionError(code, "Synthetic public diagnostic.", status)),
        )
        failure(read(api), status, code)


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
def test_only_get_is_allowed_and_nothing_is_recorded(api: StoppingAPI, method: str) -> None:
    before = api.database.read_bytes()
    response = api.client.request(
        method,
        api.path + "?collection_revision=1",
        json={
            "collection_revision": 1,
            "expected_decision_revision": 0,
            "decision": "exclude",
            "reason": "Must never be recorded.",
        },
    )
    failure(response, 405, "screening_stopping_method_not_allowed")
    assert response.headers["allow"] == "GET"
    assert api.database.read_bytes() == before
