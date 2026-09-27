"""Offline HTTP acceptance for append-only judgments on frozen saved answers."""

import json
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn
from urllib.parse import quote
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from scripts.demo_evidence_export import offline_settings

from agent.models import AgentState, StateTransition
from api.application import create_app
from api.dependencies import AppContainer
from llm.schemas import LLMRequest, LLMResponse


def no_live_work(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("Reviews must not generate, retrieve, or use live HTTP")


@dataclass
class ReviewAPI:
    application: FastAPI
    client: TestClient
    database_path: Path
    run_id: str
    chunk_id: str

    @property
    def container(self) -> AppContainer:
        container: AppContainer = self.application.state.container
        return container

    @property
    def path(self) -> str:
        return f"/runs/{self.run_id}/reviews"


@pytest.fixture
def review_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[ReviewAPI]:
    """Create a DONE run through the actual app factory and fake-adapter pipeline."""
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", no_live_work)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", no_live_work)
    database_path = tmp_path / "reviews.sqlite3"
    application = create_app(offline_settings(database_path))
    with TestClient(application) as client:
        ingested = client.post(
            "/ingest/text",
            json={
                "title": "Synthetic review evidence",
                "text": "Synthetic data, not a paper. GraphRAG connects research entities.",
                "source": "synthetic:answer-review",
            },
        )
        assert ingested.status_code == 200
        completed = client.post("/query", json={"query": "What does GraphRAG connect?"})
        assert completed.status_code == 200
        result = completed.json()["result"]
        assert result["state"] == "DONE"
        exported = client.get(f"/runs/{result['run_id']}/export")
        assert exported.status_code == 200
        assert exported.json()["generation"]["provider"] == "fake"
        yield ReviewAPI(
            application,
            client,
            database_path,
            result["run_id"],
            exported.json()["snapshot"]["sources"][0]["chunk"]["chunk_id"],
        )


def review_body(api: ReviewAPI, **changes: object) -> dict[str, Any]:
    return {
        "review_id": str(uuid4()),
        "decision": "needs_revision",
        "comment": "Explain the limits of this synthetic evidence before using the answer.",
        "cited_chunk_ids": [api.chunk_id],
        **changes,
    }


def raw_events(api: ReviewAPI) -> list[tuple[object, ...]]:
    with sqlite3.connect(api.database_path) as connection:
        return connection.execute("SELECT * FROM agent_events ORDER BY id").fetchall()


def forbid_runtime_work(container: AppContainer, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(container.runner, "run", no_live_work)
    monkeypatch.setattr(container.llm, "generate", no_live_work)
    monkeypatch.setattr(container.hybrid_retriever, "retrieve", no_live_work)
    monkeypatch.setattr(container.document_store, "list_chunks", no_live_work)
    monkeypatch.setattr(container.graph_store, "chunks_for_entities", no_live_work)


def test_completed_answer_review_survives_restart_without_changing_export(
    review_api: ReviewAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = review_api
    export_path = f"/runs/{api.run_id}/export"
    before = {
        fmt: api.client.get(export_path, params={"format": fmt}).content
        for fmt in ("json", "markdown")
    }
    events_before = raw_events(api)
    recorded_run = api.client.get("/runs").json()
    forbid_runtime_work(api.container, monkeypatch)
    body = review_body(api)

    created = api.client.post(api.path, json=body)
    assert created.status_code == 201, created.text
    saved = created.json()
    assert set(saved) == {
        "schema_version",
        "sequence",
        "run_id",
        "review_id",
        "decision",
        "comment",
        "cited_chunk_ids",
        "created_at",
    }
    assert saved["schema_version"] == "1.0"
    assert saved["run_id"] == api.run_id
    assert saved["sequence"] > 0
    assert saved["created_at"].endswith("Z")
    for field, value in body.items():
        assert saved[field] == value

    restarted = create_app(offline_settings(api.database_path))
    forbid_runtime_work(restarted.state.container, monkeypatch)
    with TestClient(restarted) as client:
        history = client.get(api.path)
        assert history.status_code == 200, history.text
        assert history.json() == {"reviews": [saved], "next_cursor": None}
        assert history.headers["cache-control"] == "no-store"
        assert history.headers["x-content-type-options"] == "nosniff"
        for fmt, content in before.items():
            assert client.get(export_path, params={"format": fmt}).content == content
        assert client.get("/runs").json() == recorded_run
    assert raw_events(api) == events_before


def test_empty_review_history_for_an_exportable_run(review_api: ReviewAPI) -> None:
    response = review_api.client.get(review_api.path)
    assert response.status_code == 200, response.text
    assert response.json() == {"reviews": [], "next_cursor": None}


def test_retry_idempotency_conflict_and_append_are_visible_over_http(
    review_api: ReviewAPI,
) -> None:
    api = review_api
    body = review_body(api)
    first = api.client.post(api.path, json=body)
    assert first.status_code == 201, first.text
    retry = api.client.post(api.path, json=body)
    assert retry.status_code == 200
    assert retry.json() == first.json()

    conflict = api.client.post(api.path, json={**body, "comment": "A different opinion."})
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["code"] == "review_id_conflict"
    second = api.client.post(
        api.path,
        json=review_body(api, decision="accepted", comment="Accepted for this synthetic exercise."),
    )
    assert second.status_code == 201
    assert second.json()["sequence"] > first.json()["sequence"]
    assert api.client.get(api.path).json() == {
        "reviews": [second.json(), first.json()],
        "next_cursor": None,
    }


@pytest.mark.parametrize("decision", ["accepted", "needs_revision", "rejected"])
def test_decisions_are_explicit_opinions_with_exact_unicode_comments(
    review_api: ReviewAPI, decision: str
) -> None:
    comment = (
        "  Opinion, not proof: \u7814\u7a76 <script>inert text</script>\n" + "\U0001f52c" * 3944
    )
    assert len(comment) <= 4000
    body = review_body(review_api, decision=decision, comment=comment)
    body.pop("cited_chunk_ids")
    created = review_api.client.post(review_api.path, json=body)
    assert created.status_code == 201
    assert created.json()["decision"] == decision
    assert created.json()["comment"] == comment
    assert created.json()["cited_chunk_ids"] == []
    assert created.headers["content-type"] == "application/json"
    retry = review_api.client.post(review_api.path, json={**body, "cited_chunk_ids": []})
    assert retry.status_code == 200
    assert retry.json() == created.json()


@pytest.mark.parametrize(
    "change",
    [
        {"review_id": "not-a-uuid"},
        {"review_id": 123},
        {"review_id": None},
        {"decision": "verified"},
        {"decision": "ACCEPTED"},
        {"decision": ""},
        {"decision": True},
        {"decision": None},
        {"comment": ""},
        {"comment": " \t\n\u3000"},
        {"comment": "text\x00tail"},
        {"comment": "x" * 4001},
        {"comment": 42},
        {"comment": None},
        {"cited_chunk_ids": None},
        {"cited_chunk_ids": "not-an-array"},
        {"cited_chunk_ids": {}},
        {"cited_chunk_ids": [42]},
        {"cited_chunk_ids": [""]},
        {"cited_chunk_ids": [" "]},
        {"cited_chunk_ids": [" padding"]},
        {"cited_chunk_ids": ["padding "]},
        {"cited_chunk_ids": ["control\ncharacter"]},
        {"cited_chunk_ids": ["x" * 257]},
        {"cited_chunk_ids": ["duplicate", "duplicate"]},
        {"cited_chunk_ids": [f"chunk-{index}" for index in range(101)]},
        {"reviewer_label": "Not an authenticated identity"},
        {"created_at": "2026-01-01T00:00:00Z"},
        {"run_id": "another-run"},
        {"sequence": 1},
    ],
)
def test_invalid_review_fields_are_bounded_sanitized_and_do_not_append(
    review_api: ReviewAPI, change: dict[str, object]
) -> None:
    response = review_api.client.post(review_api.path, json=review_body(review_api, **change))
    assert response.status_code == 422, response.text
    assert response.json()["detail"]["code"] == "invalid_review_request"
    assert response.headers["cache-control"] == "no-store"
    assert review_api.client.get(review_api.path).json() == {"reviews": [], "next_cursor": None}


@pytest.mark.parametrize("field", ["review_id", "decision", "comment"])
def test_required_review_fields_cannot_be_omitted(review_api: ReviewAPI, field: str) -> None:
    body = review_body(review_api)
    del body[field]
    assert review_api.client.post(review_api.path, json=body).status_code == 422


def test_malformed_json_and_invalid_unicode_are_explicit_client_errors(
    review_api: ReviewAPI,
) -> None:
    for body in (
        b'{"comment":',
        json.dumps(review_body(review_api, comment="\ud800")).encode("ascii"),
    ):
        response = review_api.client.post(
            review_api.path, content=body, headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
        assert response.json()["detail"]["code"] == "invalid_review_request"


def test_exact_comment_bound_and_uuid_normalization(review_api: ReviewAPI) -> None:
    body = review_body(review_api, comment="\U0001f52c" * 4000)
    created = review_api.client.post(review_api.path, json=body)
    assert created.status_code == 201
    assert created.json()["comment"] == body["comment"]
    retry = review_api.client.post(
        review_api.path, json={**body, "review_id": body["review_id"].upper()}
    )
    assert retry.status_code == 200
    assert retry.json() == created.json()


@pytest.mark.parametrize("field", ["decision", "comment", "cited_chunk_ids"])
def test_reused_uuid_with_changed_content_is_always_a_conflict(
    review_api: ReviewAPI, field: str
) -> None:
    body = review_body(review_api)
    created = review_api.client.post(review_api.path, json=body)
    changed = {
        "decision": "rejected",
        "comment": body["comment"] + " ",
        "cited_chunk_ids": ["not-even-in-the-bundle"],
    }
    conflict = review_api.client.post(review_api.path, json={**body, field: changed[field]})
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["code"] == "review_id_conflict"
    assert review_api.client.get(review_api.path).json()["reviews"] == [created.json()]


def test_review_ids_and_history_are_scoped_to_the_run(review_api: ReviewAPI) -> None:
    api = review_api
    body = review_body(api)
    first = api.client.post(api.path, json=body)
    run = api.client.post("/query", json={"query": "What connects research entities?"})
    other_id = run.json()["result"]["run_id"]
    other_path = f"/runs/{other_id}/reviews"
    second = api.client.post(other_path, json={**body, "comment": "A separate run."})
    assert first.status_code == second.status_code == 201
    assert second.json()["run_id"] == other_id
    assert api.client.get(api.path).json()["reviews"] == [first.json()]
    assert api.client.get(other_path).json()["reviews"] == [second.json()]


@pytest.mark.parametrize(
    "params",
    [
        {"limit": "0"},
        {"limit": "101"},
        {"limit": "-1"},
        {"limit": "1.5"},
        {"limit": "true"},
        {"cursor": "0"},
        {"cursor": "-1"},
        {"cursor": str(2**63)},
        {"cursor": "1.5"},
        {"cursor": "garbage"},
        {"cursor": ""},
    ],
)
def test_history_query_bounds(review_api: ReviewAPI, params: dict[str, str]) -> None:
    response = review_api.client.get(review_api.path, params=params)
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_review_request"


@pytest.mark.parametrize("run_id", [" ", "x" * 257, "padded "])
def test_review_run_id_bounds(review_api: ReviewAPI, run_id: str) -> None:
    path = f"/runs/{quote(run_id, safe='')}/reviews"
    assert review_api.client.get(path).status_code == 422
    assert review_api.client.post(path, json=review_body(review_api)).status_code == 422


def test_history_uses_exclusive_keysets_lookahead_and_not_timestamps(review_api: ReviewAPI) -> None:
    api = review_api
    saved = []
    for index in range(5):
        result = api.client.post(api.path, json=review_body(api, comment=f"Judgment {index}."))
        assert result.status_code == 201
        saved.append(result.json())
    with sqlite3.connect(api.database_path) as connection:
        connection.execute("UPDATE answer_reviews SET created_at = '2026-09-22T12:00:00Z'")
    first = api.client.get(api.path, params={"limit": 2}).json()
    assert [r["sequence"] for r in first["reviews"]] == [r["sequence"] for r in saved[-1:-3:-1]]
    assert first["next_cursor"] == saved[3]["sequence"]
    inserted_later = api.client.post(
        api.path, json=review_body(api, comment="Added between pages.")
    )
    assert inserted_later.status_code == 201
    second = api.client.get(api.path, params={"limit": 2, "cursor": first["next_cursor"]}).json()
    assert [r["sequence"] for r in second["reviews"]] == [
        saved[2]["sequence"],
        saved[1]["sequence"],
    ]
    assert second["next_cursor"] == saved[1]["sequence"]
    third = api.client.get(api.path, params={"limit": 2, "cursor": second["next_cursor"]}).json()
    assert [r["sequence"] for r in third["reviews"]] == [saved[0]["sequence"]]
    assert third["next_cursor"] is None
    assert api.client.get(api.path, params={"cursor": saved[0]["sequence"]}).json() == {
        "reviews": [],
        "next_cursor": None,
    }
    exact = api.client.get(api.path, params={"limit": 2, "cursor": saved[2]["sequence"]}).json()
    assert len(exact["reviews"]) == 2
    assert exact["next_cursor"] is None
    assert api.client.get(api.path, params={"limit": 1}).json()["reviews"] == [
        inserted_later.json()
    ]


def test_default_and_maximum_history_limits(review_api: ReviewAPI) -> None:
    for _ in range(101):
        assert (
            review_api.client.post(review_api.path, json=review_body(review_api)).status_code == 201
        )
    default = review_api.client.get(review_api.path).json()
    maximum = review_api.client.get(review_api.path, params={"limit": 100}).json()
    assert len(default["reviews"]) == 20
    assert default["next_cursor"] == default["reviews"][-1]["sequence"]
    assert len(maximum["reviews"]) == 100
    assert maximum["next_cursor"] == maximum["reviews"][-1]["sequence"]
    last = review_api.client.get(
        review_api.path, params={"limit": 100, "cursor": maximum["next_cursor"]}
    ).json()
    assert len(last["reviews"]) == 1
    assert last["next_cursor"] is None


def test_frozen_chunk_membership_survives_current_corpus_edits_and_restart(
    review_api: ReviewAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = review_api
    export_path = f"/runs/{api.run_id}/export"
    exports = {
        fmt: api.client.get(export_path, params={"format": fmt}).content
        for fmt in ("json", "markdown")
    }
    events = raw_events(api)
    current = api.client.post(
        "/ingest/text",
        json={
            "title": "New corpus",
            "text": "An unrelated current source.",
            "source": "synthetic:new",
        },
    ).json()["chunk_ids"][0]
    unknown = api.client.post(api.path, json=review_body(api, cited_chunk_ids=[current]))
    assert unknown.status_code == 422
    assert unknown.json()["detail"]["code"] == "invalid_review_references"
    with sqlite3.connect(api.database_path) as connection:
        connection.execute("UPDATE chunks SET text = 'Replaced current corpus text'")
        connection.execute("DELETE FROM chunks WHERE chunk_id = ?", (api.chunk_id,))
        connection.execute("DELETE FROM documents")
    restarted = create_app(offline_settings(api.database_path))
    forbid_runtime_work(restarted.state.container, monkeypatch)
    with TestClient(restarted) as client:
        created = client.post(api.path, json=review_body(api))
        assert created.status_code == 201
        assert created.json()["cited_chunk_ids"] == [api.chunk_id]
        assert client.get(api.path).json()["reviews"] == [created.json()]
        for fmt, content in exports.items():
            assert client.get(export_path, params={"format": fmt}).content == content
    assert raw_events(api) == events


def test_citations_use_scoped_final_bundle_not_any_known_chunk(review_api: ReviewAPI) -> None:
    api = review_api
    added = api.client.post(
        "/ingest/text",
        json={
            "title": "Other paper",
            "text": "GraphRAG connects other entities.",
            "source": "synthetic:other",
        },
    ).json()
    scoped = api.client.post(
        "/query",
        json={"query": "What does GraphRAG connect?", "document_ids": [added["document_id"]]},
    ).json()["result"]
    path = f"/runs/{scoped['run_id']}/reviews"
    outside = api.client.post(path, json=review_body(api))
    assert outside.status_code == 422
    allowed = api.client.post(path, json=review_body(api, cited_chunk_ids=[added["chunk_ids"][0]]))
    assert allowed.status_code == 201


@pytest.mark.parametrize(
    ("kind", "status", "code"),
    [
        ("missing", 404, "run_not_found"),
        ("incomplete", 409, "run_incomplete"),
        ("failed", 409, "run_failed"),
        ("legacy", 409, "snapshot_unavailable"),
        ("invalid_json", 409, "invalid_run_record"),
        ("invalid_snapshot", 409, "invalid_run_record"),
        ("invalid_blob", 409, "invalid_run_record"),
    ],
)
def test_nonexportable_runs_fail_explicitly_on_both_routes(
    review_api: ReviewAPI,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    status: int,
    code: str,
) -> None:
    api = review_api
    run_id = kind
    if kind in {"incomplete", "legacy"}:
        api.container.event_log.append_transition(
            StateTransition(
                agent_id="synthetic-agent",
                run_id=run_id,
                from_state=AgentState.IDLE if kind == "incomplete" else AgentState.ANSWERING,
                to_state=AgentState.PLANNING if kind == "incomplete" else AgentState.DONE,
                payload={"query": "synthetic-private-query"},
            )
        )
    elif kind == "failed":

        async def fail(request: LLMRequest) -> LLMResponse:
            raise RuntimeError("synthetic-private-provider-error")

        monkeypatch.setattr(api.container.llm, "generate", fail)
        result = api.client.post("/query", json={"query": "A failing synthetic query"}).json()[
            "result"
        ]
        assert result["state"] == "ERROR"
        run_id = result["run_id"]
    elif kind.startswith("invalid_"):
        run_id = api.run_id
        replacement: str | bytes = {
            "invalid_json": "{synthetic-private-broken-json",
            "invalid_snapshot": "{}",
            "invalid_blob": b"\xff\xfeinvalid",
        }[kind]
        with sqlite3.connect(api.database_path) as connection:
            connection.execute(
                "UPDATE agent_events SET payload = ? "
                "WHERE run_id = ? AND event_type = 'evidence_snapshot'",
                (replacement, run_id),
            )
    path = f"/runs/{run_id}/reviews"
    for response in (api.client.get(path), api.client.post(path, json=review_body(api))):
        assert response.status_code == status, response.text
        assert response.json()["detail"]["code"] == code
        assert "synthetic-private" not in response.text
        assert str(api.database_path) not in response.text
        assert response.headers["cache-control"] == "no-store"
    with sqlite3.connect(api.database_path) as connection:
        assert connection.execute("SELECT count(*) FROM answer_reviews").fetchone()[0] == 0


def test_history_does_not_hide_later_evidence_corruption(review_api: ReviewAPI) -> None:
    body = review_body(review_api)
    assert review_api.client.post(review_api.path, json=body).status_code == 201
    with sqlite3.connect(review_api.database_path) as connection:
        connection.execute(
            "UPDATE agent_events SET payload = '{}' WHERE event_type = 'evidence_snapshot'"
        )
    assert review_api.client.get(review_api.path).json()["detail"]["code"] == "invalid_run_record"
    assert (
        review_api.client.post(review_api.path, json=body).json()["detail"]["code"]
        == "invalid_run_record"
    )


def test_ungrounded_answer_and_empty_frozen_context_can_be_reviewed(review_api: ReviewAPI) -> None:
    result = review_api.client.post(
        "/query",
        json={"query": "No matching paper", "document_ids": ["unknown-document"]},
    ).json()["result"]
    assert result["state"] == "DONE"
    assert result["answer"]["ungrounded"] is True
    path = f"/runs/{result['run_id']}/reviews"
    accepted = review_api.client.post(
        path, json=review_body(review_api, decision="rejected", cited_chunk_ids=[])
    )
    assert accepted.status_code == 201
    assert review_api.client.post(path, json=review_body(review_api)).status_code == 422


def test_no_update_or_delete_review_surface_is_exposed(review_api: ReviewAPI) -> None:
    body = review_body(review_api)
    created = review_api.client.post(review_api.path, json=body)
    for method in ("PUT", "PATCH", "DELETE"):
        response = review_api.client.request(method, review_api.path, json=body)
        assert response.status_code == 405
        individual = review_api.client.request(
            method, f"{review_api.path}/{body['review_id']}", json=body
        )
        assert individual.status_code == 404
    assert review_api.client.get(review_api.path).json()["reviews"] == [created.json()]
