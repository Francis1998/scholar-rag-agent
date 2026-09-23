"""Offline HTTP acceptance for append-only judgments on frozen saved answers."""

import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from scripts.demo_evidence_export import offline_settings

from api.application import create_app
from api.dependencies import AppContainer


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
