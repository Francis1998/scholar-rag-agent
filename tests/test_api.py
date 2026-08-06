"""Tests for FastAPI endpoints."""

from pathlib import Path

from fastapi.testclient import TestClient

from api.dependencies import AppContainer
from api.main import app
from config import Settings


def test_api_ingest_and_query(tmp_path: Path) -> None:
    """API ingests text and returns an agent query response."""
    app.state.container = AppContainer(Settings(database_path=tmp_path / "api.sqlite3"))
    client = TestClient(app)
    ingest_response = client.post(
        "/ingest/text",
        json={
            "title": "GraphRAG API Fixture",
            "text": "GraphRAG connects entities for multi-hop scientific retrieval.",
            "source": "fixture",
        },
    )
    assert ingest_response.status_code == 200
    query_response = client.post("/query", json={"query": "What does GraphRAG connect?"})
    assert query_response.status_code == 200
    assert query_response.json()["result"]["state"] == "DONE"


def test_api_query_honors_max_sources(tmp_path: Path) -> None:
    """POST /query must accept max_sources and clamp retrieval accordingly."""
    app.state.container = AppContainer(Settings(database_path=tmp_path / "api-max-sources.sqlite3"))
    client = TestClient(app)
    for index in range(3):
        ingest_response = client.post(
            "/ingest/text",
            json={
                "title": f"Source {index}",
                "text": (
                    f"Hybrid retrieval document {index} explains GraphRAG entity linking "
                    "for scientific literature answers."
                ),
                "source": "fixture",
            },
        )
        assert ingest_response.status_code == 200

    query_response = client.post(
        "/query",
        json={"query": "What is GraphRAG entity linking?", "max_sources": 1},
    )
    assert query_response.status_code == 200
    result = query_response.json()["result"]
    assert result["state"] == "DONE"
    events = client.get(f"/runs/{result['run_id']}/events")
    assert events.status_code == 200
    retrieving = [
        event
        for event in events.json()
        if event.get("event_type") == "state_transition"
        and event.get("payload", {}).get("to_state") == "REASONING"
    ]
    assert retrieving
    chunk_ids = retrieving[0]["payload"].get("payload", {}).get("chunk_ids", [])
    assert len(chunk_ids) <= 1
