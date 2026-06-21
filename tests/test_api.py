"""Tests for FastAPI endpoints."""

from pathlib import Path

from api.dependencies import AppContainer
from api.main import app
from config import Settings
from fastapi.testclient import TestClient


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
