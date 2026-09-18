"""Application instances and explicitly offline settings remain independently scoped."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from scripts.demo_evidence_export import offline_settings

from api.application import create_app


def test_explicit_factory_apps_have_isolated_containers_and_all_routes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ambient = tmp_path / "must-not-create.sqlite3"
    monkeypatch.setenv("SCHOLAR_RAG_DATABASE_PATH", str(ambient))
    monkeypatch.setenv("SCHOLAR_RAG_MAX_HOPS", "invalid")
    first = create_app(offline_settings(tmp_path / "first.sqlite3"))
    second = create_app(offline_settings(tmp_path / "second.sqlite3"))
    assert first.state.container is not second.state.container
    assert not ambient.exists()
    assert {
        "/health",
        "/ingest/text",
        "/query",
        "/runs",
        "/runs/{run_id}/events",
        "/runs/{run_id}/export",
    } <= set(first.openapi()["paths"])
    with TestClient(first) as one, TestClient(second) as two:
        assert one.get("/health").json() == {"status": "ok"}
        ingested = one.post(
            "/ingest/text",
            json={"title": "Synthetic note", "text": "GraphRAG connects synthetic entities."},
        )
        assert ingested.status_code == 200
        result = one.post("/query", json={"query": "What does GraphRAG connect?"})
        assert result.status_code == 200
        assert result.json()["result"]["state"] == "DONE"
        assert two.get("/runs").json() == {"runs": [], "next_cursor": None}
        recorded = one.get("/runs").json()["runs"][0]
        assert one.get(recorded["events_url"]).status_code == 200
        assert one.get(recorded["export_url"]).json()["generation"]["provider"] == "fake"


def test_default_factory_still_validates_production_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "not-created.sqlite3"
    monkeypatch.setenv("SCHOLAR_RAG_DATABASE_PATH", str(path))
    monkeypatch.setenv("SCHOLAR_RAG_MAX_HOPS", "invalid")
    with pytest.raises(ValidationError, match="max_hops"):
        create_app()
    assert not path.exists()
    with pytest.raises(ValidationError, match="max_hops"):
        type(offline_settings(path))(database_path=path, max_hops=99)
