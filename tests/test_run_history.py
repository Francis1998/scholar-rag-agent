"""Offline acceptance tests for restart-safe, bounded run discovery."""

import asyncio
import json
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Any, NoReturn
from urllib.parse import quote

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from agent.models import AgentState, StateTransition
from agent.safety import CancellationToken
from api.dependencies import AppContainer
from api.main import app
from config import Settings
from llm.schemas import LLMRequest, LLMResponse
from storage.event_log import SQLiteEventLog
from storage.run_history import RunHistoryPage, SQLiteRunHistory

TIED_TIME = "2026-09-18T16:00:00Z"
QUERY_LIMIT = 300


def _settings(path: Path) -> Settings:
    return Settings(
        _env_file=None,
        database_path=path,
        default_model="fake",
        OPENAI_API_KEY="",
        ANTHROPIC_API_KEY="",
        GEMINI_API_KEY="",
        MOONSHOT_API_KEY="",
    )


def _no_work(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("Run discovery must not read evidence, retrieve, generate, or use HTTP")


@dataclass
class HistoryAPI:
    application: FastAPI
    client: TestClient
    container: AppContainer
    database_path: Path


@pytest.fixture
def history_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[HistoryAPI]:
    path = tmp_path / "history.sqlite3"
    application = FastAPI()
    application.include_router(app.router)
    container = AppContainer(_settings(path))
    application.state.container = container
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", _no_work)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _no_work)
    with TestClient(application) as client:
        yield HistoryAPI(application, client, container, path)


def _transition(
    api: HistoryAPI,
    run_id: str,
    state: AgentState = AgentState.PLANNING,
    *,
    query: str | None = "Synthetic question",
    from_state: AgentState = AgentState.IDLE,
) -> int:
    return api.container.event_log.append_transition(
        StateTransition(
            agent_id="synthetic-agent",
            run_id=run_id,
            from_state=from_state,
            to_state=state,
            payload={"query": query} if state == AgentState.PLANNING else {},
        )
    )


def _page(api: HistoryAPI, **params: str | int) -> dict[str, Any]:
    response = api.client.get("/runs", params=params)
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    result: dict[str, Any] = response.json()
    assert set(result) == {"runs", "next_cursor"}
    return result


def _query(api: HistoryAPI, query: str) -> dict[str, Any]:
    response = api.client.post("/query", json={"query": query})
    assert response.status_code == 200, response.text
    result: dict[str, Any] = response.json()["result"]
    return result


def test_empty_catalog(history_api: HistoryAPI) -> None:
    assert _page(history_api) == {"runs": [], "next_cursor": None}


def test_real_api_run_restart_discovery_and_existing_exports(
    history_api: HistoryAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recover an actual completed run without keeping its ID across restart."""
    api = history_api
    ingest = api.client.post(
        "/ingest/text",
        json={
            "title": "Synthetic note",
            "text": "Synthetic data, not a paper. GraphRAG connects research entities.",
            "source": "synthetic:run-history",
        },
    )
    assert ingest.status_code == 200
    question = "  What does GraphRAG connect? \u7814\u7a76  "
    completed = _query(api, question)
    assert completed["state"] == "DONE"
    before = api.client.get(f"/runs/{completed['run_id']}/export")
    assert before.status_code == 200
    with sqlite3.connect(api.database_path) as connection:
        first_id, count = connection.execute(
            "SELECT min(id), count(*) FROM agent_events WHERE run_id = ?",
            (completed["run_id"],),
        ).fetchone()

    restarted = AppContainer(_settings(api.database_path))
    api.application.state.container = restarted
    monkeypatch.setattr(restarted.llm, "generate", _no_work)
    monkeypatch.setattr(restarted.runner, "run", _no_work)
    monkeypatch.setattr(restarted.hybrid_retriever, "retrieve", _no_work)
    monkeypatch.setattr(restarted.document_store, "list_chunks", _no_work)
    monkeypatch.setattr(restarted.graph_store, "chunks_for_entities", _no_work)
    original_events = restarted.event_log.list_events
    monkeypatch.setattr(restarted.event_log, "list_events", _no_work)
    page = _page(api)
    assert page["next_cursor"] is None
    assert len(page["runs"]) == 1
    summary = page["runs"][0]
    assert set(summary) == {
        "run_id",
        "agent_id",
        "first_event_id",
        "recorded_state",
        "query_summary",
        "query_truncated",
        "started_at",
        "updated_at",
        "event_count",
        "events_url",
        "export_url",
    }
    assert summary["run_id"] == completed["run_id"]
    assert summary["agent_id"] == "local-agent"
    assert summary["recorded_state"] == "DONE"
    assert summary["query_summary"] == question
    assert summary["query_truncated"] is False
    assert summary["event_count"] == count == 8
    assert summary["first_event_id"] == first_id
    assert summary["started_at"].endswith("Z")
    assert summary["updated_at"].endswith("Z")

    monkeypatch.setattr(restarted.event_log, "list_events", original_events)
    events = api.client.get(summary["events_url"])
    exported = api.client.get(summary["export_url"])
    markdown = api.client.get(summary["export_url"], params={"format": "markdown"})
    assert events.status_code == 200
    assert len(events.json()) == count
    assert exported.status_code == markdown.status_code == 200
    assert exported.content == before.content
    assert markdown.text.startswith("# Research evidence bundle\n")


def test_real_failed_run_omits_error_details(
    history_api: HistoryAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fail(request: LLMRequest) -> LLMResponse:
        del request
        raise RuntimeError("synthetic-private-provider-error")

    monkeypatch.setattr(history_api.container.llm, "generate", fail)
    failed = _query(history_api, "Why did this synthetic run fail?")
    assert failed["state"] == "ERROR"
    summary = _page(history_api, state="ERROR")["runs"][0]
    assert summary["run_id"] == failed["run_id"]
    assert summary["recorded_state"] == "ERROR"
    assert summary["query_summary"] == "Why did this synthetic run fail?"
    assert "synthetic-private-provider-error" not in json.dumps(summary)
    assert history_api.client.get(summary["export_url"]).status_code == 409


async def test_real_concurrent_listing_does_not_claim_liveness(
    history_api: HistoryAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """List while a real query is paused, then after it persists DONE."""
    entered = asyncio.Event()
    release = asyncio.Event()
    original_generate = history_api.container.llm.generate

    async def paused(request: LLMRequest) -> LLMResponse:
        entered.set()
        await release.wait()
        return await original_generate(request)

    monkeypatch.setattr(history_api.container.llm, "generate", paused)
    transport = httpx.ASGITransport(app=history_api.application)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        task = asyncio.create_task(client.post("/query", json={"query": "Synthetic overlap"}))
        try:
            await asyncio.wait_for(entered.wait(), timeout=5)
            response = await client.get("/runs", params={"state": "REASONING"})
            assert response.status_code == 200
            summary = response.json()["runs"][0]
            assert summary["recorded_state"] == "REASONING"
            assert summary["query_summary"] == "Synthetic overlap"
            assert "running" not in response.text.lower()
            assert "active" not in response.text.lower()
            release.set()
            finished = await asyncio.wait_for(task, timeout=5)
            assert finished.json()["result"]["state"] == "DONE"
            done = await client.get("/runs", params={"state": "DONE"})
            assert done.status_code == 200
            assert done.json()["runs"][0]["run_id"] == summary["run_id"]
            assert done.json()["runs"][0]["event_count"] > summary["event_count"]
            old_state = await client.get("/runs", params={"state": "REASONING"})
            assert old_state.json() == {"runs": [], "next_cursor": None}
        finally:
            release.set()
            await asyncio.wait_for(task, timeout=5)


def test_keyset_pages_ignore_later_events_and_new_runs(history_api: HistoryAPI) -> None:
    """Tied clocks, interleaving, and late completions never move creation keys."""
    api = history_api
    first_ids = {}
    for number in range(7):
        run_id = f"run-{number}"
        first_ids[run_id] = _transition(api, run_id, query=f"Original {number}")
        if number > 0:
            api.container.event_log.append_event(
                "synthetic-agent", f"run-{number - 1}", "audit", {"ignored": "private"}
            )
    with sqlite3.connect(api.database_path) as connection:
        connection.execute("UPDATE agent_events SET timestamp = ?", (TIED_TIME,))

    first = _page(api, limit=2)
    assert [run["run_id"] for run in first["runs"]] == ["run-6", "run-5"]
    assert first["next_cursor"] == first_ids["run-5"]
    _transition(api, "run-new", query="Created after the first page")
    _transition(api, "run-0", AgentState.ERROR, from_state=AgentState.PLANNING)
    _transition(api, "run-6", AgentState.ERROR, from_state=AgentState.PLANNING)
    # Even another PLANNING record must not replace the original query or creation key.
    _transition(api, "run-2", query="Later query must not become the summary")
    with sqlite3.connect(api.database_path) as connection:
        connection.execute("UPDATE agent_events SET timestamp = ?", (TIED_TIME,))

    seen = first["runs"]
    cursor = first["next_cursor"]
    while cursor is not None:
        page = _page(api, limit=2, cursor=cursor)
        seen.extend(page["runs"])
        cursor = page["next_cursor"]
    assert [run["run_id"] for run in seen] == [f"run-{number}" for number in reversed(range(7))]
    assert len({run["run_id"] for run in seen}) == 7
    assert all(run["first_event_id"] == first_ids[run["run_id"]] for run in seen)
    assert all(run["started_at"] == run["updated_at"] == TIED_TIME for run in seen)
    assert next(run for run in seen if run["run_id"] == "run-2")["query_summary"] == "Original 2"
    assert seen[-1]["recorded_state"] == "ERROR"
    assert _page(api, limit=1)["runs"][0]["run_id"] == "run-new"


def test_state_filter_precedes_pagination_and_uses_latest_transition(
    history_api: HistoryAPI,
) -> None:
    api = history_api
    keys = [_transition(api, f"r-{number}") for number in range(6)]
    for number in (0, 2, 4):
        _transition(api, f"r-{number}", AgentState.ERROR, from_state=AgentState.PLANNING)
        api.container.event_log.append_event("synthetic-agent", f"r-{number}", "audit", {})
    first = _page(api, state="ERROR", limit=2)
    assert [run["run_id"] for run in first["runs"]] == ["r-4", "r-2"]
    assert first["next_cursor"] == keys[2]
    last = _page(api, state="ERROR", limit=2, cursor=first["next_cursor"])
    assert [run["run_id"] for run in last["runs"]] == ["r-0"]
    assert last["next_cursor"] is None
    assert _page(api, state="DONE") == {"runs": [], "next_cursor": None}


def test_filtered_pages_are_not_a_frozen_state_snapshot(history_api: HistoryAPI) -> None:
    api = history_api
    for number in range(5):
        _transition(api, f"r-{number}")
    first = _page(api, state="PLANNING", limit=2)
    assert [run["run_id"] for run in first["runs"]] == ["r-4", "r-3"]
    _transition(api, "r-2", AgentState.ERROR, from_state=AgentState.PLANNING)
    _transition(api, "r-4", AgentState.ERROR, from_state=AgentState.PLANNING)
    second = _page(api, state="PLANNING", limit=2, cursor=first["next_cursor"])
    assert [run["run_id"] for run in second["runs"]] == ["r-1", "r-0"]
    assert second["next_cursor"] is None
    assert [run["run_id"] for run in _page(api, state="ERROR")["runs"]] == ["r-4", "r-2"]


@pytest.mark.parametrize("count", [0, 1, 2, 3, 4])
def test_next_cursor_only_when_another_run_exists(history_api: HistoryAPI, count: int) -> None:
    keys = [_transition(history_api, f"r-{number}") for number in range(count)]
    page = _page(history_api, limit=2)
    assert len(page["runs"]) == min(count, 2)
    assert page["next_cursor"] == (keys[-2] if count > 2 else None)
    assert _page(history_api, cursor=1) == {"runs": [], "next_cursor": None}


def test_default_and_maximum_page_size(history_api: HistoryAPI) -> None:
    for number in range(103):
        _transition(history_api, f"r-{number}")
    assert len(_page(history_api)["runs"]) == 20
    maximum = _page(history_api, limit=100)
    assert len(maximum["runs"]) == 100
    assert maximum["next_cursor"] is not None
    last = _page(history_api, limit=100, cursor=maximum["next_cursor"])
    assert len(last["runs"]) == 3
    assert last["next_cursor"] is None


@pytest.mark.parametrize(
    "params",
    [
        {"limit": "0"},
        {"limit": "-1"},
        {"limit": "101"},
        {"limit": "2.5"},
        {"limit": "many"},
        {"cursor": "0"},
        {"cursor": "-1"},
        {"cursor": "2.5"},
        {"cursor": "not-a-cursor"},
        {"cursor": "9223372036854775808"},
        {"state": "RUNNING"},
        {"state": "done"},
        {"state": ""},
    ],
)
def test_invalid_parameters_are_422(history_api: HistoryAPI, params: dict[str, str]) -> None:
    response = history_api.client.get("/runs", params=params)
    assert response.status_code == 422, response.text


def test_event_only_legacy_and_precancelled_records_are_discoverable(
    history_api: HistoryAPI,
) -> None:
    api = history_api
    api.container.event_log.append_event(
        "legacy-agent", "event-only", "diagnostic", {"query": "not a planning query"}
    )
    _transition(api, "legacy-done", AgentState.DONE)
    _transition(api, "query-missing", query=None)
    token = CancellationToken()
    token.cancel()
    cancelled = asyncio.run(api.container.runner.run("Never observed", token=token))
    assert cancelled.state == AgentState.ERROR
    page = _page(api)
    by_id = {run["run_id"]: run for run in page["runs"]}
    assert by_id["event-only"]["recorded_state"] is None
    assert by_id["legacy-done"]["recorded_state"] == "DONE"
    for run in by_id.values():
        assert run["query_summary"] is None
        assert run["query_truncated"] is False
    assert by_id[cancelled.run_id]["recorded_state"] == "ERROR"
    assert api.client.get(by_id["legacy-done"]["export_url"]).status_code == 409
    assert [run["run_id"] for run in _page(api, state="DONE")["runs"]] == ["legacy-done"]


@pytest.mark.parametrize(
    "query",
    [
        "",
        "x" * QUERY_LIMIT,
        "x" * (QUERY_LIMIT + 1),
        "\u7814\u7a76\U0001f52c" * QUERY_LIMIT,
        "before\x00after" + "x" * QUERY_LIMIT,
    ],
    ids=["empty", "at-limit", "over-limit", "unicode", "embedded-nul"],
)
def test_query_summary_has_exact_bounded_prefix_and_truncation(
    history_api: HistoryAPI, query: str
) -> None:
    _transition(history_api, "bounded", query=query)
    summary = _page(history_api)["runs"][0]
    assert summary["query_summary"] == query[:QUERY_LIMIT]
    assert summary["query_truncated"] is (len(query) > QUERY_LIMIT)


def test_catalog_never_deserializes_source_snapshots_or_leaks_other_payloads(
    history_api: HistoryAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = history_api
    sentinel = "synthetic-private-metadata"
    _transition(api, "bounded", query="Safe synthetic query")
    api.container.event_log.append_event(
        "synthetic-agent",
        "bounded",
        "evidence_snapshot",
        {"source_text": sentinel * 100_000, "api_key": sentinel},
    )
    api.container.event_log.append_event(
        "synthetic-agent",
        "bounded",
        "generation_record",
        {"answer": sentinel, "headers": {"Authorization": sentinel}},
    )
    api.container.event_log.append_event(
        "synthetic-agent",
        "bounded",
        "state_transition",
        {
            "from_state": "PLANNING",
            "to_state": "ERROR",
            "payload": {"error": sentinel, "metadata": sentinel, "source_text": sentinel},
        },
    )
    with sqlite3.connect(api.database_path) as connection:
        connection.execute(
            "UPDATE agent_events SET payload = ? WHERE event_type = 'generation_record'",
            ("not JSON; the catalog must never parse model payloads",),
        )
    monkeypatch.setenv("OPENAI_API_KEY", sentinel)
    monkeypatch.setattr(api.container.event_log, "list_events", _no_work)
    monkeypatch.setattr(api.container.evidence_exporter, "export", _no_work)
    monkeypatch.setattr("agent.evidence.EvidenceSnapshot.model_validate", _no_work)
    monkeypatch.setattr(api.container.llm, "generate", _no_work)
    monkeypatch.setattr(api.container.hybrid_retriever, "retrieve", _no_work)
    response = api.client.get("/runs")
    assert response.status_code == 200
    assert len(response.content) < 1500
    assert sentinel not in response.text
    summary = response.json()["runs"][0]
    assert summary["event_count"] == 4
    assert summary["recorded_state"] == "ERROR"


@pytest.mark.parametrize(
    "payload",
    [
        "not JSON",
        "null",
        "[]",
        "{}",
        '{"to_state":"DONE","payload":{}}',
        '{"from_state":"IDLE","to_state":"FUTURE","payload":{}}',
        '{"from_state":"IDLE","to_state":"DONE","payload":null}',
        '{"from_state":"IDLE","to_state":"PLANNING","payload":{"query":123}}',
        '{"from_state":"IDLE","to_state":"PLANNING","payload":{"query":{"private":"secret"}}}',
    ],
)
@pytest.mark.parametrize("state_filter", [None, "DONE"])
def test_malformed_projection_is_explicit_not_skipped_by_filter(
    history_api: HistoryAPI, payload: str, state_filter: str | None
) -> None:
    _transition(history_api, "damaged")
    with sqlite3.connect(history_api.database_path) as connection:
        connection.execute("UPDATE agent_events SET payload = ?", (payload,))
    params = {} if state_filter is None else {"state": state_filter}
    response = history_api.client.get("/runs", params=params)
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "invalid_run_record"
    assert response.headers["cache-control"] == "no-store"
    assert "secret" not in response.text
    assert payload not in response.text


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("run_id", ""),
        ("run_id", "r" * 257),
        ("agent_id", ""),
        ("agent_id", "a" * 257),
        ("timestamp", "not-a-timestamp"),
        ("timestamp", "t" * 100_000),
    ],
    ids=["empty-run", "long-run", "empty-agent", "long-agent", "bad-time", "long-time"],
)
def test_unbounded_or_malformed_identity_and_times_are_not_silent_fallbacks(
    history_api: HistoryAPI, column: str, value: str
) -> None:
    _transition(history_api, "damaged")
    statements = {
        "run_id": "UPDATE agent_events SET run_id = ?",
        "agent_id": "UPDATE agent_events SET agent_id = ?",
        "timestamp": "UPDATE agent_events SET timestamp = ?",
    }
    with sqlite3.connect(history_api.database_path) as connection:
        connection.execute(statements[column], (value,))
    response = history_api.client.get("/runs")
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "invalid_run_record"
    assert len(response.content) < 500


def test_conflicting_agent_identity_is_an_error(history_api: HistoryAPI) -> None:
    _transition(history_api, "shared-id")
    history_api.container.event_log.append_event("different-agent", "shared-id", "audit", {})
    response = history_api.client.get("/runs")
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "invalid_run_record"


def test_navigation_links_encode_ids_without_claiming_export_eligibility(
    history_api: HistoryAPI,
) -> None:
    hostile_id = 'run"?state=DONE#\u7814\u7a76'
    _transition(history_api, hostile_id)
    summary = _page(history_api)["runs"][0]
    assert summary["run_id"] == hostile_id
    assert summary["events_url"] == f"/runs/{quote(hostile_id, safe='')}/events"
    assert summary["export_url"] == f"/runs/{quote(hostile_id, safe='')}/export"
    assert history_api.client.get(summary["events_url"]).status_code == 200
    assert history_api.client.get(summary["export_url"]).status_code == 409


@pytest.mark.parametrize("run_id", ["nested/id", ".", ".."])
def test_legacy_ids_that_existing_routes_cannot_address_have_no_links(
    history_api: HistoryAPI, run_id: str
) -> None:
    _transition(history_api, run_id)
    summary = _page(history_api)["runs"][0]
    assert summary["run_id"] == run_id
    assert summary["events_url"] is summary["export_url"] is None


def test_one_listing_remains_consistent_during_an_independent_sqlite_write(
    history_api: HistoryAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An overlapping writer must not mix old states with new counts in one page."""
    api = history_api
    _transition(api, "old")
    before = _page(api)
    with sqlite3.connect(api.database_path) as connection:
        assert connection.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
    read_started = Event()
    write_finished = Event()
    connect = sqlite3.connect

    class PausingConnection(sqlite3.Connection):
        def execute(
            self, sql: str, parameters: Mapping[str, object] | Sequence[object] = (), /
        ) -> sqlite3.Cursor:
            cursor = super().execute(sql, parameters)
            if sql.lstrip().startswith("WITH"):
                read_started.set()
                assert write_finished.wait(5), "Concurrent writer did not finish"
            return cursor

    def open_connection(database: str | Path, *, uri: bool = False) -> sqlite3.Connection:
        factory = PausingConnection if uri else sqlite3.Connection
        return connect(database, uri=uri, factory=factory)

    def write() -> None:
        assert read_started.wait(5), "Catalog read did not start"
        try:
            _transition(api, "old", AgentState.ERROR, from_state=AgentState.PLANNING)
            _transition(api, "new")
        finally:
            write_finished.set()

    monkeypatch.setattr(sqlite3, "connect", open_connection)
    with ThreadPoolExecutor(max_workers=1) as pool:
        writer = pool.submit(write)
        during = _page(api)
        writer.result(timeout=5)
    assert during == before
    after = _page(api)
    assert [run["run_id"] for run in after["runs"]] == ["new", "old"]
    assert after["runs"][1]["recorded_state"] == "ERROR"
    assert after["runs"][1]["event_count"] == 2


def test_sql_projects_only_bounded_rows_and_scalars(
    history_api: HistoryAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = history_api
    for number in range(25):
        _transition(api, f"r-{number}", query="x" * 100_000)
    api.container.event_log.append_event(
        "synthetic-agent", "r-24", "evidence_snapshot", {"text": "private-source" * 100_000}
    )
    connect = sqlite3.connect
    row_counts = []
    statements = []

    class ProjectionCursor(sqlite3.Cursor):
        def fetchall(self) -> list[sqlite3.Row]:
            rows = super().fetchall()
            row_counts.append(len(rows))
            for row in rows:
                columns = set(row.keys())
                assert "payload" not in columns
                assert "query_prefix" in columns
                for value in row:
                    if isinstance(value, str | bytes):
                        assert len(value) <= 4 * (QUERY_LIMIT + 1)
            return rows

    class ProjectionConnection(sqlite3.Connection):
        def execute(
            self, sql: str, parameters: Mapping[str, object] | Sequence[object] = (), /
        ) -> sqlite3.Cursor:
            statements.append(sql)
            return self.cursor(factory=ProjectionCursor).execute(sql, parameters)

    def open_connection(database: str | Path, *, uri: bool = False) -> sqlite3.Connection:
        return connect(database, uri=uri, factory=ProjectionConnection)

    monkeypatch.setattr(sqlite3, "connect", open_connection)
    page = _page(api, limit=2)
    assert len(page["runs"]) == 2
    assert page["runs"][0]["query_summary"] == "x" * QUERY_LIMIT
    assert row_counts == [3]
    assert len(statements) == 2
    assert statements[0] == "PRAGMA encoding"
    assert statements[1].lstrip().startswith("WITH")


@pytest.mark.parametrize("encoding", ["UTF-8", "UTF-16le", "UTF-16be"])
def test_python_catalog_supports_existing_database_encodings_without_schema_changes(
    tmp_path: Path, encoding: str
) -> None:
    path = tmp_path / "existing.sqlite3"
    pragmas = {
        "UTF-8": "PRAGMA encoding='UTF-8'",
        "UTF-16le": "PRAGMA encoding='UTF-16le'",
        "UTF-16be": "PRAGMA encoding='UTF-16be'",
    }
    with sqlite3.connect(path) as connection:
        connection.execute(pragmas[encoding])
        connection.execute(
            "CREATE TABLE agent_events (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "timestamp TEXT NOT NULL, agent_id TEXT NOT NULL, run_id TEXT NOT NULL, "
            "event_type TEXT NOT NULL, payload TEXT NOT NULL)"
        )
    log = SQLiteEventLog(path)
    question = "\u7814\u7a76\x00\U0001f52c" * 500
    log.append_transition(
        StateTransition(
            agent_id="agent-\u7814\u7a76",
            run_id="run-\U0001f52c",
            from_state=AgentState.IDLE,
            to_state=AgentState.PLANNING,
            payload={"query": question},
        )
    )
    with sqlite3.connect(path) as connection:
        schema_before = connection.execute("SELECT sql FROM sqlite_master ORDER BY name").fetchall()
    page = SQLiteRunHistory(path).list_runs(state=AgentState.PLANNING)
    assert isinstance(page, RunHistoryPage)
    assert page.runs[0].query_summary == question[:QUERY_LIMIT]
    assert page.runs[0].query_truncated is True
    assert page.runs[0].agent_id == "agent-\u7814\u7a76"
    assert page.runs[0].run_id == "run-\U0001f52c"
    with sqlite3.connect(path) as connection:
        assert (
            connection.execute("SELECT sql FROM sqlite_master ORDER BY name").fetchall()
            == schema_before
        )


@pytest.mark.parametrize(
    "options",
    [
        {"limit": 0},
        {"limit": 101},
        {"limit": True},
        {"limit": 1.5},
        {"cursor": 0},
        {"cursor": 2**63},
        {"cursor": True},
        {"state": "unknown"},
    ],
)
def test_python_catalog_validates_before_database_access(
    tmp_path: Path, options: dict[str, Any]
) -> None:
    path = tmp_path / "must-not-create.sqlite3"
    history = SQLiteRunHistory(path)
    with pytest.raises(ValidationError):
        history.list_runs(**options)
    assert not path.exists()


def test_database_errors_are_not_empty_catalogs(tmp_path: Path) -> None:
    path = tmp_path / "not-events.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE unrelated (id INTEGER)")
    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        SQLiteRunHistory(path).list_runs()


def test_latest_event_timestamp_is_not_maximum_wall_clock(history_api: HistoryAPI) -> None:
    first = _transition(history_api, "clock-adjusted")
    latest = history_api.container.event_log.append_event(
        "synthetic-agent", "clock-adjusted", "audit", {}
    )
    with sqlite3.connect(history_api.database_path) as connection:
        connection.execute(
            "UPDATE agent_events SET timestamp = '2026-09-18T16:00:02Z' WHERE id = ?", (first,)
        )
        connection.execute(
            "UPDATE agent_events SET timestamp = '2026-09-18T16:00:01Z' WHERE id = ?", (latest,)
        )
    summary = _page(history_api)["runs"][0]
    assert summary["started_at"] == "2026-09-18T16:00:02Z"
    assert summary["updated_at"] == "2026-09-18T16:00:01Z"
    assert summary["recorded_state"] == "PLANNING"


def test_earlier_malformed_transition_cannot_be_hidden_by_later_state(
    history_api: HistoryAPI,
) -> None:
    first = _transition(history_api, "damaged")
    _transition(history_api, "damaged", AgentState.ERROR, from_state=AgentState.PLANNING)
    with sqlite3.connect(history_api.database_path) as connection:
        connection.execute(
            "UPDATE agent_events SET payload = 'invalid JSON' WHERE id = ?", (first,)
        )
    response = history_api.client.get("/runs", params={"state": "DONE"})
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "invalid_run_record"


@pytest.mark.parametrize("field", ["agent_id", "query"])
def test_trailing_invalid_utf8_is_not_mistaken_for_a_truncated_prefix(
    history_api: HistoryAPI, field: str
) -> None:
    _transition(history_api, "damaged-encoding")
    with sqlite3.connect(history_api.database_path) as connection:
        if field == "agent_id":
            connection.execute(
                "UPDATE agent_events SET agent_id = CAST(? AS TEXT)", (b"agent\xe2",)
            )
        else:
            connection.execute(
                "UPDATE agent_events SET payload = CAST(? AS TEXT)",
                (
                    b'{"from_state":"IDLE","to_state":"PLANNING",'
                    b'"payload":{"query":"truncated\xe2"}}',
                ),
            )
    response = history_api.client.get("/runs")
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "invalid_run_record"
