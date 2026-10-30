"""Tests for SQLite event persistence."""

import json
import math
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from scripts.demo_evidence_export import offline_settings

from agent.models import AgentState, StateTransition
from api.application import create_app
from storage.event_log import SQLiteEventLog


def test_event_log_persists_transition(tmp_path: Path) -> None:
    """A state transition is persisted with timestamp and payload."""
    log = SQLiteEventLog(tmp_path / "events.sqlite3")
    log.append_transition(
        StateTransition(
            agent_id="agent-1",
            run_id="run-1",
            from_state=AgentState.IDLE,
            to_state=AgentState.PLANNING,
            payload={"query": "test"},
        )
    )
    events = log.list_events("run-1")
    assert len(events) == 1
    assert events[0]["payload"]["to_state"] == "PLANNING"


def test_event_log_persists_decision_log_payload(tmp_path: Path) -> None:
    """A decision_log event should round-trip its structured payload."""
    log = SQLiteEventLog(tmp_path / "events.sqlite3")

    event_id = log.append_event(
        agent_id="agent-1",
        run_id="run-1",
        event_type="decision_log",
        payload={
            "tasks": [{"task_id": "task-1", "query": "GraphRAG"}],
            "rationale_trace": ["comparison query"],
        },
    )
    events = log.list_events("run-1")

    assert event_id == 1
    assert len(events) == 1
    assert events[0]["event_type"] == "decision_log"
    assert events[0]["payload"] == {
        "tasks": [{"task_id": "task-1", "query": "GraphRAG"}],
        "rationale_trace": ["comparison query"],
    }


@pytest.mark.parametrize(
    "non_finite", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "-inf"]
)
@pytest.mark.parametrize("as_key", [False, True], ids=["value", "key"])
def test_event_log_rejects_non_finite_payloads(
    tmp_path: Path, non_finite: float, as_key: bool
) -> None:
    """Reject nested non-finite values and keys without consuming an event id."""
    log = SQLiteEventLog(tmp_path / "events.sqlite3")
    first_id = log.append_event("agent-1", "run-1", "metrics", {"score": 0.25})
    before = log.list_events()
    payload = {"nested": [{non_finite: "score"} if as_key else {"score": non_finite}]}

    with pytest.raises(ValueError, match="Out of range float values are not JSON compliant"):
        log.append_event("agent-1", "run-1", "metrics", payload)

    assert log.list_events() == before
    next_id = log.append_event("agent-1", "run-2", "metrics", {"score": 0.75})
    assert (first_id, next_id) == (1, 2)
    assert [event["id"] for event in log.list_events()] == [first_id, next_id]
    assert log.list_events("run-1") == before


@pytest.mark.parametrize(
    "non_finite", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "-inf"]
)
def test_event_log_rejects_non_finite_transitions(tmp_path: Path, non_finite: float) -> None:
    """Transition payloads follow the same contract without changing their envelope."""
    log = SQLiteEventLog(tmp_path / "events.sqlite3")
    transition = StateTransition(
        agent_id="agent-1",
        run_id="run-1",
        from_state=AgentState.IDLE,
        to_state=AgentState.PLANNING,
        payload={"nested": [{"score": non_finite}]},
    )

    with pytest.raises(ValueError, match="Out of range float values are not JSON compliant"):
        log.append_transition(transition)

    assert log.list_events() == []
    transition.payload = {"score": 0.5}
    assert log.append_transition(transition) == 1
    event = log.list_events("run-1")[0]
    assert event["event_type"] == "state_transition"
    assert event["payload"] == {
        "from_state": "IDLE",
        "to_state": "PLANNING",
        "payload": {"score": 0.5},
    }


@pytest.mark.parametrize(
    ("payload", "error_type"),
    [
        ({"nested": [float("nan")]}, ValueError),
        ({"nested": [object()]}, TypeError),
    ],
    ids=["non-finite", "unsupported-object"],
)
def test_event_log_validates_payload_before_opening_database(
    tmp_path: Path, payload: dict[str, object], error_type: type[Exception]
) -> None:
    """Serialization errors must not even recreate a missing database file."""
    database_path = tmp_path / "events.sqlite3"
    log = SQLiteEventLog(database_path)
    database_path.unlink()

    with pytest.raises(error_type):
        log.append_event("agent-1", "run-1", "metrics", payload)

    assert not database_path.exists()


def test_event_log_round_trips_finite_json(tmp_path: Path) -> None:
    """Keep finite numbers, JSON literals, key coercion, sorting, and run filtering."""
    database_path = tmp_path / "events.sqlite3"
    log = SQLiteEventLog(database_path)
    payload = {
        "nested": [
            {
                "values": [
                    None,
                    True,
                    False,
                    0,
                    1.25,
                    -0.0,
                    1.7976931348623157e308,
                    -1.7976931348623157e308,
                    5e-324,
                    "NaN",
                    "Infinity",
                    "-Infinity",
                ]
            }
        ],
        "numeric_keys": {2.5: "finite"},
    }
    assert log.append_event("agent-1", "other-run", "metrics", {}) == 1
    assert log.append_event("agent-1", "run-1", "metrics", payload) == 2

    reopened = SQLiteEventLog(database_path)
    events = reopened.list_events()
    assert [event["id"] for event in events] == [1, 2]
    assert reopened.list_events("run-1") == [events[1]]
    assert events[1]["payload"] == payload | {"numeric_keys": {"2.5": "finite"}}
    assert json.loads(json.dumps(events, allow_nan=False)) == events
    with closing(sqlite3.connect(database_path)) as connection:
        stored = connection.execute("SELECT payload FROM agent_events WHERE id = 2").fetchone()[0]
    assert stored == json.dumps(payload, sort_keys=True, allow_nan=False)


def test_event_log_rejected_payload_keeps_events_api_serializable(tmp_path: Path) -> None:
    """A rejected append must not poison the real run-events endpoint."""
    application = create_app(offline_settings(tmp_path / "api.sqlite3"))
    log: SQLiteEventLog = application.state.container.event_log
    log.append_event("agent-1", "run-1", "metrics", {"score": 0.25})

    with pytest.raises(ValueError, match="Out of range float values are not JSON compliant"):
        log.append_event("agent-1", "run-1", "metrics", {"nested": [{"score": float("nan")}]})

    log.append_event("agent-1", "run-1", "metrics", {"score": 0.75})
    with TestClient(application) as client:
        response = client.get("/runs/run-1/events")
    assert response.status_code == 200
    events = response.json()
    assert [event["id"] for event in events] == [1, 2]
    assert [event["payload"]["score"] for event in events] == [0.25, 0.75]
    assert json.loads(json.dumps(events, allow_nan=False)) == events


def test_event_log_leaves_legacy_non_finite_rows_unchanged(tmp_path: Path) -> None:
    """Strict new writes do not migrate or change permissive reads of existing rows."""
    database_path = tmp_path / "events.sqlite3"
    log = SQLiteEventLog(database_path)
    legacy_payload = '{"scores": [NaN, Infinity, -Infinity]}'
    with closing(sqlite3.connect(database_path)) as connection, connection:
        connection.execute(
            """
            INSERT INTO agent_events (timestamp, agent_id, run_id, event_type, payload)
            VALUES (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), 'agent-1', 'legacy', 'metrics', ?)
            """,
            (legacy_payload,),
        )

    assert log.append_event("agent-1", "run-1", "metrics", {"score": 0.5}) == 2
    scores = log.list_events("legacy")[0]["payload"]["scores"]
    assert math.isnan(scores[0])
    assert scores[1:] == [float("inf"), float("-inf")]
    with closing(sqlite3.connect(database_path)) as connection:
        stored = connection.execute("SELECT payload FROM agent_events WHERE id = 1").fetchone()[0]
    assert stored == legacy_payload
