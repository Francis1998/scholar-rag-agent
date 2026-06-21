"""Tests for SQLite event persistence."""

from pathlib import Path

from agent.models import AgentState, StateTransition
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
