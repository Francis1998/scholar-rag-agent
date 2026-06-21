"""SQLite durable event log for agent state transitions."""

import json
import sqlite3
from pathlib import Path
from typing import Any

from agent.models import StateTransition


class SQLiteEventLog:
    """Append-only SQLite event log for agent runs."""

    def __init__(self, database_path: Path | str) -> None:
        """Create an event log and ensure the schema exists."""
        self._database_path = str(database_path)
        self._initialize()

    def append_transition(self, transition: StateTransition) -> int:
        """Persist a state transition and return its event id."""
        return self.append_event(
            agent_id=transition.agent_id,
            run_id=transition.run_id,
            event_type="state_transition",
            payload={
                "from_state": transition.from_state.value,
                "to_state": transition.to_state.value,
                "payload": transition.payload,
            },
        )

    def append_event(
        self,
        agent_id: str,
        run_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> int:
        """Persist a generic JSON event and return its id."""
        with sqlite3.connect(self._database_path) as connection:
            cursor = connection.execute(
                """
                INSERT INTO agent_events (timestamp, agent_id, run_id, event_type, payload)
                VALUES (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), ?, ?, ?, ?)
                """,
                (agent_id, run_id, event_type, json.dumps(payload, sort_keys=True)),
            )
            connection.commit()
            event_id = cursor.lastrowid
            if event_id is None:
                raise RuntimeError("SQLite did not return an event id")
            return event_id

    def list_events(self, run_id: str | None = None) -> list[dict[str, Any]]:
        """Return persisted events, optionally scoped to one run."""
        query = "SELECT id, timestamp, agent_id, run_id, event_type, payload FROM agent_events"
        parameters: tuple[str, ...] = ()
        if run_id is not None:
            query += " WHERE run_id = ?"
            parameters = (run_id,)
        query += " ORDER BY id ASC"
        with sqlite3.connect(self._database_path) as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [
            {
                "id": row[0],
                "timestamp": row[1],
                "agent_id": row[2],
                "run_id": row[3],
                "event_type": row[4],
                "payload": json.loads(row[5]),
            }
            for row in rows
        ]

    def _initialize(self) -> None:
        """Create the event-log schema when missing."""
        with sqlite3.connect(self._database_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_agent_events_run_id ON agent_events(run_id, id)"
            )
            connection.commit()
