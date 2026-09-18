"""Bounded run summaries projected from the existing SQLite event log."""

import codecs
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Annotated
from urllib.parse import quote

from pydantic import AwareDatetime, BaseModel, Field, ValidationError

from agent.models import AgentState

DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100
MAX_QUERY_CHARACTERS = 300
MAX_ID_CHARACTERS = 256
MAX_EVENT_ID = 2**63 - 1
_QUERY_PREFIX_BYTES = 4 * (MAX_QUERY_CHARACTERS + 1)

PageSize = Annotated[int, Field(strict=True, ge=1, le=MAX_PAGE_SIZE)]
EventID = Annotated[int, Field(strict=True, ge=1, le=MAX_EVENT_ID)]
Identity = Annotated[str, Field(strict=True, min_length=1, max_length=MAX_ID_CHARACTERS)]


class RunSummary(BaseModel):
    """Last recorded state and bounded query preview, never a liveness assertion."""

    run_id: Identity
    agent_id: Identity
    first_event_id: EventID
    recorded_state: AgentState | None
    query_summary: str | None = Field(default=None, max_length=MAX_QUERY_CHARACTERS)
    query_truncated: bool = False
    started_at: AwareDatetime
    updated_at: AwareDatetime
    event_count: int = Field(ge=1)
    events_url: str | None = None
    export_url: str | None = None


class RunHistoryPage(BaseModel):
    """Creation-ordered page, with an exclusive first-event-ID continuation."""

    runs: list[RunSummary] = Field(max_length=MAX_PAGE_SIZE)
    next_cursor: EventID | None


class _PageRequest(BaseModel):
    limit: PageSize = DEFAULT_PAGE_SIZE
    cursor: EventID | None = None
    state: AgentState | None = None


class RunHistoryError(ValueError):
    """Malformed or unsupported projected data, without exposing saved payloads."""

    code = "invalid_run_record"

    def __init__(self) -> None:
        super().__init__("Saved run summary data is inconsistent, invalid, or unsupported.")


_CATALOG_SQL = """
WITH runs AS (
    SELECT run_id, min(id) AS first_event_id, max(id) AS last_event_id,
           count(*) AS event_count, min(agent_id) != max(agent_id) AS mixed_agents,
           max(CASE WHEN event_type = 'state_transition' THEN id END) AS transition_id
    FROM agent_events
    GROUP BY run_id
    HAVING :cursor IS NULL OR min(id) < :cursor
),
transitions AS (
    SELECT e.id, e.run_id,
           CASE WHEN json_valid(e.payload) THEN
               CASE WHEN json_type(e.payload, '$.to_state') = 'text'
                    AND json_extract(e.payload, '$.to_state')
                        IN (SELECT value FROM json_each(:states))
                    THEN json_extract(e.payload, '$.to_state') END
           END AS recorded_state,
           CASE WHEN json_valid(e.payload) THEN
               CASE WHEN json_type(e.payload) = 'object'
                    AND json_type(e.payload, '$.from_state') = 'text'
                    AND json_extract(e.payload, '$.from_state')
                        IN (SELECT value FROM json_each(:states))
                    AND json_type(e.payload, '$.to_state') = 'text'
                    AND json_extract(e.payload, '$.to_state')
                        IN (SELECT value FROM json_each(:states))
                    AND json_type(e.payload, '$.payload') = 'object'
                    AND (json_extract(e.payload, '$.to_state') != 'PLANNING'
                         OR coalesce(json_type(e.payload, '$.payload.query'), 'null')
                            IN ('null', 'text'))
                    THEN 0 ELSE 1 END
               ELSE 1 END AS invalid_transition
    FROM agent_events AS e
    JOIN runs ON runs.run_id = e.run_id
    WHERE e.event_type = 'state_transition'
),
transition_summaries AS (
    SELECT run_id, max(invalid_transition) AS invalid_transition,
           min(CASE WHEN recorded_state = 'PLANNING' THEN id END) AS planning_id
    FROM transitions
    GROUP BY run_id
),
candidates AS (
    SELECT runs.first_event_id, runs.event_count,
           substr(CAST(first.run_id AS BLOB), 1, :identity_bytes) AS run_id,
           substr(CAST(first.agent_id AS BLOB), 1, :identity_bytes) AS agent_id,
           substr(CAST(first.timestamp AS BLOB), 1, 260) AS started_at,
           substr(CAST(last.timestamp AS BLOB), 1, 260) AS updated_at,
           latest.recorded_state,
           runs.mixed_agents OR coalesce(ts.invalid_transition, 0) AS invalid_record,
           CASE WHEN json_valid(planning.payload) THEN
               CASE WHEN json_type(planning.payload, '$.payload.query') = 'text'
                    THEN coalesce(
                        substr(CAST(json_extract(planning.payload, '$.payload.query') AS BLOB),
                               1, :query_bytes), X'') END
           END AS query_prefix
    FROM runs
    JOIN agent_events AS first ON first.id = runs.first_event_id
    JOIN agent_events AS last ON last.id = runs.last_event_id
    LEFT JOIN transitions AS latest ON latest.id = runs.transition_id
    LEFT JOIN transition_summaries AS ts ON ts.run_id = runs.run_id
    LEFT JOIN agent_events AS planning ON planning.id = ts.planning_id
)
SELECT first_event_id, event_count, run_id, agent_id, started_at, updated_at,
       recorded_state, invalid_record, query_prefix
FROM candidates
WHERE :state IS NULL OR recorded_state = :state OR invalid_record
ORDER BY first_event_id DESC
LIMIT :fetch_limit
"""


def _decode_prefix(value: object, encoding: str, *, byte_limit: int | None = None) -> str:
    if not isinstance(value, bytes):
        raise RunHistoryError()
    try:
        # Only a full query prefix can legitimately end inside a code point.
        final = byte_limit is None or len(value) < byte_limit
        return codecs.getincrementaldecoder(encoding)().decode(value, final=final)
    except UnicodeDecodeError as exc:
        raise RunHistoryError() from exc


class SQLiteRunHistory:
    """Read-only catalog; the append-only event table remains the only source of truth."""

    def __init__(self, database_path: Path | str) -> None:
        """Use an existing event database without creating tables or backfilling records."""
        self._database_uri = Path(database_path).resolve().as_uri() + "?mode=ro"

    def list_runs(
        self,
        *,
        limit: int = DEFAULT_PAGE_SIZE,
        cursor: int | None = None,
        state: AgentState | None = None,
    ) -> RunHistoryPage:
        """Read one SQLite snapshot; states may change between pagination requests."""
        options = _PageRequest(limit=limit, cursor=cursor, state=state)
        with closing(sqlite3.connect(self._database_uri, uri=True)) as connection:
            encoding: str = connection.execute("PRAGMA encoding").fetchone()[0]
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                _CATALOG_SQL,
                {
                    "cursor": options.cursor,
                    "state": options.state,
                    "states": json.dumps(list(AgentState)),
                    "fetch_limit": options.limit + 1,
                    "identity_bytes": 4 * (MAX_ID_CHARACTERS + 1),
                    "query_bytes": _QUERY_PREFIX_BYTES,
                },
            ).fetchall()
        summaries = []
        for row in rows:
            if row["invalid_record"]:
                raise RunHistoryError()
            record = dict(row)
            for field in ("run_id", "agent_id", "started_at", "updated_at"):
                record[field] = _decode_prefix(record[field], encoding)
            try:
                summary = RunSummary.model_validate(record)
            except ValidationError as exc:
                raise RunHistoryError() from exc
            if row["query_prefix"] is not None:
                query = _decode_prefix(
                    row["query_prefix"], encoding, byte_limit=_QUERY_PREFIX_BYTES
                )
                summary.query_summary = query[:MAX_QUERY_CHARACTERS]
                summary.query_truncated = len(query) > MAX_QUERY_CHARACTERS
            if "/" not in summary.run_id and summary.run_id not in {".", ".."}:
                segment = quote(summary.run_id, safe="")
                summary.events_url = f"/runs/{segment}/events"
                summary.export_url = f"/runs/{segment}/export"
            summaries.append(summary)
        more = len(summaries) > options.limit
        page = summaries[: options.limit]
        return RunHistoryPage(
            runs=page,
            next_cursor=page[-1].first_event_id if more else None,
        )
