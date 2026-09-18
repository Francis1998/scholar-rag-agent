# Discover persisted research runs

`GET /runs` recovers run IDs from the existing SQLite event log. Browse bounded
query previews and recorded states, then follow the existing events and evidence
export routes. You do not have to keep an ID from a terminal session or rerun a
model to find a previous answer.

This is **persisted run discovery**, not a background-job dashboard. A saved
`REASONING` record can belong to an interrupted process; it does not mean a worker
is currently running. There is no checkpoint resume, replay, conversation memory,
new database, or LangGraph integration.

![Synthetic offline run-history discovery and restart demonstration](../assets/run-history.gif)

**GIF provenance:** a generated illustration of actual synthetic/offline demo
output, not a screen recording, research UI, live inference, or scientific
quality result. Its statuses, cursor, counts, and restart comparison come from
the requests below. The incomplete trace is explicitly an inserted fixture,
not a claim that a job is running.

## Copyable offline workflow

From the repository root, with Python 3.11+ and
[uv](https://docs.astral.sh/uv/) installed:

```bash
uv sync --extra dev
DEMO_DIR="$(mktemp -d)/run-history-demo"
uv run python -m scripts.demo_run_history --output-dir "$DEMO_DIR"
printf 'Saved demo directory: %s\n' "$DEMO_DIR"
uv run python -m json.tool "$DEMO_DIR/page-1.json"
uv run python -m json.tool "$DEMO_DIR/page-2.json"
uv run python -m json.tool "$DEMO_DIR/bundle.json"
```

The script drives the real FastAPI routes with `TestClient`, without a server or
external calls. It explicitly clears model credentials, including inherited
environment and `.env` values, and uses the fake adapter. It ingests a synthetic
note, completes a query, injects a synthetic generation failure into another
real query, and appends a clearly labeled partial-trace fixture.

It then recreates `AppContainer` against the same SQLite file, discovers the
three records across two pages, filters `DONE`, and follows the recovered run's
events/export links. The JSON export must be byte-identical to the artifact
captured before recreation. The fake answer is a plumbing example, not a
scientific synthesis.

| Artifact | What it demonstrates |
| --- | --- |
| `history.sqlite3` | The existing durable event/document/graph database; no separate run table |
| `page-1.json` | Newest two runs: recorded `REASONING` and `ERROR`; non-null continuation |
| `page-2.json` | Older `DONE` run; terminal `next_cursor: null` |
| `done.json` | The same completed summary found through the state filter |
| `events.json` | Existing event route, reached using the discovered ID |
| `bundle.json`, `bundle.md` | Existing saved-evidence downloads, not regenerated answers |
| `transcript.txt` | Actual demonstration output used to render the GIF |

Choose a fresh output directory. The demo refuses to replace any of its named
artifacts; the renderer also refuses an existing destination. `run-history-demo/`
is ignored by Git, but other chosen directories may not be. Do not commit
private research artifacts.

Regenerate only this animation using Pillow, already included in the dev extra:

```bash
uv run python -m scripts.create_run_history_gif \
  --transcript "$DEMO_DIR/transcript.txt" \
  --output "$DEMO_DIR/run-history.gif"
```

The renderer checks the four expected panels and rejects text that would be
clipped. It does not regenerate other project animations. With the same
transcript and locked Pillow version, repeated renders are byte-identical.
Actual run IDs and timestamps in the saved JSON vary between demo executions.

## Explore through HTTP, then restart

Reuse the demo database rather than creating another database path:

```bash
export SCHOLAR_RAG_DATABASE_PATH="$DEMO_DIR/history.sqlite3"
OPENAI_API_KEY= ANTHROPIC_API_KEY= GEMINI_API_KEY= MOONSHOT_API_KEY= \
SCHOLAR_RAG_DEFAULT_MODEL=fake \
  uv run uvicorn api.main:app --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
BASE_URL=http://127.0.0.1:8000
curl --fail-with-body --silent --show-error "$BASE_URL/runs?limit=2" \
  | uv run python -m json.tool
curl --fail-with-body --silent --show-error "$BASE_URL/runs?state=DONE&limit=20" \
  | uv run python -m json.tool
```

Copy `next_cursor` from a response into `?limit=2&cursor=<value>` to continue.
The following real client example discovers a completed run and follows its
links without retaining an old query response:

```bash
uv run python - <<'PY'
import httpx

with httpx.Client(base_url="http://127.0.0.1:8000", trust_env=False) as client:
    response = client.get("/runs", params={"limit": 1, "state": "DONE"})
    response.raise_for_status()
    page = response.json()
    if not page["runs"]:
        raise SystemExit("No recorded DONE runs in this database.")
    run = page["runs"][0]
    print(run["run_id"], run["recorded_state"], repr(run["query_summary"]))
    if run["events_url"] is None or run["export_url"] is None:
        raise SystemExit("This legacy ID cannot be addressed by the existing HTTP routes.")
    events = client.get(run["events_url"])
    events.raise_for_status()
    print("Recorded events:", len(events.json()))
    evidence = client.get(run["export_url"])
    if evidence.status_code == 409:
        print("This record is not exportable:", evidence.json()["detail"])
    else:
        evidence.raise_for_status()
        print("Saved provider:", evidence.json()["generation"]["provider"])
PY
```

Stop the server with Ctrl-C. In that same terminal, run the identical server
command again with the same `SCHOLAR_RAG_DATABASE_PATH`; then repeat `GET /runs`.
No run-ID cache is necessary. Pointing at a new database produces a different
catalog, not recovery of the old database. A run with no persisted event cannot
be discovered, and process-local work after the last event cannot be recovered.
Back up the SQLite database using SQLite-aware tooling; this feature is not a
backup/restore or event-repair mechanism.

## API contract

`GET /runs` returns `RunHistoryPage`, also documented in `/docs` and `/openapi.json`.
It does not return answers, source text, error bodies, configuration, raw
metadata, provider payloads, or settings.

| Query parameter | Contract |
| --- | --- |
| `limit` | Default **20**, inclusive range **1..100** |
| `cursor` | Optional positive signed-64-bit integer, **1..9223372036854775807**; exclusive first-event-ID boundary |
| `state` | Optional exact uppercase `AgentState`: `IDLE`, `PLANNING`, `RETRIEVING`, `REASONING`, `ANSWERING`, `DONE`, or `ERROR` |

Response shape (illustrative values; not a claim that this ID exists):

```json
{
  "runs": [
    {
      "run_id": "synthetic-example",
      "agent_id": "local-agent",
      "first_event_id": 9,
      "recorded_state": "ERROR",
      "query_summary": "Compare two synthetic notes",
      "query_truncated": false,
      "started_at": "2026-09-18T16:00:00Z",
      "updated_at": "2026-09-18T16:00:01Z",
      "event_count": 6,
      "events_url": "/runs/synthetic-example/events",
      "export_url": "/runs/synthetic-example/export"
    }
  ],
  "next_cursor": null
}
```

| Field | Meaning |
| --- | --- |
| `run_id`, `agent_id` | Exact persisted identities, never shortened; one agent per run |
| `first_event_id` | Immutable minimum event ID for this run; the creation-order key |
| `recorded_state` | Target state of the highest-ID state transition, or `null` for event-only records |
| `query_summary` | First **300 Unicode characters** of the earliest `PLANNING` query, with original whitespace/control characters preserved in JSON |
| `query_truncated` | `true` exactly when the query exceeds 300 characters; no ellipsis is added to the prefix |
| `started_at`, `updated_at` | Timezone-aware timestamps of the first and last events by ID, not minimum/maximum wall-clock values |
| `event_count` | All persisted events for this run at the time of this read, including snapshots and later diagnostic events |
| `events_url`, `export_url` | Percent-encoded relative navigation URLs; **not** evidence-availability guarantees |
| `next_cursor` | Last returned run's `first_event_id`, only if another matching row exists; otherwise `null` |

The default state machine does not persist an initial `IDLE` transition. A legacy
record that explicitly records `IDLE` can nevertheless be inspected/filtered as
such. Missing or null planning queries produce `query_summary: null` and
`query_truncated: false`; an explicitly saved empty query remains `""`. The
catalog never guesses a query from decision logs, source text, or model output.
There is no `is_running`, inferred `DONE`, or automatic terminal-state repair.

`DONE` means a `DONE` transition was recorded, **not** that the evidence exporter
can accept the trace. Legacy completed runs without snapshots are discoverable
but their export links return the existing `409 snapshot_unavailable`. Failed,
incomplete, inconsistent, or unsupported exports retain their existing error
contract. Exports freeze their own completed trace; the catalog instead reports
the latest recorded transition. See the [evidence export guide](EVIDENCE_EXPORT_GUIDE.md).

Legacy IDs containing `/`, or equal to `.` or `..`, remain visible but have null
navigation URLs because the existing HTTP routes cannot address them reliably.
Use the exact ID with the Python event/export interfaces when needed. The
catalog does not change those routes or silently rewrite IDs.

## Pagination and concurrent changes

Ordering is **newest-created first**, defined by descending minimum event ID per
run, not a timestamp or the last event ID. The cursor means
`first_event_id < cursor`, not an offset or a page number. Timestamp ties, clock
adjustments, and interleaved transitions do not change this order.

For an unfiltered traversal of an append-only database, new runs created after
page one appear ahead of the cursor and do not intrude into later pages. Appends
to an older run update its summary without moving its creation key. Following
the returned cursor therefore does not duplicate or skip the previously
existing runs. Refresh without a cursor to discover newly created runs.

The latest-state filter is applied **before** pagination, including the one-row
lookahead used to determine `next_cursor`. An empty database, exhausted cursor,
or no matching state returns `{"runs": [], "next_cursor": null}` with HTTP 200.
An exact full final page also has a null cursor.

Each page's summary data comes from one SQLite SELECT and one consistent read
snapshot. **A multi-page traversal is not an immutable point-in-time snapshot
of changing states.** A not-yet-seen `PLANNING` run that records `ERROR` will
disappear from subsequent `state=PLANNING` pages. A record entering the filter
ahead of an existing cursor requires a first-page refresh. Use the same filter
through a traversal; changing it starts a different selection.

Cursors are documented integers, not signed authorization tokens. They need
not identify an existing run. Reusing one against another database, deleting
events, reassigning IDs, or manually changing a run's identity is outside the
append-only traversal guarantee. There is no cross-request snapshot token,
retention policy, background queue, total-count calculation, or resume action.

## Python contract

`SQLiteRunHistory` requires an existing SQLite event database. It opens it
read-only and closes the connection after each page. It creates no tables or
indexes, adds no storage to synchronize, and needs no event backfill.

```bash
uv run python - "$DEMO_DIR/history.sqlite3" <<'PY'
import sys
from pathlib import Path

from agent.models import AgentState
from storage.run_history import SQLiteRunHistory

history = SQLiteRunHistory(Path(sys.argv[1]))
cursor = None
while True:
    page = history.list_runs(limit=2, cursor=cursor)
    for run in page.runs:
        print(run.run_id, run.recorded_state, repr(run.query_summary))
    if page.next_cursor is None:
        break
    cursor = page.next_cursor

done = history.list_runs(limit=20, state=AgentState.DONE)
print(done.model_dump_json(indent=2))
PY
```

The keyword-only method is
`list_runs(*, limit=20, cursor=None, state=None) -> RunHistoryPage`.
`page.runs` contains typed `RunSummary` objects; timestamps are aware datetimes.
Invalid Python bounds, non-integer limits/cursors (including booleans), or states
raise Pydantic `ValidationError` before database access.

## Errors, bounds, and privacy

| Outcome | HTTP / Python behavior |
| --- | --- |
| Valid empty selection | HTTP 200; empty `runs`, null cursor |
| Invalid limit, cursor, or state | HTTP 422 / Pydantic `ValidationError` |
| Malformed or unsupported inspected summary | HTTP 409, `detail.code="invalid_run_record"` / `RunHistoryError` |
| Missing/inaccessible database, missing table, locking or other SQLite failures | Database exception/server error, never a fabricated empty catalog |

Transition projection checks JSON/object shape, known source/target states, an
object nested payload, and text/missing/null planning-query values. Malformed
transitions are not silently hidden by a state filter. Invalid selected
metadata, conflicting agent identities, invalid timestamps, or IDs outside
**1..256 characters** also fail rather than being clipped or guessed. Inspection
includes the lookahead row: a bad lookahead can fail that page instead of
returning a misleading continuation.

This is not full state-machine replay or evidence-integrity validation. The
catalog does not check transition-chain legality, parse non-transition payloads,
or validate snapshots/model output; the existing exporter remains authoritative
for evidence downloads. Valid event-only records and legacy traces without a
planning query are supported, not treated as corruption. Future unknown state
names are explicit unsupported records, not mapped to `DONE` or `ERROR`.

At most **101 bounded projected rows** enter Python for a request (100 results
plus lookahead). Query text is projected as a bounded byte prefix before decoding
and character truncation; Unicode and embedded NULs survive. UTF-8 and existing
UTF-16 SQLite databases are supported. Full evidence snapshots are neither
loaded into Python nor deserialized by listing; `list_events(None)` is not used.
The SQLite grouping and transition-validation work **still scales with stored
events**. Page size and returned fields are bounded, not database scan time,
database size, retention, or the cost of arbitrarily large transition JSON.

Successful catalog responses and catalog-data errors use `Cache-Control:
no-store` and `X-Content-Type-Options: nosniff`. Query previews and identifiers
can still reveal sensitive research. This is not automatic secret redaction:
anything typed into a query can appear in its preview. Treat strings as
untrusted text, not HTML or instructions. Following an events/export link
reveals much more, including full saved source text and metadata.

There is **no authentication or tenant isolation**. Keep the service on loopback
or in a trusted environment, protect the database, and review permissions and
content before sharing. The listing itself performs no retrieval, LLM calls, or
corpus reads. Normal `AppContainer` startup still rebuilds in-memory indexes from
the persisted corpus, as before; this feature does not change that startup cost.

## Research and portfolio example

On one day, compare methods in a small corpus you are allowed to process. On a
later day, reopen the same database, use query previews to find the comparison,
inspect its recorded state, and download the evidence for human review. A failed
or interrupted run remains visible instead of disappearing with terminal output;
start a **new** query if another attempt is needed.

For a portfolio, use the synthetic demo artifacts to explain the engineering:
stable creation-order cursors, filtering before pagination, explicit legacy
states, minimal response exposure, and restart discovery leading to an unchanged
evidence export. The supplied regressions cover real successful/failed API
queries, listing during an in-flight query, overlapping SQLite writers,
interleaved/tied-time pagination, Unicode bounds, and refusal to overwrite demo
artifacts. None of these is a scientific-accuracy or performance benchmark.
Follow the [research workflow](RESEARCH_WORKFLOW_GUIDE.md) for corpus selection,
warnings, evidence review, and an honest presentation of retrieval limitations.

## Inspiration and explicit non-goals

References checked **2026-09-18 America/Los_Angeles**:

- [LangGraph](https://github.com/langchain-ai/langgraph) and its
  [persistence documentation](https://docs.langchain.com/oss/python/langgraph/persistence)
  distinguish persistent storage from in-memory state lost at restart. The
  motivation here is inspectable, recoverable **discovery of saved runs**.
- [LangSmith Agent Server](https://docs.langchain.com/langsmith/agent-server)
  exposes persisted run resources and state inspection within a much broader
  managed execution system.

This project retains FastAPI, Pydantic, HTTPX, SQLite, and its custom state
machine. It does **not** add LangGraph, checkpoint resume/replay, managed server
features, conversation memory, scheduling, or feature parity with those systems.
