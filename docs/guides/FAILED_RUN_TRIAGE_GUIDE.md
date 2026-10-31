# Diagnose failed or unfinished saved runs

**Inspect first; do not resend `/query` to see whether it works now.** That is a
new run and can make paid provider requests. Reading saved history, events, or
exports does not retry generation, resume a worker, or repair an earlier run.
This workflow was reviewed **2026-10-30 America/Los_Angeles**, during the
maintainer pass begun October 29. No live model inference was used.

## 1. Classify the response before looking for an answer

For an existing `/query` response, inspect `result.run_id`, `result.state`, and
`result.error`, not just HTTP status. Only a `DONE` result has completed this
state machine; its answer can still have `ungrounded` or other warnings.

| Observation | Meaning and next check |
| --- | --- |
| HTTP 422 from query/schema validation | Invalid input, such as a blank query, empty scope, or invalid evidence bound. Validation rejects it before starting a run. Do not invent a run ID from the error body. |
| Collection/screened-scope 404, 409, or 503 before execution | Scope resolution failed before the runner. Do not drop the selection and silently search the whole corpus. See [scope selection](SCREENED_COLLECTION_GUIDE.md). |
| HTTP 200 with `result.state == "ERROR"` | The runner caught an operational failure. Normally its `ERROR` transition is durable; inspect that exact run's events. HTTP success is not answer success. |
| Lost response, client/proxy timeout, or server error | The client cannot infer whether execution started, finished, or reached a provider. Recover history before considering any new request. |
| Saved nonterminal state, or `recorded_state: null` | Only the last recorded transition, or no transition, is known. This is not a worker heartbeat, proof of an active job, or a resumable checkpoint. |
| `DONE`, but export is rejected | Completion alone does not establish a valid frozen capture. Preserve the export's error code; do not reconstruct an old answer from current documents. |

These are different boundaries: a runner timeout normally becomes HTTP 200
`ERROR`, whereas a retrieval **preview** timeout is HTTP 504. A storage failure
can also prevent the attempted error transition from being saved. Do not assume
every server error has a corresponding durable `ERROR`.

## 2. Find the exact saved run

Use the running application's same database after restart. A different
`SCHOLAR_RAG_DATABASE_PATH` can look like missing history. Check the configured
path locally without printing credentials or starting an application against a
guessed/new database. Use the [backup/recovery guide](SQLITE_BACKUP_RESTORE_GUIDE.md)
before investigating suspected database damage.

| Read-only route | Purpose |
| --- | --- |
| `GET /explore/runs` | Native saved-run browser; start with all states, then filter if useful. |
| `GET /runs?limit=20` | JSON history, newest **created** runs first, not most recently updated. |
| `GET /runs?state=ERROR&limit=20` | Failed recorded states only; this deliberately excludes incomplete runs. |
| `GET /runs?limit=20&cursor=<next_cursor>` | Continue using the returned exclusive first-event-ID cursor, not an offset or timestamp. Replace the placeholder. |
| `GET /runs/{run_id}/events` | Exact run's events in ascending event-ID order. Prefer the selected row's `events_url`. |
| `GET /runs/{run_id}/export?format=json` | Validate a completed frozen bundle; an inspection request, not a repair. Prefer `export_url`. |
| `GET /explore/run?run_id=<exact-ID>` | Native frozen evidence reader; failed/incomplete runs receive an error page rather than an invented answer. |

Choose by exact `run_id`, corroborating `agent_id`, `started_at`, `updated_at`,
query summary, and event count. Summaries contain at most 300 query characters
and report `query_truncated`; identical summaries are not identical requests.
Do not use the first row as "my failed run." State-filtered pagination is not a
frozen snapshot: concurrent transitions can change membership between pages.
Use the [history guide](RUN_HISTORY_GUIDE.md) for pagination and the
[browser guide](SAVED_RUN_BROWSER_GUIDE.md) for mounted URLs and exact-ID handling.

The raw events route currently returns **HTTP 200 and `[]` for an unknown ID**;
the export route returns **404 `run_not_found`**. An empty event list therefore
does not mean a successful empty run. There is no `/runs/{id}/retry`, resume, or
failed-answer repair endpoint. Do not guess one.

## 3. Read the ordered trace, including its first terminal event

Keep the original event IDs and order. Each event has `event_type` and an outer
`payload`. For `state_transition`, that outer object contains `from_state`,
`to_state`, and another `payload`. Thus a terminal diagnostic is at
`event["payload"]["payload"]["error"]`, not the top level. An
`evidence_snapshot` or `generation_record` has its own payload directly.

| Record | What it establishes |
| --- | --- |
| Transition to `PLANNING` | Original query, resolved `document_ids`, requested evidence policy, and effective source/hop/phase-timeout configuration. This is not a dump of credentials or all settings. |
| `decision_log` | Recorded plan, tasks, observation, and operational rationale, not hidden model reasoning. |
| Transition to `REASONING` | IDs of retrieved candidate chunks; these are not necessarily the final reranked/collapsed/capped context. |
| `evidence_snapshot` | Exact prepared request and full final-context passages, captured before answer generation in the built-in executor. It can survive a failed run. |
| `generation_record` | Returned provider/model identity and proposed claim/citation associations, recorded **after generation and grounding**. |
| First transition to `ERROR` or `DONE` | Historical terminal boundary used by the exporter. Later appended data cannot turn an earlier failure into a successful historical export. |

Find that first terminal transition, then inspect its `from_state`, nested error,
and preceding records. History reports the **latest** transition; it does not
validate the whole trace. If malformed/manual records make history and export
disagree, preserve both observations instead of editing SQLite to make them agree.
The [export contract](EVIDENCE_EXPORT_GUIDE.md#explicit-error-policy) distinguishes:

| Export outcome | Action |
| --- | --- |
| 409 `run_failed` | Preserve the terminal failure, even if a snapshot exists. There is no failed-run "completed bundle." |
| 409 `run_incomplete` | No terminal boundary was found. Check the owning process/server logs locally; do not infer liveness from its age. |
| 409 `snapshot_unavailable` | Legacy/no-capture completion. Today's corpus cannot supply the missing historical context. |
| 409 `invalid_run_record` | Inconsistent, malformed, or unsupported capture; stop normal export handling and retain a backup for investigation. |
| 422 for an unsupported format | Fix the inspection request's format, not the saved run. |

## 4. Localize the failure without overstating what was observed

The normal path is `IDLE -> PLANNING -> RETRIEVING -> REASONING -> ANSWERING -> DONE`.
Nonterminal states can instead transition to `ERROR`. There is no automatic
state-machine retry or transition from `ERROR` back to `PLANNING`.

| Last phase / saved evidence | Likely boundary to investigate; safe next action |
| --- | --- |
| `PLANNING -> ERROR`, often without `decision_log` | Analyzer/planner failure, unsupported custom executor, or a request-contract guard. Read the diagnostic and local configuration; do not change provider keys as a generic fix. |
| `RETRIEVING -> ERROR` | Hybrid/graph retrieval, scope checks, source bounds, or retrieval timeout. Inspect the selected current corpus and exact IDs separately from the historical record. |
| `REASONING -> ERROR`, no snapshot | Context preparation/reranking/capture, an early cancellation, or a custom executor without capture support. This is not automatically a provider outage. |
| Snapshot plus `code == "insufficient_evidence_documents"` | The final context failed the requested distinct-document minimum. Inspect `evidence_assessment`; the built-in guarded path stops before answer generation. Do not lower the minimum silently. |
| Snapshot then `ERROR` | Generation, provider waits, grounding, later validation, cancellation, or persistence may have failed. Inspect the actual error, not just the presence of context. |
| `generation_record` or `ANSWERING` but no `DONE` | Generation/grounding progressed, but completion was not durably established. Preserve any captured evidence; do not present it as a valid completed export. |

**No generation record is not proof of no provider request or no charge.** The
record is written after grounding; a request can time out or succeed remotely
before the local trace is completed. A missing snapshot is likewise not a
universal no-call guarantee for custom executors or interrupted persistence.
The explicit minimum-evidence guard is a narrower, testable no-generation case.

Default retrieval and reasoning limits are 30 and 60 seconds respectively;
inspect the captured `configuration` for this run rather than assuming defaults.
Preparation, generation, provider admission/backoff, and grounding share the
reasoning budget; it is not a fresh budget for every substep. Planning is
synchronous and is not wrapped in these phase timeouts. Async deadlines are
cooperative, not a hard kill of blocking code or a promise to cancel remote billing.

Errors such as `retrieval timed out after 30.0s` and
`reasoning timed out after 60.0s` identify the bounded phase, not its exact
sub-operation. Cooperative cancellation records `ERROR`; external asyncio task
cancellation attempts an `ERROR` write and then propagates, so it need not return
a normal JSON result. An abrupt process exit can leave a nonterminal trace.

## 5. Separate provider diagnostics from retry authorization

An application-level `/query` HTTP 200 and an upstream provider HTTP 200 are
different observations. Live adapters reject unusable final output with
`ProviderResponseError`: invalid JSON/envelopes, blank or malformed answer text,
thinking/tool-only output, explicit truncation, and supported block/refusal
signals. These diagnostics name a provider/category, not its raw body, hidden
thinking, or tool arguments. They are **not retried** and do not trigger fake
answers or provider failover.

Existing HTTP adapters can retry transport errors and HTTP **429, 500, 502, 503,
504** within the *same invocation*: at most **three retries after the initial
attempt**. Permanent responses such as 400/401/403/404 are not retried. Minimum
backoffs are 0.25, 0.5, and 1 second; a valid `Retry-After` can extend them.
Every attempt consumes per-instance rate admission. These waits count against
the runner's reasoning timeout, so a long hint can end the run before another
attempt. The 60-second HTTPX timeout is not a whole-invocation deadline.

The durable event log does **not** contain a per-attempt request/billing ledger.
Do not infer "four attempts happened" from one terminal HTTP error; the offline
mock below counts them directly. Do not add a client retry loop around `/query`.
An HTTP transport failure may occur after a provider accepted a request.
See [provider retry details](PROVIDER_MODELS_GUIDE.md#retry-after-and-timeouts).

For an authentication/model-access error, inspect only the selected provider,
requested ID, HTTP status/category, and configuration precedence locally.
`/query` uses the reasoning route, preferring configured Anthropic before the
normal fallback chain. Setting `default_model="fake"` alone does not disable a
configured Anthropic adapter. The examples below use the existing
`offline_settings()` helper, which ignores environment, `.env`, and file-secret
sources and explicitly empties all four model keys. Check current IDs in the
[dated provider guide](PROVIDER_MODELS_GUIDE.md); catalogs and mocks establish
neither account entitlement nor live compatibility.

## 6. Preserve history and choose a bounded next action

1. Retain the exact old run ID, terminal event ID/state, diagnostic category, and
   captured request/scope/budgets privately. Back up suspect storage to a **new**
   destination; never delete, rewrite, backfill, or append a synthetic `DONE` to
   real history.
2. Use current-corpus readers or an explicit `POST /retrieve` preview when the
   question is about today's evidence. Preview performs retrieval/preparation,
   not answer generation or agent-event writes, and rejects LLM-backed HyDE
   rather than silently substituting an algorithm. Its output is **not** a
   reconstruction of the failed run. See the [preview guide](RETRIEVAL_PREVIEW_GUIDE.md).
3. If a new run is necessary, require the user's separate authorization for the
   provider/account, permitted data, exact query, resolved document IDs, evidence
   policy, and existing budgets. Record the deliberate configuration fix and
   verify routing first. Do not widen scope, remove a guard, raise budgets,
   switch providers, or retry until success without that decision.
4. Submit the new request only as an explicit new operation, with awareness that
   its adapter may internally retry. Record its **new** run ID and inspect its
   result independently. Even identical query/scope can retrieve changed source
   content; a later `DONE` does not repair or replace the earlier `ERROR`.

### Privacy before sharing

Use loopback/local or otherwise trusted access. These routes are not a
multi-user authorization boundary. Saved queries, query summaries, scope IDs,
full context, paper metadata, generated answers, and arbitrary exception strings
may contain private material. The raw events endpoint is not a sanitized export
and does not inherit the browser/history endpoints' `no-store` protections.
FastAPI validation errors can echo submitted input.

The runner stores `str(exc)` for ordinary exceptions; **not all errors are
redacted** just because `ProviderResponseError` is. Review URLs, filesystem
paths, custom-provider messages, and server logs locally. Browser escaping,
digests, and an append-only API do not anonymize data or make SQLite tamper-proof.

Share a manually reviewed, minimal report: revision, sanitized phase/category,
HTTP status, observed event types/counts, whether a snapshot exists, effective
limits, and a synthetic reproducer. Omit keys, headers, `.env`, database copies,
raw provider bodies, queries/context, and private IDs unless explicitly
authorized. Label absent attempt/billing evidence as **unknown**, not zero.

## Credential-free exercises

Run from the repository root with the documented development dependencies
installed (`uv sync --extra dev`). These two independent snippets use temporary
databases, synthetic text, explicit offline settings, in-process `TestClient`,
and transport guards. They open no server and call no external model API.
Temporary files are removed on exit; no personal database is selected.

### A. Rejected input, saved failures, incomplete history, and a separate fake run

The intentionally unfinished record is an **inserted fixture**, not a killed
worker or resumable job. Fault injection is confined to this temporary app.

```bash
uv run python - <<'PY'
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from agent.models import AgentState, StateTransition
from api.application import create_app
from api.dependencies import AppContainer
from scripts.demo_evidence_export import DEMO_QUERY, DEMO_TEXT, offline_settings

with (
    TemporaryDirectory(prefix="failed-run-triage-") as directory,
    patch.object(httpx.HTTPTransport, "handle_request", side_effect=AssertionError("network")),
    patch.object(
        httpx.AsyncHTTPTransport, "handle_async_request", side_effect=AssertionError("network")
    ),
):
    database = Path(directory) / "synthetic.sqlite3"
    app = create_app(offline_settings(database))
    container = app.state.container
    with TestClient(app) as client:
        invalid = client.post("/query", json={"query": "   "})
        missing = client.post(
            "/query",
            json={"query": DEMO_QUERY, "collection_id": "col_" + "0" * 32},
        )
        assert invalid.status_code == 422 and missing.status_code == 404
        assert container.event_log.list_events() == []
        print("Rejected input: 422; unknown collection: 404; saved events: 0")

        ingested = client.post(
            "/ingest/text",
            json={"title": "Synthetic triage", "source": "synthetic:triage", "text": DEMO_TEXT},
        )
        ingested.raise_for_status()
        payload = {"query": DEMO_QUERY, "document_ids": [ingested.json()["document_id"]]}
        with patch.object(
            container.llm, "generate", side_effect=RuntimeError("Synthetic generation failure")
        ) as injected:
            response = client.post("/query", json=payload)
        failed = response.json()["result"]
        assert response.status_code == 200 and failed["state"] == "ERROR"
        assert failed["error"] == "Synthetic generation failure" and injected.call_count == 1
        original = client.get(f"/runs/{failed['run_id']}/events").json()
        terminal = next(
            e for e in original
            if e["event_type"] == "state_transition"
            and e["payload"]["to_state"] in ("DONE", "ERROR")
        )
        assert terminal["payload"]["from_state"] == "REASONING"
        assert terminal["payload"]["payload"]["error"] == failed["error"]
        assert sum(e["event_type"] == "evidence_snapshot" for e in original) == 1

        with patch.object(
            container.llm, "generate", side_effect=AssertionError("generation")
        ) as gen:
            guarded = client.post("/query", json={**payload, "min_evidence_documents": 2})
        blocked = guarded.json()["result"]
        guard_events = client.get(f"/runs/{blocked['run_id']}/events").json()
        assert guarded.status_code == 200 and blocked["state"] == "ERROR" and gen.call_count == 0
        assert guard_events[-1]["payload"]["payload"]["code"] == "insufficient_evidence_documents"
        print("Saved ERROR: REASONING -> ERROR; snapshot: 1; minimum guard model calls: 0")

        container.event_log.append_transition(
            StateTransition(
                agent_id="synthetic-fixture", run_id="unfinished-fixture",
                from_state=AgentState.IDLE, to_state=AgentState.PLANNING,
                payload={"query": "Explicitly unfinished synthetic fixture"},
            )
        )
        app.state.container = AppContainer(offline_settings(database))
        before = container.event_log.list_events()
        page = client.get("/runs", params={"state": "ERROR", "limit": 20}).json()
        selected = next(row for row in page["runs"] if row["run_id"] == failed["run_id"])
        assert client.get(selected["events_url"]).json() == original
        rejected = client.get(selected["export_url"])
        incomplete = client.get("/runs/unfinished-fixture/export")
        assert rejected.status_code == incomplete.status_code == 409
        assert rejected.json()["detail"]["code"] == "run_failed"
        assert incomplete.json()["detail"]["code"] == "run_incomplete"
        assert client.get("/runs/unknown-fixture/events").json() == []
        assert client.get("/runs/unknown-fixture/export").status_code == 404
        assert client.get("/explore/runs").status_code == 200
        assert client.get("/explore/run", params={"run_id": failed["run_id"]}).status_code == 409
        assert container.event_log.list_events() == before
        print("Reopened history: failed/incomplete exports 409; inspection appended 0 events")

        # Explicit new synthetic operation, same query and scope, fault injection removed.
        fresh = client.post("/query", json=payload).json()["result"]
        assert fresh["state"] == "DONE" and fresh["run_id"] != failed["run_id"]
        bundle = client.get(f"/runs/{fresh['run_id']}/export").json()
        assert bundle["generation"]["provider"] == "fake"
        assert client.get(selected["events_url"]).json() == original
        assert client.get(selected["export_url"]).json()["detail"]["code"] == "run_failed"
        print("Separate fake run: DONE; original failed events unchanged; old export still rejected")
PY
```

### B. Provider HTTP success is not answer success; transient retries are bounded

This uses the real OpenAI **adapter code**, an empty key, and HTTPX
`MockTransport`, not the OpenAI service. The first response has malformed answer
text and a synthetic private marker. The second case returns 503 on every
attempt. Default backoff sleeps are retained; their duration is not a benchmark.

```bash
uv run python - <<'PY'
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from api.application import create_app
from llm.providers import OpenAIAdapter
from scripts.demo_evidence_export import DEMO_QUERY, DEMO_TEXT, offline_settings

original_client = httpx.AsyncClient
marker = "synthetic-private-provider-body"
attempts: list[str] = []
status = 200

def reply(request: httpx.Request) -> httpx.Response:
    attempts.append(request.method)
    return httpx.Response(
        status,
        json={"choices": [{"message": {"content": [{"type": "text", "text": {"private": marker}}]}}]},
    )

def mock_client(*, timeout: float) -> httpx.AsyncClient:
    return original_client(
        timeout=timeout, transport=httpx.MockTransport(reply), trust_env=False
    )

with (
    TemporaryDirectory(prefix="provider-triage-") as directory,
    patch.object(httpx.HTTPTransport, "handle_request", side_effect=AssertionError("network")),
    patch.object(
        httpx.AsyncHTTPTransport, "handle_async_request", side_effect=AssertionError("network")
    ),
    patch("llm.providers.httpx.AsyncClient", new=mock_client),
):
    app = create_app(offline_settings(Path(directory) / "synthetic.sqlite3"))
    with TestClient(app) as client:
        ingestion = client.post(
            "/ingest/text",
            json={"title": "Synthetic mock", "source": "synthetic:mock", "text": DEMO_TEXT},
        )
        ingestion.raise_for_status()
        payload = {"query": DEMO_QUERY, "document_ids": [ingestion.json()["document_id"]]}
        for status, expected_attempts in ((200, 1), (503, 4)):
            attempts.clear()
            with patch.object(
                app.state.container.llm, "generate", new=OpenAIAdapter(api_key="").generate
            ):
                response = client.post("/query", json=payload)
            result = response.json()["result"]
            assert response.status_code == 200 and result["state"] == "ERROR"
            assert len(attempts) == expected_attempts
            events = client.get(f"/runs/{result['run_id']}/events").json()
            assert events[-1]["payload"]["to_state"] == "ERROR"
            assert not any(e["event_type"] == "generation_record" for e in events)
            assert marker not in str(events) and marker not in result["error"]
            if status == 200:
                assert result["error"] == (
                    "openai returned an invalid response: malformed answer text."
                )
            print(f"Mock upstream {status}: {len(attempts)} attempt(s); API 200 ERROR; no body leak")
PY
```

## An honest portfolio checklist

Report the measured outcomes, including failures, rather than a polished answer
alone. The exercises above establish these specific local contracts:

| Evidence to show | Observed outcome |
| --- | --- |
| Pre-run rejection | Blank query 422 and missing collection 404; zero saved events before ingestion. |
| Durable failure and phase | API 200 `ERROR`, `REASONING -> ERROR`, one captured snapshot; exact events survive container recreation. |
| Minimum-document guard | One selected document cannot meet two; zero answer-generator calls. This does not measure document relevance or independence. |
| Safe inspection | Failed/incomplete exports reject with their distinct 409 codes; history/events/browser inspection appends zero events. |
| Explicit new operation | A separate fake `DONE` has a new ID; the old error's events and rejected export remain unchanged. |
| Mocked provider behavior | Malformed HTTP 200 makes one attempt; persistent 503 makes four total attempts; neither stores the synthetic raw-body marker or completes an answer. |

Keep original failure counts in any report; do not count a rerun as repairing a
failed test case. These synthetic checks establish plumbing, preservation, and
error behavior, **not** model quality, latency/cost, scientific accuracy, account
access, or live-provider compatibility. Attempt counts are measured only inside
the mock; live attempts and charges remain unknown without separate evidence.

Implementation anchors: [runner](../../src/agent/runner.py),
[state machine](../../src/agent/state_machine.py),
[executor/capture order](../../src/agent/executor.py),
[event log](../../src/storage/event_log.py),
[history projection](../../src/storage/run_history.py),
[query/events routes](../../src/api/routes.py),
[export validation](../../src/storage/evidence_export.py), and
[provider adapters](../../src/llm/providers.py).
