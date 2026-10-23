# Find a saved answer and read its frozen evidence

Open `http://127.0.0.1:8000/explore`, choose **Browse saved runs**, then
**Inspect saved evidence**. Recover an exact run ID after restart, inspect
its recorded question and state, follow citations to captured passages, and
download the existing JSON, Markdown or offline HTML artifact. No model,
retrieval, current-corpus read or event write is needed to open saved evidence.

Choose **Human-review notebook** on a run entry or its evidence reader to read
recorded decisions/comments and exact annotations. The
[notebook guide](HUMAN_REVIEW_NOTEBOOK_GUIDE.md) covers its independent histories
and native exact-quote highlighting. Opinions stay separate from grounding;
the existing offline downloads do not acquire live notebook data.

[![Static preview of the actual synthetic saved-run browser](../assets/saved-run-browser.png)](../assets/saved-run-browser.gif)

[Play the four-frame browser GIF](../assets/saved-run-browser.gif).
The preview is intentionally still; animation is opt-in. The GIF contains
**actual browser screenshots** of the reproducible synthetic output below,
not a continuous recording, live inference, benchmark or scientific result.
It shows history, its next page, a recorded answer, and a frozen source.

## Browser workflow

1. Start the [Quickstart](../../QUICKSTART.md) API on loopback, using the same
   SQLite database that contains your runs. In `/explore`, **Browse saved runs**
   preserves the current catalog's page size, source/title filters and cursor
   as return navigation. Those filters do **not** select runs.
2. Choose an **Exact recorded state**, or **All recorded states**, and
   **Runs per page** from 1 to 100. **Apply recorded-state filter** starts at
   the newest creation key, without the previous cursor. States are saved
   facts, not a live-job monitor: `REASONING` can be an interrupted trace.
3. Inspect the exact run/agent IDs, first event ID, first/last event timestamps,
   event count and original query preview. A preview has at most 300 Unicode
   code points; a separate notice says when text was omitted. A missing query
   and an explicitly saved empty query have different messages.
4. **Next page** keeps the state, size and catalog return selection.
   **Start again with the same selection** removes only the history cursor.
   **Show all states** explicitly removes state and cursor. New runs appear
   when you restart traversal, not in the middle of an older keyset page.
5. **Inspect saved evidence** validates the exact run before showing a complete
   capture. History does not eagerly export each row or promise that `DONE`
   means exportable. Failed, incomplete, legacy and inconsistent traces receive
   honest error pages, with a return link to the original history selection.
6. In **Saved run evidence**, follow proposed references, accepted-by-recorded-
   grounding references, or final citations to `#source-N`. These links stay
   inside the **frozen capture**, never the current corpus's context reader.
   Inspect full captured text, rank, score, path, metadata, digests and trace.
   An unresolved reference is not turned into a guessed source.
7. **Back to saved runs** restores the same history page. **Back to current
   corpus** crosses an explicitly labeled boundary and restores its catalog
   context. Download JSON, Markdown or **offline HTML** for a validated run.
   The HTML download is still the original self-contained reader, with no
   application navigation or network dependency.

Use Tab/Shift-Tab, Enter, native select controls, and the first **Skip to main
content** link. Focus is visible; layouts wrap on narrow screens. Browser Find,
details and Print remain native. There is no JavaScript, client framework,
third-party resource, automatic refresh, analytics or browser storage.
Normal application startup still initializes stores and loads retrieval indexes;
the no-work guarantee applies to these page reads, not server construction.

### Recorded history versus a frozen capture

History projects the **latest recorded transition**, while the exporter freezes
the trace at the **first terminal event**. A valid completed capture can remain
readable after a later diagnostic transition changes its history state.
Conversely, a legacy `DONE` record without a snapshot is visible but not exportable.
The browser preserves both facts rather than inferring a repaired state.

Timestamps retain the history reader's timezone-aware instant, offset and
microseconds. They are the times of the first/last events **by event ID**, not
the minimum/maximum wall-clock times. The display does not round them into
relative labels such as "a few minutes ago."

## HTTP contract

Both new routes are native GET pages returning `text/html; charset=utf-8`.
The existing `GET /runs` JSON contract, evidence schema and download URLs are
unchanged. OpenAPI also documents the new routes.

| Route / field | Contract |
| --- | --- |
| `GET /explore/runs` | A bounded page from `SQLiteRunHistory.list_runs`, not an evidence/body scan per row |
| `limit` | Default 20; ASCII decimal integer from 1 to 100 |
| `state` | Omit or send blank for all; otherwise exactly `IDLE`, `PLANNING`, `RETRIEVING`, `REASONING`, `ANSWERING`, `DONE`, or `ERROR` |
| `cursor` | Omit initially; otherwise positive signed-64-bit first-event ID, 1 through 9223372036854775807 |
| `GET /explore/run?run_id=...` | Required exact raw run ID, 1-256 Unicode code points; no trimming or normalization |
| `limit`, `state`, `cursor` on `/explore/run` | Validated history-return selection only; evidence always belongs to the exact `run_id` |
| `catalog_limit` | Independent original catalog page size, 1-100, default 20 |
| `source`, `title`, `catalog_cursor` | Original catalog return state; source up to 512, title up to 300, and exact cursor ID up to 128 code points with no outer whitespace |
| Unknown or duplicate fields | 422, even for return-only fields; no last-value-wins or silently discarded filter |
| Query bytes | At most 16,384 raw ASGI query-string bytes; malformed UTF-8 fails before store reads |

Only blank `state` explicitly selects all states; only blank `source` and
`title` omit those catalog filters. Empty cursors, `state=done`, `state=null`,
decimal/exponent/boolean integer spellings and partial encoded Unicode fail.
GET has strings, not JSON null. The word `null` can still be an exact run ID.
Editing a filter starts a new traversal. Manually combining a cursor with a
different valid state selects that new state below the same numeric boundary;
integer cursors are not signed or filter-bound tokens.

With a trusted local server:

```bash
BASE_URL=http://127.0.0.1:8000
curl --fail-with-body --silent --show-error --get "$BASE_URL/runs" \
  --data-urlencode 'state=DONE' --data-urlencode 'limit=2'
curl --fail-with-body --silent --show-error --get "$BASE_URL/explore/runs" \
  --data-urlencode 'state=DONE' --data-urlencode 'limit=2'

RUN_ID='copy-an-exact-run-id-from-history'
curl --fail-with-body --silent --show-error --get "$BASE_URL/explore/run" \
  --data-urlencode "run_id=$RUN_ID" --data-urlencode 'state=DONE'
```

Use the returned first-event-ID cursor to continue; it is not a page number.
The selection uses `first_event_id < cursor`, descending. New runs and appended
events do not move an existing run's creation key. State changes can alter
membership between pages: this is **not** a cross-request point-in-time snapshot.
For complete storage semantics, see [run history](RUN_HISTORY_GUIDE.md).

### Exact IDs and mount prefixes

History IDs are JSON string representations, including quotes and escapes.
Decode that representation before pasting a raw ID into an API/Python call.
Native evidence links already encode the correct raw value. Unicode, combining
marks, bidirectional controls, slashes, dot segments, punctuation and meaningful
whitespace are not rewritten.

Query-addressed `/explore/run` can read otherwise valid legacy IDs containing
`/`, `.` or `..`. These do not receive misleading path-based download links:
the existing download routes cannot address them reliably. Use the bounded
Python reader below for their artifacts. NUL-bearing IDs can be discovered as
JSON strings, but the literal evidence HTML renderer rejects NUL explicitly.
Control-bearing query previews also fail rather than display replacement text.

NUL/CR/LF in catalog return state cannot round-trip through native text inputs.
The filter form is omitted with an explanation, while encoded page/evidence/
return links preserve the values exactly. Reset those filters in the catalog
before editing. Invalid requests have an explicitly labeled reset route; they
are never relabeled as a successful broader selection.

All generated application paths retain ASGI `root_path`, including nested
mount prefixes, forms, pagination, successful reader/download navigation and
error recovery. No user-controlled redirect URL is accepted. Optional Python
`BrowserNavigation` links must be application-local absolute paths, not
external, scheme-relative, control-bearing or backslash-obfuscated URLs.

## Read the same saved evidence in Python

The new `SQLiteEvidenceReader` opens an **existing** event database with
`mode=ro`, starts one read transaction, and supplies existing `BoundedRunEvents`
to `EvidenceExporter`. It adds no table, migration, backfill or reconstruction.
It is also available as `AppContainer.saved_run_evidence`, but standalone use
does not initialize an application, provider or corpus index.

```python
from pathlib import Path

from agent.evidence_html import render_html
from agent.models import AgentState
from storage.evidence_export import EvidenceExportError, SQLiteEvidenceReader
from storage.run_history import SQLiteRunHistory

database = Path("/absolute/path/existing-runs.sqlite3")
destination = Path("/absolute/path/new-frozen-evidence.html")
page = SQLiteRunHistory(database).list_runs(limit=10, state=AgentState.DONE)
for summary in page.runs:
    print(repr(summary.run_id), summary.recorded_state, summary.started_at.isoformat())
    print(repr(summary.query_summary), summary.query_truncated)
print("Exclusive continuation:", page.next_cursor)

try:
    bundle = SQLiteEvidenceReader(database).export("copy-an-exact-run-id")
except EvidenceExportError as error:
    raise SystemExit(f"{error.status_code} {error.code}: {error}") from error
with destination.open("xb") as output:
    output.write(render_html(bundle).encode("utf-8"))
```

This example reads one page and one explicitly selected run. It does not
silently scan until some export succeeds. `EvidenceExportError.code` and
`.status_code` describe failure; callers must not turn it into empty evidence.
Writing the requested HTML file is separate from database reads.

`render_html(bundle)` preserves the original offline output. The native route
opts into `render_html(bundle, navigation=BrowserNavigation(...))`, adding only
escaped, validated local history/catalog/download links inside the same total
byte cap. It does not duplicate the evidence renderer or replace the stored
answer. JSON/Markdown downloads still use the existing export implementations.

## Fixed bounds and honest failure

| Boundary | Limit / behavior |
| --- | --- |
| History results | At most 100 summaries plus one validated lookahead row; preview 300 code points, exact run/agent IDs 256 each |
| History HTML | Complete escaped UTF-8 page at most 1,048,576 bytes; no row is silently dropped |
| Saved event read | At most 100 events; each payload at most 1,048,576 UTF-8 bytes; total event text at most 8,388,608 UTF-8 bytes |
| Read accounting | Counts timestamp, agent/run ID, event type and payload text; includes post-terminal events before the exporter freezes its trace |
| Evidence HTML | Complete escaped UTF-8 output, navigation included, at most 4,194,304 bytes; exact limit accepted, excess rejected |
| Capture format | Existing independent 50-source, 262,144-context-byte and 1,048,576-snapshot-byte bounds remain |
| Connections | Read-only SQLite transactions; five-second connection busy timeout, not a history-scan latency guarantee |

History grouping and transition checks still scale with the event log.
Pagination limits returned/materialized summaries, not all SQLite work, and
there is no new history scan deadline or persisted index/table. UTF-16 databases
use the existing storage-byte preflight and exact decoded UTF-8 accounting.
These limits are fixed, not new environment settings or adjustable provider budgets.

| Outcome | Meaning / recovery |
| --- | --- |
| 200, empty history | Empty database, no matching state, or exhausted cursor; each has an explicit message |
| 404 | No events for that exact run ID; no replacement is guessed |
| 405 | GET-only page; protected HTML response with `Allow: GET` |
| 409, incomplete / failed | No terminal successful capture; recorded state is not liveness |
| 409, snapshot unavailable | Legacy `DONE` without captured evidence; never filled from today's corpus |
| 409, invalid record | Encountered summary/lookahead or evidence is inconsistent, invalid or unsupported; no partial page |
| 409, read limit | Existing bounded event reader rejected event count/payload/total size; no partial trace |
| 409, unrepresentable text | NUL or invalid Unicode cannot be preserved faithfully as HTML; inspect JSON/Python |
| 413 | Complete escaped history/evidence HTML exceeds its cap; reduce history page size or use appropriate existing export/Python interfaces |
| 422 | Invalid bounds, state, ID, UTF-8, unknown/duplicate query field or catalog-return state; no read ran |
| 503 | Saved database unavailable, not an empty success; check local path/permissions |

New browser success and handled error responses, including validation and
unsupported methods, receive `Cache-Control: no-store`,
`X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer` and anti-framing
headers. Errors show fixed diagnostics, not saved payloads, exceptions,
provider responses or pasted invalid values. Validated back-link parameters
remain in links. External proxy/HTTP-parser failures are outside these routes.

History reuses the explorer's hash-only stylesheet CSP, with native same-origin
form submission. Evidence reuses the offline renderer's hash-only stylesheet
CSP and denies forms; both deny resource loading by default. No `unsafe-inline`,
JavaScript, remote image/font, source URL activation or HTML/Markdown execution
is added. Escaping, exact identities and bounds are enforced independently of CSP.

### Privacy and scope

This is a local/trusted, **unauthenticated** service, not a multi-user review
platform. Page URLs can expose IDs and return filters to browser history and
server logs. Run previews, source text, answers, metadata and downloads may be
sensitive; no-store cannot prevent saving or screenshots. Review permissions
and artifacts before sharing. Deleting the current corpus does **not** erase
retained event snapshots; there is no retention or erasure feature here.

Saved history now links to the
[native comparison workbench](SAVED_RUN_COMPARISON_WORKBENCH_GUIDE.md):
choose an explicit baseline/candidate, inspect recorded changes side by side,
swap direction and follow precise frozen sources without rerunning.
There is no chat, write action, replay/resume, model selection,
auto-refresh, event repair, signed evidence or scientific validation.
Grounding records token overlap, not entailment. Sources can be incomplete,
biased, overlapping or wrong even when a citation link resolves.

## Reproduce the offline demo and real-browser GIF

From the repository root, with the existing development extra installed:

```bash
uv sync --locked --extra dev
DEMO_DIR="$(mktemp -d)/saved-run-browser"
uv run --no-sync python -m scripts.demo_saved_run_browser --output-dir "$DEMO_DIR"
cat "$DEMO_DIR/checks.json"
uv run --no-sync python -m scripts.demo_saved_run_browser --serve "$DEMO_DIR" --port 8768
```

The script uses original synthetic notes, two actual fake-adapter queries, and
explicitly inserted legacy/incomplete fixtures. It ignores environment,
dotenv, secret-file settings and ambient provider credentials. It measures
native form submission, state filtering, both continuations, preview
truncation, incomplete/legacy errors, exact source associations, API/Python
parity, return navigation and all three downloads.

After those reads it changes then deletes **only its newly created synthetic
corpus**, restarts and proves frozen HTML/downloads unchanged. Guarded page
reads leave database bytes and all 19 events unchanged, with zero generation,
retrieval, reranking, writes, external HTTP or socket connection calls. Fixture
setup does write and performs local fake generation; browsing does not.

| Artifact in the new output directory | Contents |
| --- | --- |
| `history.sqlite3` | Synthetic event database retained for browser inspection; current corpus is already empty |
| `catalog.html`, `history-first.html`, `history-next.html`, `filtered.html` | Actual native responses, including the original catalog return state |
| `legacy.html`, `incomplete.html` | Actual protected 409 responses |
| `frozen.html`, `restarted.html` | Byte-identical inline evidence before/after current-corpus deletion |
| `evidence.json`, `evidence.md`, `evidence.html` | Existing complete downloads; the last file is genuinely self-contained |
| `history.json`, `checks.json`, `transcript.txt` | Actual discovery data and measurements, not benchmark claims |
| `pages.json` | Four exact native URLs for browser frame reproduction, including generated run IDs |

Both demo and GIF tools refuse replacement of existing destinations/symlinks.
The server binds only `127.0.0.1`, disables access logs, and accepts only an
explicitly marked synthetic demo directory. Use Ctrl-C when finished; it is
not installed as a scheduled job. Do not commit demo databases or private
artifacts.

For the GIF, use a browser at **1280 by 960 CSS pixels**, device scale 1 and
100% zoom. Visit the four paths from `pages.json` on
`http://127.0.0.1:8768`, in their listed order. Capture the viewport as PNG,
without overlays or mocked DOM content, into a new frames directory:
`history-first.png`, `history-next.png`, `frozen-answer.png`, `frozen-source.png`.
The last two use real native fragment targets. Then, in another terminal:

```bash
FRAME_DIR=/absolute/path/to/new-browser-frames
DEMO_DIR=/absolute/path/to/the-generated-demo
uv run --no-sync python -m scripts.create_saved_run_browser_gif \
  --frames-dir "$FRAME_DIR" --output "$DEMO_DIR/saved-run-browser.gif"
```

The existing Pillow renderer checks dimensions, PNG decoding and four distinct
frames, then creates a 3-second-per-frame GIF. It never draws a substitute UI.
The checked-in still preview is the first actual frame. Browser/font versions,
generated run IDs and real event times can differ between reproductions;
the assertions and exact frozen-record behavior, not identical new run IDs,
are the reproducibility contract.

## Peer-informed design, stack and model check

The **2026-10-11 America/Los_Angeles** primary-source review considered
[kotaemon](https://github.com/Cinnamon/kotaemon), whose README describes an
end-user RAG UI, organized file collections and in-browser citation previews,
and [PaperQA](https://github.com/Future-House/paper-qa), whose README describes
scientific-document question answering and local full-text search.
Their useful workflow lesson is discoverability followed by source inspection.
This implementation closes that gap for **historical captures**, rather than
adding another generation interface. No external code, screenshots, paper text
or assets were copied, and no integration, parity or citation-quality claim is made.

The local stack remains Python 3.11+, FastAPI/Pydantic, HTTPX and SQLite, with
a custom planner/state machine, deterministic hash embeddings, BM25/RRF and an
offline fake default when keys are absent. The checked lock includes FastAPI
0.138.0, Pydantic 2.13.4, HTTPX 0.28.1 and Pillow 12.3.0. No dependency,
framework, retrieval or provider migration is part of this feature.

Official model pages were rechecked the same day:
[OpenAI Astra](https://developers.openai.com/api/docs/models/gpt-6-astra),
[OpenAI 6.1 Sol](https://developers.openai.com/api/docs/models/gpt-6.1-sol),
[Anthropic](https://platform.claude.com/docs/en/models/overview),
[Gemini Flash](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash),
and [Kimi](https://platform.kimi.ai/docs/overview.md).
Selected defaults remain `gpt-6-astra`, `claude-sonnet-5-5`, `gemini-3.8-flash`
and `kimi-k3`. The newer lower-cost `gpt-6.1-sol` is an alternative, with
Chat Completions limited to no-tool use and Responses required for tools;
it does not support `none`/`minimal` reasoning effort. This is a catalog-only
check, **not live account, inference, performance or budget verification**.
Endpoints, effort, budgets and separately dated migration contracts are unchanged;
see the [provider guide](PROVIDER_MODELS_GUIDE.md). Saved-run browsing uses none
of these provider settings.

## Portfolio walkthrough

Show the original catalog-to-history link, select `DONE`, then deliberately
open the legacy fixture and explain its 409. Continue to a real completed
capture, show the saved answer, follow a citation into its literal frozen
passage, and return to the exact history page. Show the empty current corpus
alongside the still-readable capture and byte-identical downloaded HTML.

Explain the engineering boundaries: strict query validation; stable exclusive
creation cursors; latest-state versus first-terminal semantics; bounded,
read-only snapshot validation; escaped semantic HTML with hash-only CSP;
root-path-safe navigation; and measured no-work/no-write invariants. Use the
GIF and checks to demonstrate an implemented workflow, not scientific accuracy,
provider quality, production readiness or a faster retrieval benchmark.
