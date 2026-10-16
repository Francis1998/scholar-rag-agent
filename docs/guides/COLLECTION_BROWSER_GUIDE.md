# Browse saved collections and human screening

Open **Browse saved collections** from `/explore`. This native, script-free
workflow discovers saved selections, inspects current membership and human
screening, and hands off to the existing readers and downloads. It does not
create collections, record decisions, select evidence automatically, or deduplicate.

![Executed synthetic collection-browser output illustration](../assets/collection-browser.gif)

The four frames illustrate **actual executed synthetic demo output**, not browser
screenshots, a screen recording, real scientific results, or a performance benchmark.
The demo follows returned HTML links and native controls, downloads actual bytes,
checks restart behavior, and measures forbidden calls. Reproduce it below.

## End-user workflow

Start the isolated loopback API in the [Quickstart](../../QUICKSTART.md), ingest
permitted papers, and save a selection through the existing
[collection API/Python interface](PAPER_COLLECTIONS_GUIDE.md). Human decisions
remain explicit operations in the [screening interface](PAPER_SCREENING_GUIDE.md).
No key, new setting, migration, browser package, or front-end framework is needed.

1. From `/explore`, choose **Browse saved collections**. Collection cards show
   exact IDs, names, revisions and **membership counts, not screened includes**.
   Page through collection-ID order. Catalog filters are retained for returning
   to the catalog; they do not filter the collection list or its membership.
2. Open a card's **Inspect membership and human screening** link. It binds the
   collection revision shown on that card. Inspect `unscreened`, `include`,
   `exclude`, `unsure` and `stale` counts, then filter or page members in saved
   membership order. Counts and current include IDs always cover all members.
   Each decision shows its reason, decision revision, reviewed collection
   revision, and timestamp. Stale includes remain stale, not current includes.
   Choose **Inspect human decision history** for a bounded newest-first view
   of retained decisions, with exact reasons/timestamps and an explicit legacy
   boundary. The [history guide](SCREENING_HISTORY_GUIDE.md) covers the separate
   query-selected API/Python reader; there is no editing control or backfill.
3. Choose **Read this exact paper / choose source context**. The existing passage
   reader selects an exact document; its source links select exact stored chunks.
   Back links preserve collection status, member/list pagination and catalog
   filters. Slashes, dot segments, Unicode, punctuation and opaque IDs are
   query-encoded, never guessed as paths. Empty papers have no invented source link.
4. **Search all current collection members** and **Open collection worksheet**
   prefill the existing tools. These use **all current members**, not the visible
   status or include IDs. Search resolves current membership when submitted.
   Worksheet GET opens controls only; POST separately performs local retrieval
   and reranking, without generation. Its 10-member bound remains enforced,
   including for larger collections: no automatic subset is chosen.
5. Download complete screening JSON/CSV or duplicate-candidate JSON/Markdown.
   Every link uses the exact collection ID and displayed revision. Downloads
   include all members regardless of the visible filter. Duplicate inspection
   uses its existing default threshold `0.92`; no browser tuning or deduplication
   action is added.

The detail page decodes one existing bounded screening JSON export, so its
membership, labels, statuses and counts share **one read transaction**.
**Separate requests are not one snapshot.** A rename or membership replacement
changes the collection revision and makes old links fail. Decision edits and
corpus changes can occur **at the same collection revision**, changing a later
page, search, worksheet or download. There is no frozen multi-request view,
authenticated approval or scientific-support claim. The separate append-only
history reader retains new decisions, not old paper contents or legacy backfill.

If only current includes should be previewed, inspect and deliberately pass the
nonempty `included_document_ids` as `document_ids` to `/retrieve`. Do not replace
an empty list with `collection_id` or an omitted scope.

## Browser and existing API contracts

| GET route | Selection and output |
| --- | --- |
| `/explore/collections` | Native HTML list; `limit` 1-100 (default 20), exclusive `cursor` collection ID. |
| `/explore/collection` | HTML detail; required `collection_id` and positive integer `collection_revision`; `limit` 1-100, exclusive member-ID `cursor`, optional `status` (blank means all). Cursor must be a member of that revision. |
| `/collections` | Existing JSON summaries, with the same collection pagination. |
| `/collections/{id}/screening?collection_revision=N` | Existing paged JSON screening queue, counts and current include IDs. |
| `/collections/{id}/screening/export?collection_revision=N&format=json` | Complete JSON; use `format=csv` for spreadsheet-safe CSV. |
| `/collections/{id}/duplicates?collection_revision=N&format=json` | Existing complete duplicate report; use `format=markdown` for literal Markdown. |

Browser return-only fields are `catalog_limit`, `source`, `title` and
`catalog_cursor`; detail additionally carries `collections_limit` and
`collections_cursor`. Follow returned URLs rather than inventing cursors.
The validated `collection_return` JSON field on downstream readers contains
structured local navigation, **not an arbitrary return URL, authorization token,
or snapshot identifier**. No client storage is used.

For an existing service, substitute an actual ID and revision from `/collections`:

```bash
COLLECTION_ID=col_REPLACE_WITH_32_LOWERCASE_HEX_DIGITS
REVISION=1
curl --fail-with-body --silent --show-error --get \
  http://127.0.0.1:8000/explore/collection \
  --data-urlencode "collection_id=$COLLECTION_ID" \
  --data-urlencode "collection_revision=$REVISION" \
  --data-urlencode "status=stale" --data-urlencode "limit=10"
curl --fail-with-body --silent --show-error --get \
  "http://127.0.0.1:8000/collections/$COLLECTION_ID/screening/export" \
  --data-urlencode "collection_revision=$REVISION" \
  --data-urlencode "format=csv" --output /tmp/screening.csv
```

CSV uses the existing `apostrophe-prefix-v1` encoding: every text cell has one
added leading apostrophe and every field is quoted. Remove exactly that prefix
when decoding text programmatically; do not treat it as part of the raw ID.
See [screening exports](SCREENING_EXPORT_GUIDE.md) and
[duplicate inspection](COLLECTION_DUPLICATES_GUIDE.md) for full schemas and limits.
Mounted applications retain `root_path` on forms, return links, errors and downloads.

## Python: use the same services

Use the actual existing database path and collection ID. Service constructors
may initialize tables, like normal application startup; the following list,
get, export and inspection calls perform no storage writes.

```python
from pathlib import Path

from storage.collection_duplicates import SQLiteCollectionDuplicates
from storage.paper_collections import SQLitePaperCollections
from storage.paper_screening import ScreeningExport, SQLitePaperScreening

database = Path("/absolute/path/to/existing/corpus.sqlite3")
collection_id = "col_REPLACE_WITH_32_LOWERCASE_HEX_DIGITS"
collections = SQLitePaperCollections(database)
screening = SQLitePaperScreening(database)
page = collections.list_collections(limit=20)
collection = collections.get(collection_id)
download = screening.export_results(
    collection.collection_id, collection_revision=collection.revision, format="json"
)
snapshot = ScreeningExport.model_validate_json(download.content)
print(snapshot.counts, snapshot.included_document_ids)
for member in snapshot.items:
    print(member.document_id, member.status, member.review)

report = SQLiteCollectionDuplicates(database).inspect(
    collection.collection_id, collection_revision=collection.revision
)
with Path("/tmp/screening.json").open("xb") as output:
    output.write(download.content)
with Path("/tmp/duplicates.md").open("x", encoding="utf-8") as output:
    output.write(report.to_markdown())
```

No second store or browser-specific screening logic is introduced. The Python
snapshot matches the browser's source data, not a promise that two sequential
calls see identical data when another writer is active.

## Bounds, failures and privacy

Each collection has at most 100 members. Titles show at most 300 characters and
sources 512, with explicit truncation labels; reasons retain the existing
1000-character bound. The **complete** screening JSON snapshot must fit 524,288
UTF-8 bytes even when only one member is visible. Escaped HTML must fit 1 MiB.
Failures never return a partial membership or silently widened selection.

Collection list/detail query strings are capped at 16,384 bytes. Downstream
paper/context/search/worksheet URLs carrying validated `collection_return` state
have a separate **65,536-byte** cap so maximum Unicode return filters and exact
IDs survive percent encoding. Return JSON is bounded to 16,384 characters.
Ordinary worksheet GET URLs retain 16,384 bytes; its form body remains capped
at 65,536 bytes. Hidden navigation uses JSON escaping to preserve CR/LF without
inflating printable Unicode into nested ASCII escapes. Control-bearing filters
that native text inputs cannot preserve use exact pagination/status links
instead of lossy forms. The existing worksheet rejects raw NUL form/query fields;
JSON-encoded document IDs and exact paper/source links retain their existing rules.

| Status | Meaning and recovery |
| --- | --- |
| 404 | Exact collection was deleted or does not exist. Return to the list; no substitute is chosen. |
| 409 | Revision changed, a member is missing, or stored metadata/text is invalid. Refresh the list or inspect trusted storage; browsing cannot repair it. |
| 413 | Complete export or escaped HTML exceeds its fixed bound. A smaller visible page helps HTML, not an oversized complete export. |
| 422 | Missing/invalid revision, unsupported status, nonmember/empty cursor, duplicate/unknown fields, invalid UTF-8 or oversized query. Correct the request; no filter is discarded. |
| 405 | Collection browsing accepts GET only; no editing actions exist. |
| 503 | Existing storage is unavailable, not an empty list or unscreened result. |

HTML successes and failures reuse `no-store`, `nosniff`, `no-referrer`, frame
denial and the hash-only style CSP with `script-src 'none'`. Dynamic names, reasons
and labels are escaped literal text, never executable markup or external links.
Diagnostics log stable codes, not private records or SQLite details. Existing
download headers and serialization remain unchanged.

There is **no built-in authentication or tenant isolation**. Use loopback or an
independent access-control boundary. IDs, names, reasons and filters can be
sensitive; URLs can remain in browser history or server logs. No-store cannot
prevent saving, screenshots or redistribution. Review downloads before sharing.
Collection browsing runs no model (including fake generation), network request,
retrieval, reranking, ingestion, mutation or agent-event write. Normal startup
still initializes stores and loads retrieval indexes.

## Reproduce the executed demo

From the repository with the existing development dependencies installed, choose
unused output paths:

```bash
uv run python -m scripts.demo_collection_browser \
  --output-dir /tmp/scholar-collection-browser-demo
uv run python -m scripts.create_collection_browser_gif \
  --transcript /tmp/scholar-collection-browser-demo/transcript.txt \
  --output /tmp/scholar-collection-browser.gif
uv run pytest tests/test_collection_explorer.py tests/test_collection_browser_demo.py
```

The demo creates and removes its own temporary SQLite corpus. Setup explicitly
saves synthetic membership/decisions before guards begin. Outputs include actual
HTML, screening JSON/CSV, duplicate JSON/Markdown, a transcript and `checks.json`.
Measured checks include all five human statuses, exact Unicode/slash source and
return links, revision-conflict HTTP 409, Python/HTTP export parity, byte-identical
restart HTML, unchanged database bytes, zero events and zero forbidden calls.
The GIF uses the existing four-panel Pillow renderer; tests decode all four
distinct 1120-by-540 frames and their durations. Neither command overwrites an
existing output directory/GIF.

## Workflow inspiration and portfolio scope

The **2026-10-14 America/Los_Angeles** comparison used
[Cinnamon/kotaemon](https://github.com/Cinnamon/kotaemon) (25,805 stars at review),
whose documented UI organizes collections, and
[Future-House/paper-qa](https://github.com/Future-House/paper-qa) (9,373 stars),
whose scientific RAG workflow emphasizes evidence. These are workflow
inspirations, not copied code, compatible project formats, performance
equivalence, or evidence that star counts measure quality.

This feature demonstrates an integrated discovery-to-source/export workflow with
revision-aware errors, bounded serialization, literal-safe HTML and executable
read-only/restart contracts. The actual stack is FastAPI, Pydantic, HTTPX, SQLite
and the existing custom agent state machine, not LangGraph or a new front-end
stack. It does not implement automatic screening, new cue extraction,
connectors, active learning, automatic deduplication, authentication or LLM calls.
The [provider catalog refresh](PROVIDER_MODELS_GUIDE.md) is documentation-only;
selected defaults and runtime payload contracts do not change.

Related: [architecture](../../ARCHITECTURE.md), [safety](../../SAFETY.md),
[configuration](../../CONFIGURATION.md), [source context](SOURCE_CONTEXT_GUIDE.md),
[passage search](BROWSER_PASSAGE_SEARCH_GUIDE.md), and
[research worksheets](RESEARCH_WORKSHEET_GUIDE.md).
