# Retrieve only human-included papers

Use `screened_collection` on `POST /retrieve` or `POST /query` to hand a
revision-checked human reading selection directly to the existing retrieval
pipeline. You no longer need to copy `included_document_ids` from a queue page.
Ordinary `collection_id` still selects **all** members, regardless of screening.
This is explicit corpus selection, not automatic screening or access control.

[Measured offline workflow GIF](../assets/screened-collection.gif) |
[Exact text transcript](../assets/screened-collection.txt)

The opt-in GIF is an original Pillow illustration of executed synthetic API
output, not browser footage, a live-model demonstration, or research findings.
The complete reproducible example below checks the measurements before saving
the illustration.

## Workflow gap and inspiration

[ASReview](https://github.com/ASReview/asreview) documents human label management
and exporting reviewed records. [PaperQA](https://github.com/Future-House/paper-qa)
supports scientific question answering over a chosen local paper collection.
This feature joins those workflow ideas: explicitly select current human
inclusions, inspect the actual evidence, then optionally generate and preserve
the selected document IDs. It is original glue over the existing screening
store and scoped retrievers, not imported code, a connector, active learning,
or a claim of either project's capabilities or evaluation results.

The stack remains Python 3.11+, FastAPI, Pydantic, HTTPX, SQLite, and the
hand-written Observe -> Decide -> Act state machine. Selection itself has no
model calls; `/retrieve` has neither fake/live generation nor event writes.
`/query` uses the existing routed generator and durable evidence capture.

Official catalogs were rechecked **2026-10-26 America/Los_Angeles**:
[OpenAI](https://developers.openai.com/api/docs/guides/latest-model) lists
GPT-6 Astra, GPT-6.1 Sol, and GPT-6 Luna;
[Anthropic](https://platform.claude.com/docs/en/models/overview) lists Fable 5.1
and Opus/Sonnet/Haiku 5.5;
[Google](https://ai.google.dev/gemini-api/docs/models/gemini-4-argon) identifies
Gemini 4 Argon as its flagship; and
[Kimi](https://platform.kimi.ai/docs/get-your-api-key.md) recommends K3.1 Pro,
with a Highspeed variant. This feature changes none of the selected
`gpt-6-astra`, `claude-sonnet-5-5`, `gemini-3.8-flash`, and `kimi-k3.1-pro`
defaults. See the [provider guide](PROVIDER_MODELS_GUIDE.md) for endpoint and
payload limits. Catalog checks do not establish account access or live inference
compatibility; the demonstration uses only the explicit offline fake.

## Execute the complete offline demonstration

From a checkout with Python 3.11+ and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev
OUTPUT="$(mktemp -d)/screened-demo"
uv run python scripts/demo_screened_collection.py \
  --output "$OUTPUT" --gif "$OUTPUT/screened-collection.gif"
printf 'Artifacts: %s\n' "$OUTPUT"
```

The script explicitly ignores environment, dotenv, and secret-file settings.
It creates three synthetic papers and human include/exclude/unsure decisions
in a temporary database, restarts the app, compares ordinary and screened-only
previews, and saves a fake answer with exact selected IDs. It then changes the
human decision, checks that empty and stale selections fail, and verifies that
the earlier evidence export is unchanged after another restart.

External HTTP and live generation are forbidden throughout; fake generation
and event appends are also forbidden during previews and empty-selection checks.
Expected measurements are three evidence documents for ordinary scope versus
one for screened scope, zero preview events, HTTP 409 for both empty selections
and outdated collection revisions, and unchanged saved evidence. The script
checks these values rather than drawing an invented result.

Inspect `screened-preview.json`, `screened-evidence.json`, `screened-demo.json`,
and `transcript.txt` in the output directory. The temporary SQLite corpus is
removed on exit. Existing named output files or GIFs are never overwritten.
The JSON includes generated IDs/timestamps, so independent runs need not be
byte-identical. Its measurements and the rendered text are reproducible.

## API: select, preview, then decide whether to generate

Start an isolated local API using the [Quickstart](../../QUICKSTART.md), ingest
your permitted papers, and create a named
[collection](PAPER_COLLECTIONS_GUIDE.md). Review it through the existing
[screening API](PAPER_SCREENING_GUIDE.md). Read the collection's actual ID and
current revision; do not invent or increment them locally.

For a collection returned as `COLLECTION_ID`, keep its revision in `REVISION`.
The following POSIX-shell example prompts for those two values, safely creates
JSON using Python, and previews without generation:

```bash
export BASE_URL=http://127.0.0.1:8000
printf 'Collection ID from GET /collections: '
read -r COLLECTION_ID
export COLLECTION_ID
COLLECTION="$(curl --fail-with-body --silent --show-error \
  "$BASE_URL/collections/$COLLECTION_ID")"
REVISION="$(printf '%s' "$COLLECTION" | uv run python -c \
  'import json,sys; print(json.load(sys.stdin)["revision"])')"
export REVISION
REQUEST="$(uv run python -c '
import json, os
print(json.dumps({
    "query": "Compare the methods in these reviewed papers",
    "screened_collection": {
        "collection_id": os.environ["COLLECTION_ID"],
        "collection_revision": int(os.environ["REVISION"]),
    },
    "max_chunks_per_document": 2,
    "min_evidence_documents": 1,
}))
')"
curl --fail-with-body --silent --show-error "$BASE_URL/retrieve" \
  -H 'Content-Type: application/json' -d "$REQUEST" | uv run python -m json.tool
```

Inspect `plan.observation.document_ids`, `sources`, and `evidence_assessment`.
The selected IDs may include a paper with no matching passages; selection does
not guarantee retrieval coverage. The existing `min_evidence_documents` policy
checks actual final evidence, not the number of human include decisions.

Only after reviewing the preview, use the same request for generation:

```bash
curl --fail-with-body --silent --show-error "$BASE_URL/query" \
  -H 'Content-Type: application/json' -d "$REQUEST" | uv run python -m json.tool
```

Check `result.state` and `result.error`, not just HTTP success. A valid selection
can still produce an `ERROR` run because of provider, retrieval, or evidence
requirements. For a completed run, use its returned `run_id` with the
[evidence export API](EVIDENCE_EXPORT_GUIDE.md). The plan and initial run event
retain the exact resolved `document_ids`; the snapshot retains actual context.
The new collection selector, decision revisions, and human reasons are **not**
new saved-bundle fields. Use screening/history exports separately when needed.

Each request resolves current decisions independently. A human edit between
preview and query can change the selection without changing the collection
revision. To deliberately reuse the preview's exact IDs, pass that nonempty
`plan.observation.document_ids` as `document_ids` instead; this still does not
freeze current corpus text. Do not claim the two requests are a transaction.

## Python

Resolve once from the same database as the runner, then explicitly pass the
returned tuple:

```python
from pathlib import Path

from agent.retrieval_preview import RetrievalPreview
from api.dependencies import AppContainer
from scripts.demo_evidence_export import offline_settings


async def preview_included(
    database_path: Path, collection_id: str, collection_revision: int
) -> RetrievalPreview:
    container = AppContainer(offline_settings(database_path))
    selected = container.paper_screening.select_included(
        collection_id, collection_revision=collection_revision
    )
    return await container.runner.preview("Compare the methods", document_ids=selected)
```

`SQLitePaperScreening.select_included` returns a detached, nonempty tuple in
saved collection order. It validates all members and current review records
in one SQLite read transaction, not just a default queue page or only included
reviews. Even malformed excluded records invalidate the selection. The method
does not serialize reasons or catalog labels, so queue/export response-byte
caps do not turn a complete ID selection into a partial one.

## Contract, errors, and limits

| Condition | Result before planning, retrieval, generation, or event writes |
| --- | --- |
| Omit all scope fields | Existing full-corpus behavior |
| Supply ordinary `collection_id` | All current members; screening is not consulted |
| Supply `screened_collection` | Only current `include` decisions for the exact collection revision |
| Combine it with `collection_id` or `document_ids` | HTTP 422 |
| Explicit null, invalid ID, missing/extra nested field, noninteger revision | HTTP 422 |
| Unknown collection | HTTP 404, `collection_not_found` |
| Outdated collection revision | HTTP 409, `collection_revision_conflict` |
| No current inclusions | HTTP 409, `screened_collection_empty`; never empty/unscoped fallback |
| Missing member, broken collection, malformed review | Existing explicit collection/screening conflict |
| Storage failure | HTTP 503 with a stable public code, no private SQLite diagnostic |

Nested `collection_revision` is a strict integer from 1 through
9,223,372,036,854,775,807; strings, bools, floats, NaN, and infinities are invalid.
Collection IDs keep their existing `col_` plus 32 lowercase hexadecimal format.
No new settings, tables, dependencies, or model budgets are added. The existing
100-member selection and bounded review-record checks remain in force.

`exclude`, `unsure`, `unscreened`, and `stale` are not eligible. Editing a
collection increments its revision and makes older labels stale; re-screen
explicitly instead of retrying with an automatically substituted revision.
Decision revisions and collection revisions are different optimistic-concurrency
counters. New decisions on the same collection revision are visible to the next
selection request.

Worksheets and retrieval comparisons have separate request schemas and reject
`screened_collection` rather than silently ignoring it. Native browser selection
and screening writes retain their existing behavior. The API/Python handoff
does not add another browser form or change normal collection links.

## Safety and portfolio use

The service is unauthenticated and intended for local/trusted use. Human
inclusions are opinions, not authenticated reviewer identity, scientific truth,
medical advice, relevance scores, or proof of complete evidence. Staleness is
based on collection revision, **not** changes to corpus contents.

Resolved IDs are copied before asynchronous work. A later screening edit or
collection deletion cannot widen an in-flight selection, and excluded graph
edges cannot bridge into evidence. The read transaction does not remain open
during retrieval/generation; document text and indexes are not version-frozen
by this feature. Existing saved evidence is a separate frozen capture.
No-store/nosniff headers and safe validation errors are not authorization.
Selection IDs, queries, reasons, and saved excerpts may be private.

For a portfolio demonstration, run the synthetic script, show the ordinary
versus screened source IDs and 409 failures, and inspect the saved plan/context
after the labels change. Link the measured transcript beside the opt-in GIF.
Explain the difference between a human opinion, a selected document, a retrieved
passage, and a fake answer. Do not present the illustration as research quality,
active-learning performance, or a recording of a UI that this feature does not add.
