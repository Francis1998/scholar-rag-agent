# Read human opinions beside frozen evidence

From `http://127.0.0.1:8000/explore`, choose **Browse saved runs**, then
**Human-review notebook**. Read the completed answer, all three recorded review
decisions, exact comments and exact evidence annotations. Follow a note into
its full frozen source with only the recorded occurrence highlighted.
Nothing is generated, retrieved, edited or reconstructed from today's corpus.

[![Actual synthetic human-review notebook browser frame](../assets/human-review-notebook.png)](../assets/human-review-notebook.gif)

[Play the four-frame browser GIF](../assets/human-review-notebook.gif).
These are **actual browser screenshots** of the executed synthetic fixture
below, not a continuous recording, scientific result or performance metric.
The still preview makes animation opt-in.

## Use the notebook

1. Start the [Quickstart](../../QUICKSTART.md) API with the same SQLite database
   that holds the saved run. Both a saved-run entry and its evidence reader
   expose **Human-review notebook**. Incomplete, failed, legacy or corrupt runs
   fail explicitly; `DONE` alone does not guarantee an exportable capture.
2. Inspect the exact run/agent IDs, completed time, saved answer and recorded
   ungrounded flag. **Human opinions are separate from model grounding and
   scientific truth.** An `accepted` opinion does not verify the answer.
3. Read **Append-only answer reviews**: `accepted`, `needs_revision` and
   `rejected`, original comment, UTC timestamp, sequence, UUID and schema.
   Cited chunk/document IDs link to native `#source-N` frozen evidence anchors,
   not current-corpus passages or external source URLs.
4. Read **Append-only evidence annotations**: original quote and note,
   document/chunk IDs, full-source SHA-256, timestamp, sequence, UUID and
   half-open Unicode code-point offsets. There is no inferred reviewer identity.
5. **Follow exact quote in frozen source** opens the recorded UUID in that
   exact run. The service validates the annotation and source together; the
   browser highlights the recorded slice, not every repeated phrase or a
   normalized/fuzzy substitute. The complete captured chunk remains visible,
   including escaped markup, Unicode, whitespace and carriage returns.
6. Each history has its own **Next page**, restart link and 1-100 page-size
   control. Changing one history preserves the other's cursor/limit. **Back to
   this notebook page** from an annotation restores both selections.
   **Back to saved runs** preserves its state, cursor, limit and catalog return
   context. Opening the complete evidence reader retains the saved-run return
   state; its notebook entry starts a fresh opinion traversal.
7. Empty histories mean no recorded opinions, not endorsement. Conflicting
   decisions coexist; the UI does not infer consensus, an overall verdict or
   total-history counts. To add an opinion, use the existing
   [review API/Python service](ANSWER_REVIEWS_GUIDE.md) or
   [exact-annotation API/Python service](EVIDENCE_ANNOTATIONS_GUIDE.md).
   There are no browser editing controls or new write endpoints.

**Snapshot boundary:** the answer, reviews and annotation history are
independently read, not one shared snapshot. Cursors select strictly smaller
immutable sequences, newest first; restart a history to see new appends.
Unseen pages are not audited for corruption. A source link uses a stable
run-scoped UUID rather than its moving page position. That individual
annotation and its source are read in one read-only SQLite transaction.
For a **complete shared snapshot** rather than visible pages, use the new
**Download complete notebook (JSON/Markdown)** links. The
[portable notebook guide](PORTABLE_REVIEW_NOTEBOOK_GUIDE.md) covers whole-history
limits, offline use and the executed demo; existing page semantics stay unchanged.

Keyboard navigation, visible focus, skip links, Find and Print are native.
Source metadata remains literal text. A highlighted quote is provenance,
not proof that it supports a claim or represents a complete paper.

## Routes, selection and bounds

| Route or parameter | Contract |
| --- | --- |
| `GET /explore/run/notebook` | Native HTML answer, reviews and annotations for one completed saved run |
| `GET /explore/run/annotation` | Native HTML for one exact annotation and its full frozen source |
| `run_id` | Required exact raw ID, 1-256 Unicode code points; no trimming, path interpretation or normalization |
| `annotation_id` | Required UUID only on the annotation route; looked up within the exact run |
| `review_limit`, `annotation_limit` | Independent ASCII decimal integers, 1-100, default 20 each |
| `review_cursor`, `annotation_cursor` | Independent positive sequences through 9223372036854775807; omitted initially, not page numbers |
| `limit`, `cursor`, `state` | Existing saved-run browser return selection, **not** opinion filters |
| `catalog_limit`, `source`, `title`, `catalog_cursor` | Existing validated current-catalog return selection, **not** a source lookup |
| Query string | At most 16,384 raw bytes; unknown/duplicate fields, invalid UTF-8, empty cursors and invalid bounds fail before reading |
| Complete HTML | At most 1,048,576 escaped UTF-8 bytes, including navigation, notes, sources and shell; no silent clipping |
| Review content | Existing 4,000-code-point comments and up to 100 cited chunk IDs; requested rows plus lookahead validated |
| Annotation content | Existing 1,000-code-point notes/quotes, selected-record byte checks and 262,144-byte compact JSON history cap |
| Frozen event reads | Existing 100 events, 1 MiB per payload and 8 MiB total UTF-8 event text; count/byte checks include later events |

Follow generated links: IDs are shown as JSON strings for lossless copying.
Decode the JSON representation when a raw ID is needed. Slashes, dot segments,
combining marks and punctuation stay in encoded query values. Only blank
`state` selects all states; only blank catalog `source`/`title` omit those filters.
Native forms are omitted with an explanation when CR/LF/NUL in return values
cannot round-trip through inputs; exact encoded pagination/return links remain.
All links and forms retain `root_path` and nested mount prefixes.

The two browser pages reuse the existing explorer escaping, CSP, URL, paging
and sanitized logged error contracts. No JavaScript, client framework, external
resources, redirects, new dependencies, tables or settings are
introduced by these pages. Existing offline evidence downloads remain self-contained and do
not acquire the live notebook or human opinion records.

| Outcome | Meaning and recovery |
| --- | --- |
| 200, empty history | No records, or no records older than that history's cursor; restart explicitly |
| 404 | Exact run or run-scoped annotation absent; no replacement guessed |
| 405 | GET only, with `Allow: GET`; no write was executed |
| 409 | Incomplete/failed/legacy capture, corrupt review/annotation/lookahead, invalid anchor, bounded-event overflow or unrepresentable HTML text; no partial page |
| 413 | Annotation JSON or full escaped HTML exceeds its cap; reduce the relevant page size, not the recorded content |
| 422 | Invalid selection, pagination, return parameters or query encoding; no silent widening |
| 503 | Saved storage unavailable, not an empty notebook; check database availability/permissions |

Response/error bodies do not echo raw exceptions or invalid submitted text.
Valid return parameters remain in their links. Stored NUL/invalid Unicode
cannot be silently rendered as replacement text. SHA-256 and exact-slice
checks detect inconsistency, not a database owner's coherent rewrite.

**Compatibility:** `AppContainer.notebook_reviews` is a lightweight
`AnswerReviewService` using `saved_run_evidence` and the same already-initialized
`SQLiteAnswerReviews` store as `answer_reviews`. No store is constructed per GET.
Existing POST/GET review APIs retain their original first-terminal-event
exporter behavior. A valid old review API request can therefore succeed while
the notebook rejects an over-limit run with many ignored post-DONE events.
No prior opinions or saved events are changed.

## Existing API/Python access

These HTML routes complement, rather than replace,
`GET /runs/{run_id}/reviews`, `GET /runs/{run_id}/annotations` and the existing
submission APIs. The notebook exposes no automatic grading. Complete human-opinion
downloads use the separate [portable notebook endpoint](PORTABLE_REVIEW_NOTEBOOK_GUIDE.md).
For query-addressed legacy run IDs that contain slashes, use the browser or
Python rather than rewriting IDs to fit path-based APIs.

```bash
BASE_URL=http://127.0.0.1:8000
RUN_ID='copy-an-exact-saved-run-id'
curl --fail-with-body --silent --show-error --get "$BASE_URL/explore/run/notebook" \
  --data-urlencode "run_id=$RUN_ID" \
  --data-urlencode 'review_limit=1' --data-urlencode 'annotation_limit=1'
```

`SQLiteEvidenceAnnotations.get_annotation(run_id, UUID(...))` now returns the
validated `(SavedEvidenceAnnotation, EvidenceSource)` pair from one snapshot.
It uses the existing bounded reader and anchor validator, never a current
corpus lookup. The store's existing constructor initializes its additive
schema; application startup still initializes stores and loads retrieval
indexes. The no-work/no-write guarantee applies to reads, not startup or setup.

## Reproduce the offline workflow and GIF

Use Python 3.11+, uv and the existing development extra. The following output
directory must be new; no ambient database or credentials are used.

```bash
uv sync --locked --extra dev
DEMO_DIR="$(mktemp -d)/human-review-notebook"
uv run --no-sync python -m scripts.demo_review_notebook --output-dir "$DEMO_DIR"
cat "$DEMO_DIR/checks.json"
uv run --no-sync python -m scripts.demo_review_notebook --serve "$DEMO_DIR" --port 8770
```

Setup ingests one original synthetic note and makes one real fake-adapter query,
then records three reviews and two annotations through existing HTTP services.
The measured browser reads follow independent pages and the second repeated
quote. Guards reject generation, retrieval, opinion/corpus/event writes,
unbounded event reads and external HTTP during browsing. Separate socket
guards cover the offline fixture execution.

Only the newly generated fixture's current corpus is changed, then deleted.
After restart, notebook/source/history-page HTML is byte-identical; saved
events and opinions remain. The generated database is retained for inspection,
not committed. The loopback server rejects a mismatched fixture marker or
changed database hash, disables access logging and has the same browser-read
guards. Stop it with Ctrl-C; it is not a scheduled process.

The output contains the synthetic database, actual HTML responses,
`reviews.json`, `annotations.json`, measured `checks.json`, `transcript.txt`
and four exact URLs in `pages.json`. These are demo artifacts, not new product
export formats.

For screenshots, use a browser at **1280 x 960 CSS pixels**, device scale 1,
100% zoom. Visit the four `pages.json` URLs on `http://127.0.0.1:8770`.
Save unmodified viewport PNGs, in order, as `saved-runs.png`,
`answer-reviews.png`, `evidence-annotations.png` and `exact-quote.png` in a new
frames directory. The fragments use actual native anchors, not a mocked DOM.
Assemble them using the existing Pillow renderer:

```bash
FRAME_DIR=/absolute/path/to/new-browser-frames
uv run --no-sync python -m scripts.create_review_notebook_gif \
  --frames-dir "$FRAME_DIR" --output "$DEMO_DIR/human-review-notebook.gif"
```

The renderer requires four distinct full-size PNGs and refuses replacement.
It preserves real screenshot pixels; it does not draw a pretend browser.
Reproduction yields new run UUIDs/timestamps rather than backdated records.

Read the same generated records directly without a server:

```python
import json
from pathlib import Path
from uuid import UUID

from api.dependencies import AppContainer
from scripts.demo_evidence_export import offline_settings
from scripts.demo_review_notebook import notebook_guards

directory = Path("/absolute/path/to/generated-notebook-demo")
checks = json.loads((directory / "checks.json").read_text())
container = AppContainer(offline_settings(directory / "notebook.sqlite3"))
with notebook_guards(container):
    page = container.notebook_reviews.list_reviews(checks["run_id"], limit=2)
    for review in page.reviews:
        print(review.sequence, review.decision, repr(review.comment))
    annotation, source = container.evidence_annotations.get_annotation(
        checks["run_id"], UUID(checks["selected_annotation_id"])
    )
    assert source.chunk.text[annotation.start : annotation.end] == annotation.quote
    print(annotation.created_at.isoformat(), repr(annotation.quote), repr(annotation.note))
```

## Privacy, portfolio and inspiration

This remains a local/trusted **unauthenticated** service, not peer review,
moderation or a multi-tenant notebook. IDs/return filters can remain in browser
history and server logs. Comments, notes, answers and source text may be
sensitive. No-store, nosniff, no-referrer and anti-framing headers do not prevent
saving or screenshots. Deleting the current corpus does not erase retained
evidence/opinions/backups. Share only reviewed synthetic artifacts.

For a portfolio, demonstrate discovery, all three decisions, independent
pagination, repeated-quote disambiguation and unchanged frozen output after
corpus deletion/restart. Show the measured guards and explain the independent
snapshot boundary. Describe this as an integrated Python/FastAPI/Pydantic/
HTTPX/SQLite workflow over the existing custom state machine, **not LangGraph**.
Do not claim scientific accuracy, identity verification or performance gains.

**Design inspiration, checked 2026-10-20:** [PaperQA](https://github.com/Future-House/paper-qa)
(9,425 stars at that check) emphasizes grounded cited scientific answers and
inspectable references; [RAGFlow](https://github.com/infiniflow/ragflow)
(92,537 stars) emphasizes traceable citations, visual chunk inspection and
human intervention. The local gap was discoverability of already-recorded
human feedback beside immutable evidence. This is original, narrower
implementation work, not copied code, a new integration or feature parity.

**Dated model context, 2026-10-20 (America/Los_Angeles):**
[OpenAI's catalog](https://developers.openai.com/api/docs/models.md) recommends
`gpt-6-astra` and lists `gpt-6.1-sol`/`gpt-6-luna`;
[Claude](https://platform.claude.com/docs/en/models/overview) lists Fable 5.1,
Opus 5.5, Sonnet 5.5 (released September 28) and Haiku 5.5;
[Google](https://ai.google.dev/gemini-api/docs/latest-model) names
`gemini-4-argon` latest; [Kimi](https://platform.kimi.ai/docs/guide/kimi-k3-quickstart)
names `kimi-k3` flagship. Existing Astra/Sonnet 5.5/Gemini 3.8 Flash/Kimi K3
defaults are unchanged. This notebook needs none and makes no model calls.
Catalog checks do not test inference, account entitlement or new aliases
through the existing adapters.
