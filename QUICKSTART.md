# Quickstart

Run the local workflow without model credentials or downloaded papers.
Requirements: Python 3.11+, [uv](https://docs.astral.sh/uv/), and `curl` for the
API examples. Commands assume a POSIX shell and the repository root.

## 1. Install

```bash
git clone https://github.com/Francis1998/scholar-rag-agent.git
cd scholar-rag-agent
uv sync --extra dev
```

If you already cloned the repository, start with `uv sync --extra dev`.

## 2. Run the offline demo

```bash
uv run python scripts/demo_local.py
```

Expect an ingested chunk ID, a run ID, `State: DONE`, planner tasks, and a cited
placeholder answer. The script explicitly uses `FakeLLMAdapter` and a temporary
database, so it does not call a model even if you have provider keys configured.
That database is removed on exit; this demo does not preload the API.

The fake adapter echoes the question with source IDs. It exercises the plumbing,
not scientific summarization or factual validation.

## 3. Start an isolated offline API

```bash
export SCHOLAR_RAG_DATABASE_PATH="$(mktemp -d)/corpus.sqlite3"
printf 'Demo database: %s\n' "$SCHOLAR_RAG_DATABASE_PATH"
OPENAI_API_KEY= ANTHROPIC_API_KEY= GEMINI_API_KEY= MOONSHOT_API_KEY= \
  uv run uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

The empty key values override matching `.env` entries for this command and keep
inference offline. Keep the printed database path for restart; do not create a
new temporary directory if you want to read the same runs later. This is a local
development server, not an authenticated deployment.

Open `http://127.0.0.1:8000/docs` for FastAPI's interactive API documentation.
There is no PDF-upload or paper-chat UI. Stop the server with Ctrl-C.

## 4. Ingest text and ask a question

Leave the server running. In a second terminal at the repository root:

```bash
curl --fail-with-body --silent --show-error http://127.0.0.1:8000/health

curl --fail-with-body --silent --show-error http://127.0.0.1:8000/ingest/text \
  -H 'Content-Type: application/json' \
  -d '{
    "title":"Synthetic GraphRAG note",
    "text":"Synthetic workshop note, not a publication. GraphRAG connects co-mentioned entities across passages. These links help retrieve related text but do not establish that a claim is true.",
    "source":"synthetic:quickstart"
  }'

curl --fail-with-body --silent --show-error http://127.0.0.1:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"What does GraphRAG connect?"}' \
  | uv run python -m json.tool
```

Health returns `{"status":"ok"}`; ingestion returns `document_id` and `chunk_ids`.
The query returns a `result` containing `run_id`, `state`, `plan`, `answer`, and
`error`. Inspect `result.state` as well as the HTTP status: runtime failures can
return `"ERROR"` inside a successful HTTP response. Read `answer.claims`,
`answer.citations`, `answer.ungrounded`, and `answer.warnings`; a grounding flag is
only a lexical check, not proof of support.

Both `/query` and `/retrieve` require a nonblank string. Empty or whitespace-only
questions, including Unicode whitespace, return HTTP 422 before planning,
retrieval, model calls, or agent-event writes. Validation preserves nonblank input;
the existing analyzer still trims outer whitespace in observations, plans, and
generation prompts.

To search only selected papers, pass their returned IDs in `document_ids` on
`/query`. Omit the field for the whole corpus; an empty list or explicit `null`
is rejected rather than widened. The [document scope guide](docs/guides/DOCUMENT_SCOPE_GUIDE.md)
has complete offline API/Python examples, validation rules, and a reproducible
selected-versus-unscoped demonstration.

## 5. Recover document IDs and select papers

```bash
curl --fail-with-body --silent --show-error \
  'http://127.0.0.1:8000/documents?limit=20' | uv run python -m json.tool
```

The catalog returns IDs, bounded titles/sources, and stored chunk counts without
running the agent or returning document bodies. Follow `next_cursor` and use
optional `source` (exact) or `title` (literal substring) filters to choose papers.
Pass their `document_id` values to `/query` `document_ids`. Reuse the same
database on restart; the [document catalog guide](docs/guides/DOCUMENT_CATALOG_GUIDE.md)
includes a complete offline example, privacy limits, and a measured animation.

### Preview before generating

For a saved collection, use
`GET /collections/{collection_id}/screening?collection_revision=1` to inspect
human labels and `PUT /collections/{collection_id}/screening/{document_id}` to
record a revision-checked decision. Use the actual collection revision. Pass
only the returned nonempty `included_document_ids` as `document_ids` to preview;
ordinary `collection_id` scope still includes every member. The complete
[paper-screening guide and offline GIF](docs/guides/PAPER_SCREENING_GUIDE.md)
cover setup, reasons, restart, stale decisions, and explicit selection.

```bash
curl --fail-with-body --silent --show-error http://127.0.0.1:8000/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"query":"What does GraphRAG connect?"}' | uv run python -m json.tool
```

Add `document_ids` from the catalog to inspect only selected papers. Preview
returns the actual planned, post-rerank chunks and context digest with no live
or fake LLM call, answer, or agent-event writes. Unlike `/query`, runtime
failures are HTTP errors rather than `ERROR` run bodies. See the complete
[retrieval preview guide](docs/guides/RETRIEVAL_PREVIEW_GUIDE.md) for scope,
limits, Python usage, privacy, and a measured offline demo.

To inspect the same questions against each selected paper, use
`POST /research/worksheet` with explicit document IDs or a saved collection.
The [worksheet guide](docs/guides/RESEARCH_WORKSHEET_GUIDE.md) provides a complete
offline example and JSON/Markdown downloads without generation or new run events.

## 6. Recover run IDs after restart

```bash
curl --fail-with-body --silent --show-error \
  'http://127.0.0.1:8000/runs?limit=20' | uv run python -m json.tool
```

Reuse the printed database path when restarting the server. The catalog returns
saved query previews, last recorded states, and existing events/export links;
follow `next_cursor` to older runs. A nonterminal record does not mean a process
is still executing, and a legacy `DONE` record may not have exportable evidence.
The [run-history guide](docs/guides/RUN_HISTORY_GUIDE.md) includes a complete
offline restart demonstration, filtering, pagination, and Python contracts.

### Export the saved answer's cited references

```bash
RUN_ID='replace-with-your-completed-run-id'
BIBLIOGRAPHY_DIR="$(mktemp -d)"
curl --fail-with-body --silent --show-error \
  "http://127.0.0.1:8000/runs/$RUN_ID/bibliography?format=json" \
  > "$BIBLIOGRAPHY_DIR/bibliography.json"
uv run --no-sync python -m json.tool "$BIBLIOGRAPHY_DIR/bibliography.json"
curl --fail-with-body --silent --show-error \
  "http://127.0.0.1:8000/runs/$RUN_ID/bibliography" \
  > "$BIBLIOGRAPHY_DIR/bibliography.bib"
```

BibTeX is the default; review the JSON warnings and exact cited-chunk mapping
before importing the `.bib` file into a reference manager. Only saved final
citations qualify, not every retrieved source. Missing author/year/DOI fields
remain absent; this text-ingestion example does not capture scholarly metadata.
No current-corpus read, generation, metadata lookup, or event write occurs.
See the [complete saved bibliography guide](docs/guides/SAVED_BIBLIOGRAPHY_GUIDE.md)
for the standalone Python reader, fixed byte limits, errors, and measured GIF.

### Keep a note on an exact frozen quote

For a completed exportable run, POST `/runs/{run_id}/annotations` with a client
UUID, exact document/chunk IDs, the frozen source's `text_sha256` as
`source_text_sha256`, Python Unicode `start`/`end` offsets, exact `quote`, and
a nonblank human `note`. Read these from the full saved export, not current
corpus text or a truncated answer citation. Identical retries return 200 after
the first 201; changed payloads under the same UUID return 409.

The [complete offline API/Python guide](docs/guides/EVIDENCE_ANNOTATIONS_GUIDE.md)
includes working selector construction, limits, errors, privacy, and a GIF.
For a self-contained measured demonstration with no running server:

```bash
ANNOTATION_DEMO="$(mktemp -d)"
uv run --no-sync python -m scripts.demo_evidence_annotations \
  --output-dir "$ANNOTATION_DEMO/results"
uv run --no-sync python -m scripts.create_evidence_annotations_gif \
  --transcript "$ANNOTATION_DEMO/results/transcript.txt" \
  --output "$ANNOTATION_DEMO/evidence-annotations.gif"
```

The output directory must be new. The demo guards against ambient keys and
network access, checks persistence after corpus removal/restart, and removes
its synthetic database. Human notes are opinions, not scientific verification.

### Check saved evidence against the current corpus

For a completed run with a valid saved snapshot:

```bash
RUN_ID='replace-with-your-completed-run-id'
curl --fail-with-body --silent --show-error \
  "http://127.0.0.1:8000/runs/$RUN_ID/corpus-drift" | uv run python -m json.tool
```

The bounded read-only report preserves frozen rank order and distinguishes
unchanged, changed, and missing chunks. It does not generate, retrieve, or
rewrite saved evidence. Matching chunks are not proof of whole-paper equality
or scientific validity. See the [corpus-drift guide](docs/guides/CORPUS_DRIFT_GUIDE.md)
for exact fields, explicit errors, Python usage, and a fully offline demonstration.

## Next steps

Follow the [research workflow](docs/guides/RESEARCH_WORKFLOW_GUIDE.md) for a clean
three-note corpus, comparison/hypothesis queries, human review, and saved exports.
[API examples](docs/EXAMPLES.md) show event and export requests;
[the evidence export guide](docs/guides/EVIDENCE_EXPORT_GUIDE.md) explains snapshot
and error behavior. For optional live inference, consult the dated
[provider model guide](docs/guides/PROVIDER_MODELS_GUIDE.md) and review
[data-handling limitations](SAFETY.md) first.
