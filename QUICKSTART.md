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

To search only selected papers, pass their returned IDs in `document_ids` on
`/query`. Omit the field for the whole corpus; an empty list or explicit `null`
is rejected rather than widened. The [document scope guide](docs/guides/DOCUMENT_SCOPE_GUIDE.md)
has complete offline API/Python examples, validation rules, and a reproducible
selected-versus-unscoped demonstration.

## 5. Recover run IDs after restart

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

## Next steps

Follow the [research workflow](docs/guides/RESEARCH_WORKFLOW_GUIDE.md) for a clean
three-note corpus, comparison/hypothesis queries, human review, and saved exports.
[API examples](docs/EXAMPLES.md) show event and export requests;
[the evidence export guide](docs/guides/EVIDENCE_EXPORT_GUIDE.md) explains snapshot
and error behavior. For optional live inference, consult the dated
[provider model guide](docs/guides/PROVIDER_MODELS_GUIDE.md) and review
[data-handling limitations](SAFETY.md) first.
