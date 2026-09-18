# Portable research evidence bundles

A completed query can now be downloaded as a **versioned JSON bundle** or a
**readable Markdown report**. Both use the same saved run: original question,
retrieval plan and operational rationale, answer, claims, citations, exact
final-context passages, provider/model identity when available, and ordered events.

Forgot the run ID? [Run history](RUN_HISTORY_GUIDE.md) adds `GET /runs` discovery
from persisted events, including after restart. Its export URLs are navigation
links; all evidence-export validation and legacy/failed-run errors remain intact.

This closes a practical gap between an answer and an auditable research artifact.
The old citation snippet contains at most 240 characters; it cannot recover the
full context once an index changes. The new snapshot is recorded **after reranking
and before generation**, not reconstructed later from today's corpus.

![Offline evidence export demonstration](../assets/evidence-export.gif)

**GIF provenance:** generated illustration from the actual deterministic,
synthetic demo transcript, not a screen recording. The demo uses no real paper,
API key, external service, or live model. Run IDs and event timestamps in the
downloaded files are real per-run values; the illustrated counts, digest, and
checks come from the demo.

## Run the complete offline demo

From the repository root, with Python 3.11 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m scripts.demo_evidence_export --output-dir evidence-demo
```

The script drives the real ingestion, query, and export routes through FastAPI's
`TestClient`. It explicitly disables all provider credentials for this demo,
including credentials configured in your environment or `.env`. It does not
start a server. Choose a **fresh output directory**: it refuses to overwrite
existing demo files.

The public `offline_settings(path)` helper uses validated defaults without
ambient environment/dotenv/secret-file sources. Both offline evidence/history
demos construct `api.application.create_app(settings)` explicitly, never import
the initialized deployment app, and do not open the ambient database.

The output directory contains:

| File | Inspectable result |
| --- | --- |
| `bundle.json` | Complete version-1.0 API response, including full Unicode evidence |
| `bundle.md` | The same saved run rendered as a literal Markdown report |
| `demo.sqlite3` | Durable event log, with this demo's source corpus deliberately removed |
| `transcript.txt` | Actual requests/statuses, counts, digest, and restart comparisons |

The demo's synthetic passage has **485 characters**, while its citation snippet
has **240**. It produces eight operational events and records `provider=fake`,
`model=None`. After deleting **only its newly created synthetic corpus** from
document and graph tables, it opens a new `AppContainer` against the same SQLite
file and requires byte-identical JSON **and** Markdown downloads.

`evidence-demo/` is ignored by Git. Other output directories are not automatically
ignored; do not commit private artifacts. Re-running the demo in the same
directory intentionally fails rather than replacing a prior research record.

Inspect the actual response shape and independently check its context digest:

```bash
python - <<'PY'
import hashlib
import json
from pathlib import Path
from agent.evidence import EvidenceBundle

path = Path("evidence-demo/bundle.json")
bundle = EvidenceBundle.model_validate_json(path.read_bytes())
print("Top-level fields:", ", ".join(json.loads(path.read_text(encoding="utf-8"))))
print("Version/status:", bundle.schema_version, bundle.status)
print("Original query:", bundle.query)
print("Provider/model:", bundle.generation.provider, bundle.generation.model_name)
print("First claim:", bundle.claim_evidence[0].model_dump())
for source in bundle.snapshot.sources:
    assert hashlib.sha256(source.chunk.text.encode("utf-8")).hexdigest() == source.text_sha256
    print("Evidence:", source.rank, source.chunk.chunk_id, len(source.chunk.text))
assert hashlib.sha256(
    bundle.snapshot.request.context.encode("utf-8")
).hexdigest() == bundle.snapshot.context_sha256
print("Exact context digest:", bundle.snapshot.context_sha256)
print(Path("evidence-demo/bundle.md").read_text(encoding="utf-8")[:700])
PY
```

The deterministic context digest for this demo is
`a52cdea6ab8804a1ea0fdf2c996d733a2fff89ee7af9ed88ee268d672cb2845e`.
This is a content fingerprint, **not a signature or proof of authenticity**.

To regenerate only this guide's GIF, using the actual transcript and the existing
Pillow development dependency:

```bash
python -m scripts.create_evidence_gif --transcript evidence-demo/transcript.txt
```

This writes `docs/assets/evidence-export.gif`; it does not regenerate other assets.

## Use the HTTP API

For a separate, local-only synthetic API session, start the server in one terminal.
Explicit empty credentials matter: merely selecting `fake` as the default does
not override the task router's preference for a configured reasoning provider.

```bash
OPENAI_API_KEY= ANTHROPIC_API_KEY= GEMINI_API_KEY= MOONSHOT_API_KEY= \
SCHOLAR_RAG_DEFAULT_MODEL=fake \
SCHOLAR_RAG_DATABASE_PATH=evidence-api.sqlite3 \
scholar-rag-api
```

In another terminal with the same virtual environment:

```bash
curl --fail-with-body --silent --show-error http://127.0.0.1:8000/ingest/text \
  --json '{"title":"Synthetic note","text":"GraphRAG connects entities in synthetic research evidence.","source":"synthetic:api"}'

run_id=$(curl --fail-with-body --silent --show-error http://127.0.0.1:8000/query \
  --json '{"query":"What does GraphRAG connect?"}' \
  | python -c 'import json,sys; result=json.load(sys.stdin)["result"]; assert result["state"]=="DONE", result; print(result["run_id"])')

mkdir -p evidence-demo/api
curl --fail-with-body --silent --show-error \
  "http://127.0.0.1:8000/runs/$run_id/export?format=json" \
  --output evidence-demo/api/bundle.json
curl --fail-with-body --silent --show-error \
  "http://127.0.0.1:8000/runs/$run_id/export?format=markdown" \
  --output evidence-demo/api/bundle.md
python -m json.tool evidence-demo/api/bundle.json
```

`GET /runs/{run_id}/export` defaults to JSON. Responses are attachments with
`application/json` or `text/markdown; charset=utf-8`, `Cache-Control: no-store`,
and `X-Content-Type-Options: nosniff`. The suggested filename is
`evidence-<16 hex characters>.json` or `.md`, derived from a hash of the run ID,
never from a source title, URL, or path. The curl commands above choose their own
fixed output filenames.

The complete typed JSON schema is published under `EvidenceBundle` in
`GET /openapi.json` and exposed in FastAPI's `/docs`. For real provider
configuration, see [the provider models guide](PROVIDER_MODELS_GUIDE.md).

## JSON contract and associations

| Field | Meaning |
| --- | --- |
| `schema_version` | String `"1.0"`; consumers should reject unsupported versions |
| `run_id`, `agent_id`, `status`, `completed_at` | Saved identity, terminal `"DONE"`, and saved UTC completion timestamp; no new export-time timestamp |
| `query` | Original submitted query, including whitespace; `plan.observation.original_query` is the analyzer's trimmed query |
| `configuration` | Allowlisted effective `max_source_docs`, `max_hops`, `retrieval_timeout_seconds`, and `reasoning_timeout_seconds`, frozen before asynchronous work |
| `plan` | Existing `QueryPlan`, including tasks, clamped hop limits, and deterministic planner rationale, not hidden model thinking |
| `answer` | Existing `AgentAnswer` unchanged, including warnings, `[UNGROUNDED]`, claims, citations, and the old short snippets |
| `snapshot.request` | Exact provider-independent `LLMRequest` submitted for generation: prompt, context, routing task type, and citation IDs |
| `snapshot.sources` | Post-rerank order; each entry has one-based `rank`, final `score`, `retriever`, retrieval `path`, full `chunk`, and `text_sha256` |
| `snapshot.sources[].chunk` | Original captured `chunk_id`, `document_id`, `title`, **full text**, source string, and available string metadata |
| `snapshot.context_sha256` | SHA-256 of the exact context's UTF-8 bytes, without normalization |
| `snapshot.context_format`, `snapshot.limits` | Versioned context assembly format and the explicitly allowed capture-limit fields |
| `generation` | Returned provider label, optional configured/requested `model_name`, task type, and proposed citation IDs per claim |
| `claim_evidence` | One-based claim number, proposed IDs, final grounding IDs, resolved evidence ranks, and explicitly missing IDs |
| `citation_evidence` | One-based citation number, chunk ID, and its evidence rank, or `null` if missing |
| `events` | Ordered saved event envelopes through the **first terminal event**; later notes cannot change an already completed artifact |
| `warnings` | Privacy and interpretation limits, missing-reference warnings, and explicit unknown-event payload omissions |

The saved context format is:

```text
[{chunk_id}] {title}: {full_chunk_text}
```

Entries are joined by one newline, in final rank order. The bundle includes the
complete assembled context as well as the structured passages, so consumers can
compare the digest and reconstruct the model input. The source text is the
**post-ingestion, post-rerank chunk**, not necessarily the original PDF's text or
the whole paper. Snapshot hashes cover context/text, not every artifact field.

Proposed citation IDs are retained before grounding drops unmatched IDs.
`missing_chunk_ids` reports references outside the frozen source set; those IDs
are never re-retrieved to fill a gap. A resolved ID means only that a source is
present. An unsupported claim can still reference a present passage. Final claim
and citation structures and warnings are preserved rather than rewritten to
make the evidence look stronger.

`model_name` is the configured/requested provider API identifier when the adapter
knows it, **not a verified resolved backend model snapshot**. Fake or unavailable
identity is `null`; an unavailable provider label is `"unknown"`. Export does not
infer identity from today's settings. No settings dump, API key, request headers,
environment, raw provider response, or hidden thinking is added.

The trace uses `payload_ref: "snapshot"` or `"generation"` for the two large/new
payloads instead of duplicating them inside `events`. Other recognized
operational payloads are parsed through their existing models. Unknown event
payloads are `null` with `payload_omitted: true` and a warning; unrecognized
diagnostic data is not assumed safe to share.

## Persistence, bounds, and compatibility

Capture reuses `SQLiteEventLog` in the configured database; there is no second
event system or mutable "last answer" cache. Each executor call has its own
detached snapshot and run-bound callbacks. The successful sequence is:

```text
IDLE -> PLANNING
decision_log
PLANNING -> RETRIEVING
RETRIEVING -> REASONING
evidence_snapshot       (durable before generation)
generation_record       (returned identity and proposed claim IDs)
REASONING -> ANSWERING
ANSWERING -> DONE
```

`POST /query` retains its response shape. `/runs/{run_id}/events` retains its
existing behavior with **two additive event types** and allowlisted
`configuration` metadata in the initial `PLANNING` payload; that endpoint now also
contains full saved source content. Consumers asserting the full sequence must
accept these events. Runtime source/hop caps and timeouts are copied at run start
and used throughout that run; later configuration changes do not rewrite them.

Legacy `Executor.answer(plan, retrieved)` overrides continue to run unchanged.
The runner checks callback signature compatibility **before** invoking an
executor, and never retries a generation after a `TypeError`. To support export,
custom executors can accept `on_context` and `on_generation` and record truthful
data. Runs that omit capture receive `409 snapshot_unavailable`; no snapshot is
invented for them.

Version one allows at most **50 source passages**, **262,144 UTF-8 context bytes**,
and **1,048,576 bytes for the snapshot's SQLite JSON payload**, including escaped
Unicode, metadata, and the request. These are capture bounds, not token counts
or guarantees about a provider's context window. The whole assembled snapshot is
checked; metadata cannot bypass the bound. No passage is silently clipped or
dropped. An oversized or invalid snapshot, or failed snapshot write, fails the
run before generation; narrow the source scope/input instead. Normal configured
retrieval limits still apply.

Empty retrieval is different from absent capture: a successful run with a
recorded empty context can be exported honestly with zero sources and its
ungrounded warnings. Deleting or updating corpus rows does not change saved
events. Reopening the same database preserves exportability without any LLM,
retrieval, or external network call during export.

Optional [document-scoped queries](DOCUMENT_SCOPE_GUIDE.md) record eligible IDs in
`plan.observation.document_ids` and the initial planning event. Markdown includes
a Document scope section. Export compares those records and checks captured
source/citation documents against the saved selection, without retrieving them
again. Older version-one records without scope remain readable as unscoped;
`RunConfiguration` and the bundle version are unchanged. Selection is not
authorization, and a bundle captures final-context passages, not the whole
selected corpus.

## Explicit error policy

| HTTP status | `detail.code` | Meaning |
| --- | --- | --- |
| `404` | `run_not_found` | No saved events for this ID |
| `422` | FastAPI validation response | `format` is not exactly `json` or `markdown` |
| `409` | `run_incomplete` | No terminal event, including an interrupted process |
| `409` | `run_failed` | First terminal event is `ERROR`, even if a snapshot was already saved |
| `409` | `snapshot_unavailable` | Legacy/completed run has no generation-time snapshot |
| `409` | `invalid_run_record` | Missing/inconsistent required records, invalid digests, or unsupported saved version |

Except for FastAPI's standard validation response, errors use:

```json
{"detail":{"code":"snapshot_unavailable","message":"This run predates evidence capture or has no saved snapshot; it cannot be reconstructed from the current corpus."}}
```

There are no success-shaped empty artifacts for unknown, legacy, failed, or
incomplete runs. There is no backfill from a mutable corpus. Run a new query if
you need a new artifact, and treat it as a **different run**, not recovered
historical evidence.

## Privacy and interpretation limits

This API has **no authentication or tenant isolation**. Use it only locally or
in a trusted environment; a run ID is not authorization. Protect the database,
event endpoint, exported files, backups, and any shared output directory.
Snapshots intentionally retain sensitive source text and metadata. Deleting
source documents does **not** erase their evidence snapshots. There is no new
retention/purge API; account for the event database and copies when removing data.

Review before sharing. Application credentials are excluded, but secrets that a
user puts **inside a query, source text, or metadata** are part of that content
and are not automatically redacted. The Markdown renderer puts dynamic data in
literal fenced blocks with sufficiently long delimiters: source HTML/Markdown,
links, code fences, and Unicode are not promoted into report markup or filenames.
It is a readable report, not an executable notebook.

Citation grounding checks **non-stopword token overlap**, not semantic
entailment, truth, study quality, or scientific consensus. The default dense
retriever uses `HashEmbeddingModel`; default reranking is lexical. The agent
uses the repository's explicit state machine, not LangGraph. An artifact makes
the saved evidence inspectable; it does not upgrade those algorithms.

Append-only application events are not signed or tamper-proof. A database owner
can edit rows and recompute hashes. Frozen context does not promise identical
future LLM output, stable provider routing/backend revisions, or a full replay
of provider HTTP/system prompts. The bundle records the final
**provider-independent context request**, not hidden model reasoning.

## Inspiration and the repository gap

Primary references checked on 2026-09-18 UTC (2026-09-17 Pacific):

- [PaperQA README](https://github.com/Future-House/paper-qa/blob/57e89f7223b0960d5ee5ea048c69e3c47e088572/README.md):
  evidence gathering/answering, saved prior answers, and question/answer/context
  objects inspired retaining the research result with its context.
- [PaperQA LFRQA per-question JSON logs](https://github.com/Future-House/paper-qa/blob/57e89f7223b0960d5ee5ea048c69e3c47e088572/docs/tutorials/running_on_lfrqa.md#L252-L263):
  a useful precedent for inspectable per-question artifacts.
- [Haystack AnswerBuilder](https://github.com/deepset-ai/haystack/blob/52df9672a7808e1091995f33189d1ff8e91b6998/docs-website/docs/pipeline-components/builders/answerbuilder.mdx#L25-L33):
  associating replies, questions, documents, and parsed reference indices informed
  explicit claim/citation-to-evidence associations, not an entailment claim.
- [Haystack tracing privacy](https://github.com/deepset-ai/haystack/blob/52df9672a7808e1091995f33189d1ff8e91b6998/docs-website/docs/development/tracing.mdx#L8-L32):
  content logging is distinct from ordinary operational tracing. Here, evidence
  capture intentionally stores content, so privacy and retention must be explicit.

The gap addressed is **this repository's** lack of one durable downloadable run
artifact combining an answer, its exact final-context passages, associations,
and an operational trace. Existing BibTeX export, paper chat memory, and lexical
claim/evidence helpers do not capture this generation-time context. These
inspirations are not claims that the other projects categorically lack exports,
and no competitor source or documentation text is copied.
