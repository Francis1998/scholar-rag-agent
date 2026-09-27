# Minimum evidence documents before generation

Opt in with **`min_evidence_documents`** on `POST /query`, `POST /retrieve`,
`AgentRunner.run`, or `AgentRunner.preview`. A query whose **final captured
context** contains fewer distinct actual `document_id` values than requested
records `ERROR` without calling the answer generator. A preview retains the
same inspectable passages and returns a typed count assessment, even when the
requirement fails.

This is a document-count admission rule, **not scientific answerability**:
passing does not establish relevance, source independence, completeness, quality,
or factual support. Omission keeps the previous behavior, including generation
from empty context.

![Measured synthetic minimum-evidence demonstration](../assets/minimum-evidence-documents.gif)

**Provenance:** this original illustration is rendered from the
[executed offline transcript](../assets/minimum-evidence-documents.txt), not a
screen recording or a model's research findings. Three passages from one
synthetic document fail a minimum of two. The failed run retains all three
passages and makes zero generation calls. A saved two-paper selection with
one passage per paper passes and generates with the fake adapter.

## Reproduce the measured demonstration

From the repository root, with Python 3.11+ and uv:

```bash
uv sync --extra dev
DEMO_DIR="$(mktemp -d)/minimum-evidence"
uv run python -m scripts.demo_minimum_evidence_documents --output-dir "$DEMO_DIR"
uv run python -m scripts.create_minimum_evidence_documents_gif \
  --transcript "$DEMO_DIR/transcript.txt" \
  --output "$DEMO_DIR/minimum-evidence-documents.gif"
uv run python -m json.tool "$DEMO_DIR/checks.json"
printf 'Review synthetic artifacts in %s\n' "$DEMO_DIR"
```

The demo executes real API routes, planning, hybrid/graph retrieval, lexical
reranking, quotas, capture, SQLite events, exports, and comparisons. It reuses
the existing explicit six-chunk synthetic corpus and Pillow renderer. It
ignores ambient keys, `.env`, secret-file settings, and database settings using
`offline_settings`; external HTTP is also denied. Setting only a default
provider to `fake` would not safely isolate ambient credentials.

All previews make zero generation calls and zero agent-event writes. The one
insufficient query makes zero generation calls; two sufficient queries each
call only the fake adapter once. Restart reads make no retrieval, generation,
or event writes. The temporary database is removed on exit; named output files
are never overwritten. Use a fresh directory/filename.

| Artifact | What it contains |
| --- | --- |
| `single-preview.json` | Three passages from one document; required 2, observed 1, failed |
| `unknown-preview.json` | Unknown selected ID retained in scope; required 1, observed 0 |
| `sufficient-preview.json` | Two selected papers after cap 1; required 2, observed 2, passed |
| `failed-run.json`, `failed-events.json` | Actual `ERROR` result, exact snapshot, and count diagnostic |
| `bundle.json`, `bundle.md` | Completed sufficient query with recorded policy and exact evidence |
| `comparison.json` | Minimum 1 versus 2 with identical context; only the policy changes |
| `checks.json`, `transcript.txt` | Measured counts, call/event checks, restart parity, and four GIF panels |

Run IDs and timestamps vary. The measured transcript and generated frames are
reproducible for these fixtures and locked dependencies; this is a plumbing
demonstration, not a retrieval-quality or latency benchmark.

## Request and execution contract

| Input | Behavior |
| --- | --- |
| Omitted | No minimum; old runner/executor signatures and unconfigured output remain supported |
| Strict integer `1` through `50` | Require that many distinct document IDs in final captured context |
| HTTP `null`, bool, float (including `1.0`), string, zero, negative, or above 50 | 422 before runner work or agent events |
| Python `None` or omitted | No minimum; other invalid Python values raise Pydantic `ValidationError` (a `ValueError`) before run IDs/events |

The order is **resolve scope -> bounded retrieval/merge -> rerank -> optional
per-document quota -> exact capture -> assess distinct documents -> generate
only if sufficient**. The check uses captured `sources[*].chunk.document_id`,
not raw hits, chunk count, titles, URLs, source labels, or caller-selected IDs.
Equality passes. Repeated passages from the same document count once; distinct
IDs with identical titles count separately. No document ID is normalized or
deduplicated by content during this assessment.

The minimum does not refill the candidate pool, retry retrieval, broaden scope,
lower itself, or change scores/order/provenance. `max_chunks_per_document` still
applies first using `DiversityCapGate`; its `diversity_cap_gate` label and path
remain intact. The shared count helper from `MinUniqueSourcesGate` is reused,
but **not its `gate()` discard/provenance-rewrite behavior**. Discarding evidence
would hide failed previews and could send empty context to a generator.

`document_ids` and saved `collection_id` remain mutually exclusive. Collection
membership resolves once before awaits; Python callers can use
`container.paper_collections.resolve(collection_id)`. Unknown IDs match no
passages and do not expand scope. Unknown, broken, or corrupt collections fail
through their existing error paths.

The frozen `EvidencePolicy`, copied observation, immutable resolved scope, and
copied effective limits are request-local before the first await. No shared
index or limit is mutated. A preview is not a reservation: later corpus changes
may change a query's evidence, so `/query` always evaluates its own capture.

## Runnable offline API example

This uses HTTP routes in process, without a network listener or live credentials.
The JSON bodies also work on a trusted local server started via the
[Quickstart](../../QUICKSTART.md).

```bash
uv run python - <<'PY'
from pathlib import Path
from tempfile import TemporaryDirectory
from fastapi.testclient import TestClient
from api.application import create_app
from scripts.demo_evidence_export import offline_settings
from scripts.demo_per_paper_evidence_limits import seed_corpus

with TemporaryDirectory() as directory:
    settings = offline_settings(Path(directory) / "api.sqlite3")
    app = create_app(settings)
    seed_corpus(app.state.container)
    with TestClient(app) as client:
        body = {
            "query": "retrieval evidence",
            "document_ids": ["paper-a"],
            "min_evidence_documents": 2,
        }
        response = client.post("/retrieve", json=body)
        response.raise_for_status()
        preview = response.json()
        assert len(preview["sources"]) == 3
        assert preview["evidence_assessment"] == {
            "required_documents": 2, "observed_documents": 1, "passed": False
        }
        assert app.state.container.event_log.list_events() == []
        response = client.post("/query", json=body)
        response.raise_for_status()
        failed = response.json()["result"]
        assert failed["state"] == "ERROR" and failed["answer"] is None
        events = client.get(f"/runs/{failed['run_id']}/events").json()
        assert events[-1]["payload"]["payload"]["code"] == "insufficient_evidence_documents"
        assert events[-2]["payload"]["sources"] == preview["sources"]
        print("API blocked:", failed["state"], "observed:", preview["evidence_assessment"]["observed_documents"])

        body.update(document_ids=["paper-a", "paper-b"], max_chunks_per_document=1)
        sufficient = client.post("/retrieve", json=body).json()
        assert sufficient["evidence_assessment"]["passed"]
        result = client.post("/query", json=body).json()["result"]
        assert result["state"] == "DONE", result["error"]
        bundle = client.get(f"/runs/{result['run_id']}/export").json()
        assert sufficient["context_sha256"] == bundle["snapshot"]["context_sha256"]
        print("API admitted:", result["state"], "sources:", len(sufficient["sources"]))
PY
```

To use a saved selection, create it with
`POST /collections` and `{"name":"My selection","document_ids":["paper-a","paper-b"]}`,
then replace `document_ids` with the returned `collection_id`. The minimum counts
actual captured members, not the collection's nominal membership.

## Runnable offline Python example

```bash
uv run python - <<'PY'
import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from api.dependencies import AppContainer
from scripts.demo_evidence_export import offline_settings
from scripts.demo_per_paper_evidence_limits import seed_corpus

async def main():
    with TemporaryDirectory() as directory:
        container = AppContainer(offline_settings(Path(directory) / "python.sqlite3"))
        seed_corpus(container)
        options = {"document_ids": ["paper-a"], "min_evidence_documents": 2}
        preview = await container.runner.preview("retrieval evidence", **options)
        assert preview.evidence_assessment.observed_documents == 1
        assert not preview.evidence_assessment.passed
        result = await container.runner.run("retrieval evidence", **options)
        assert result.state == "ERROR" and result.answer is None
        print("Python blocked:", result.state, "observed:", preview.evidence_assessment.observed_documents)

        options["document_ids"] = ["paper-a", "paper-b"]
        options["max_chunks_per_document"] = 1
        preview = await container.runner.preview("retrieval evidence", **options)
        result = await container.runner.run("retrieval evidence", **options)
        assert preview.evidence_assessment.passed
        assert result.state == "DONE", result.error
        bundle = container.evidence_exporter.export(result.run_id)
        assert preview.sources == bundle.snapshot.sources
        assert bundle.plan.observation.evidence_policy.min_evidence_documents == 2
        print("Python admitted:", result.state, "sources:", len(preview.sources))

asyncio.run(main())
PY
```

## Diagnostics, provenance, and remediation

An assessed `/retrieve` response adds only this typed field to its existing
inspection contract:

```json
{
  "evidence_assessment": {
    "required_documents": 2,
    "observed_documents": 1,
    "passed": false
  }
}
```

This is a successful **inspection**, with HTTP 200, full actual context, and no
run ID or agent events. Omitted minimum means the field is absent, not a
fabricated positive assessment. It does not call a live or fake generator.

For insufficient `/query`, the existing response is HTTP 200 with
`result.state="ERROR"`, `answer=null`, and an explicit `error` string containing
required/observed counts. Check the state, not just the HTTP status. The final
`REASONING -> ERROR` event has `payload.code="insufficient_evidence_documents"`
and `payload.evidence_assessment` with the same typed counts. A valid exact
`evidence_snapshot` precedes the error, even for an empty context. There is no
`generation_record`, `ANSWERING`, or `DONE` event and no invented answer.

Use `/runs` to recover the run ID and `/runs/{run_id}/events` to inspect the saved
diagnostic and passages after restart. Failed runs still cannot be exported,
compared as completed answers, or reviewed as completed answers; those existing
operations return 409 rather than manufacturing a bundle.

The requested minimum is preserved in `plan.observation.evidence_policy` and
the initial planning event, alongside any maximum quota. Completed JSON exports
retain this policy; Markdown also reports counts from saved context. The exporter
revalidates the saved policy, matching plan/initial records, and sufficiency.
Saved-run comparisons copy the policy into each side and set
`changes.evidence_policy_changed` even when contexts and answers match.

No database migration or backfill is needed. Existing version-one bundles,
events, previews, and quota-only policies load without the new field. Quota-only
JSON remains `{"max_chunks_per_document": ...}`; `RunConfiguration` retains its
four fields. This is backward readability, not a promise that older binaries
understand newly configured policies.

| Situation | Result and action |
| --- | --- |
| Insufficient document count | Inspect retained passages and IDs; ingest eligible material, explicitly change scope/query, or deliberately choose a lower minimum before starting a new run |
| Minimum exceeds the bounded candidate pool | It may never pass; inspect `configuration.max_source_docs` (a chunk cap), reranking, and quota results instead of expecting automatic refill |
| Invalid request field | 422; send a JSON integer in range, or omit the field |
| Unknown / invalid or broken collection / storage failure | Existing 404 / 409 / 503; repair the selection rather than retrying unscoped |
| Unsupported custom executor or malformed preparation | Query records `ERROR`; preview reports sanitized 500 with its phase code and returns no partial evidence |
| Configured LLM-backed HyDE | Opt-in query records `ERROR` before retrieval; preview remains 409 `generative_retrieval`; no silent algorithm substitution |
| Timeout or cancellation | Existing labeled phase errors or propagated task cancellation; no automatic retry or extra budget |
| Invalid saved policy or insufficient context in a purported completed run | Export/comparison 409 `invalid_run_record`; no reconstruction from current corpus |

## Execution limits and extension contract

The existing effective candidate chunk cap, graph-hop limit, retrieval timeout,
50-source snapshot cap, 262144-byte context limit, and 1048576-byte snapshot
limit all remain enforced. Oversized/inconsistent captures fail explicitly;
they are never truncated into a passing assessment. **Preparation and generation
share one existing reasoning timeout**, not two independent budgets. Captured
evidence may remain after a timeout/cancellation; a cooperative cancellation
observed after preparation also stops before generation.

The built-in executor adds `answer_prepared(plan, snapshot, *, on_generation=...)`
to consume the validated, already captured context without reranking again.
Unconfigured calls keep the legacy `answer` path. With a minimum, an overridden
`answer` cannot silently inherit the base prepared hook: the extension must
explicitly implement the prepared-answer contract or the runner rejects it.
Its `prepare_context` must accept `evidence_policy`, honor scope/quota/capture,
and be model-free. Its prepared answer must consume that snapshot unchanged,
not retrieve/rerank again, and report the actual `GenerationRecord` through the
callback. The runner never invokes this generation hook on insufficient evidence.

Custom Python components are trusted, not sandboxed. They must truthfully expose
whether retrieval uses an LLM and obey the model-free preparation contract.
This guard does not police arbitrary side effects inside injected Python code.
Known LLM-backed HyDE is explicitly rejected for opt-in queries, so the supported
built-in path makes zero live/fake generation calls on insufficiency.

## Limitations, privacy, and design references

Different IDs may represent duplicate ingestions, versions, or dependent studies.
Passing documents may be irrelevant, contradictory, retracted, or poor quality.
Neither a minimum nor the existing token-overlap grounding establishes a claim.
This is not a systematic review, evidence-quality grade, or medical decision aid.

Full previews, error snapshots, queries, IDs, metadata, and saved answers can be
sensitive. Failed runs retain source text durably; deleting/changing corpus
documents does not remove their recorded evidence. Review before sharing.
The API has no built-in authentication or tenant isolation. Scope and no-store
headers are not access control; digests are not signatures. On a sufficient
live query the selected provider still receives the question and full context.
No new provider integration, paid call, model swap, or account-entitlement check
is introduced; see the [current model guide](PROVIDER_MODELS_GUIDE.md).

**Official design references reviewed 2026-09-27:**

| Reference | Inspiration and the remaining gap |
| --- | --- |
| [PaperQA](https://github.com/Future-House/paper-qa) | Distinct gather-evidence, relevance-selection, and answer-generation phases motivate an explicit boundary. This project does not implement PaperQA's LLM-based scoring/summarization or iterative research agent. |
| [LlamaIndex node postprocessors](https://developers.llamaindex.ai/python/framework/module_guides/querying/node_postprocessors/node_postprocessors/) | Processing retrieved nodes before response synthesis motivates checking final context. Similarity/keyword filters are not this exact distinct-document admission contract. |
| [LightRAG](https://github.com/HKUDS/LightRAG) | Returning contexts for inspection/evaluation motivates keeping failed evidence visible. This is not LightRAG's graph algorithm, RAGAS integration, or a claim of equivalent quality. |

The local missing capability was an integrated, opt-in pre-generation requirement
with durable failed-run evidence and generation-free diagnostics. The older
library-only `MinUniqueSourcesGate.gate()` did not provide those contracts.
Only its non-mutating document-count primitive is shared; no upstream assets,
code, or scientific performance claims were copied. The runtime remains the
custom Observe-Decide-Act runner, SQLite, HTTPX/Pydantic, lexical hash vectors,
BM25, and lexical reranking, not LangGraph or semantic proof.

See [retrieval preview](RETRIEVAL_PREVIEW_GUIDE.md),
[per-paper quotas](PER_PAPER_EVIDENCE_LIMITS_GUIDE.md),
[saved-run comparison](RUN_COMPARISON_GUIDE.md),
[Architecture](../../ARCHITECTURE.md), [Safety](../../SAFETY.md), and the
[documentation catalog](../README.md).
