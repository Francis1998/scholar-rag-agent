# Opt-in near-duplicate evidence collapse

Pass **`near_duplicate_threshold`** to `POST /query`, `POST /retrieve`,
`AgentRunner.run`, or `AgentRunner.preview` to remove lexically overlapping
passages from the already bounded, reranked evidence pool. Both endpoints reuse
the existing `NearDuplicateCollapser`; there is no second deduplication algorithm.
Omission preserves the previous behavior, response shape, and legacy call contracts.

This is **text-term overlap, not semantic equivalence, paper identity, evidence
independence, or scientific truth**. Important similar passages can be dropped.
Inspect a baseline and an opted-in preview before choosing a threshold.

![Measured synthetic offline evidence collapse](../assets/near-duplicate-evidence.gif)

**Actual-output illustration:** the [executed transcript](../assets/near-duplicate-evidence.txt)
shows five synthetic passages from four documents becoming three passages from
two documents at threshold `0.8`. Context changes from **435 to 264 UTF-8 bytes**.
Adding a per-document cap of one leaves two passages. A query requiring four
documents records `ERROR` and the exact collapsed snapshot without generation.
The entire demonstration makes **zero live or fake model calls and zero
external HTTP attempts**; previews make zero agent-event writes. This is not
a screen recording, token-cost claim, latency benchmark, or retrieval-quality score.

## Reproduce the measured demo

From the repository root with Python 3.11+ and uv:

```bash
uv sync --locked --extra dev
DEMO_DIR="$(mktemp -d)/near-duplicate-evidence"
uv run python -m scripts.demo_near_duplicate_evidence --output-dir "$DEMO_DIR"
uv run python -m scripts.create_near_duplicate_evidence_gif \
  --transcript "$DEMO_DIR/transcript.txt" \
  --output "$DEMO_DIR/near-duplicate-evidence.gif"
uv run python -m json.tool "$DEMO_DIR/checks.json"
printf 'Review synthetic artifacts in %s\n' "$DEMO_DIR"
```

The demo executes real FastAPI routes, planning, hybrid/graph retrieval, lexical
reranking, collapse, quotas, capture, and SQLite error events. Its explicit
`offline_settings` ignores ambient environment, `.env`, secret files, database
settings, and provider keys. Model generation and external HTTP are also denied.
The temporary database is removed on exit; use fresh output paths because named
artifacts and GIFs are never overwritten. Pillow is already a development dependency.

| Artifact | Contents |
| --- | --- |
| `baseline-preview.json` | Unconfigured five-passage evidence and exact context |
| `collapsed-preview.json` | Threshold `0.8`, three surviving passages and provenance |
| `term-set-preview.json` | Threshold `1`, four passages; equal term sets are not necessarily identical text |
| `capped-preview.json` | Collapse first, then one passage per document |
| `failed-run.json`, `failed-events.json` | Actual insufficient query, saved context, and minimum-document diagnostic |
| `checks.json`, `transcript.txt` | Measured identities, bytes, counts, integrity and zero-call checks; the four GIF panels |

Run IDs and event timestamps vary. The transcript and image frames are reproducible
for these fixtures and locked dependencies. The GIF does not show a completed
answer: the demo intentionally stops its query before any model call.

## Request contract

| Input | Behavior |
| --- | --- |
| Omitted | Collapse disabled, not the library's `0.9` default |
| Finite JSON number greater than zero and at most one | Inclusive lexical similarity threshold; `1` and `1.0` are accepted |
| HTTP `null`, bool, numeric string, array/object, zero, negative, above one, NaN/infinity/overflow | HTTP 422 before runner work or events |
| Python omitted or `None` | Collapse disabled |
| Invalid Python value | Pydantic `ValidationError` (a `ValueError`) before observation, run IDs, or event writes |

Invalid threshold diagnostics retain their field location, type, and message,
but omit rejected input values so even non-finite numbers yield valid JSON, not
an error-serialization failure. Unrelated validation keeps its existing contract.
The request field and the corresponding `EvidencePolicy` field are frozen.

```json
{
  "query": "Compare the evidence in my selected papers",
  "document_ids": ["paper-a", "paper-b"],
  "near_duplicate_threshold": 0.9,
  "max_chunks_per_document": 2,
  "min_evidence_documents": 2
}
```

Use either `document_ids` or a saved `collection_id`, never both. Collection
membership resolves once before awaits. Unknown document IDs match no chunks;
unknown or broken collections fail without widening scope. Python callers can
resolve a collection with `container.paper_collections.resolve(collection_id)`
and pass the resulting IDs to the runner.

## What executes, in order

**Resolve scope -> bounded retrieval/merge -> rerank -> optional collapse ->
optional per-document cap -> exact capture -> optional minimum-document
assessment -> query generation only if sufficient.**

The existing collapser considers candidates in stable descending score order;
ties preserve their incoming order. It uses `meaningful_terms` on **chunk text**:
whitespace tokenization, lowercase, selected surrounding punctuation removal,
stopword removal, and retention of terms containing a Unicode letter or number.
Titles, IDs, metadata, citations, and source URLs do not decide matches.

For each candidate, Jaccard similarity is the shared-term count divided by the
union-term count. A match at or above the threshold against any already-kept
representative drops the candidate. Otherwise it survives. This is a greedy
representative procedure, **not transitive connected-component clustering**:
dropped passages do not themselves suppress later candidates.

Threshold `1` means **equal meaningful-term sets**, not byte-identical text.
Word order and repetition are ignored: `alpha beta alpha` and `beta alpha`
have equal sets. Two empty sets, including stopword-only texts, match at `1`;
empty versus nonempty sets have similarity zero. Lower thresholds allow weaker
overlaps to match; exact survivor choices depend on the greedy ordering.

The integrated path preserves the collapser's survivor order, exact scores, and
whole chunks, then renumbers captured ranks from one. It does not average scores,
merge text, change paper ownership, or update the corpus. The optional
`DiversityCapGate` runs **after** collapse, so a representative removed by the
quota does not cause a previously dropped near-duplicate to reappear.

There is no refill, oversampling, extra retrieval, network work, or shared mutable
setting. `max_source_docs` remains a chunk-result cap, not a paper minimum.
Existing source/hop/timeout and capture limits remain unchanged (at most 50 final
sources, 262,144 context bytes, and 1,048,576 serialized snapshot bytes).

## Runnable offline API example

These in-process requests need no server, keys, or network:

```bash
uv run python - <<'PY'
from pathlib import Path
from tempfile import TemporaryDirectory
from fastapi.testclient import TestClient
from api.application import create_app
from scripts.demo_evidence_export import offline_settings
from scripts.demo_near_duplicate_evidence import QUERY, seed_corpus

with TemporaryDirectory() as directory:
    app = create_app(offline_settings(Path(directory) / "api.sqlite3"))
    seed_corpus(app.state.container)
    with TestClient(app) as client:
        body = {"query": QUERY, "near_duplicate_threshold": 0.8}
        response = client.post("/retrieve", json=body)
        response.raise_for_status()
        preview = response.json()
        assert len(preview["sources"]) == 3
        assert app.state.container.event_log.list_events() == []
        body["min_evidence_documents"] = 4
        response = client.post("/query", json=body)
        response.raise_for_status()
        result = response.json()["result"]
        assert result["state"] == "ERROR" and result["answer"] is None
        events = client.get(f"/runs/{result['run_id']}/events").json()
        assert events[-2]["payload"]["sources"] == preview["sources"]
        assert events[-1]["payload"]["payload"]["code"] == "insufficient_evidence_documents"
        print("API: 3 surviving passages; minimum 4 -> ERROR without generation")
PY
```

The same JSON bodies work on a trusted loopback server from the
[Quickstart](../../QUICKSTART.md). Inspect `/query`'s `result.state`, not only
HTTP status. To generate normally, choose a supported minimum or omit it; that
permits the existing configured provider or fake adapter to receive the context.

## Runnable offline Python example

```bash
uv run python - <<'PY'
import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from api.dependencies import AppContainer
from scripts.demo_evidence_export import offline_settings
from scripts.demo_near_duplicate_evidence import QUERY, seed_corpus

async def main():
    with TemporaryDirectory() as directory:
        container = AppContainer(offline_settings(Path(directory) / "python.sqlite3"))
        seed_corpus(container)
        baseline = await container.runner.preview(QUERY, near_duplicate_threshold=None)
        preview = await container.runner.preview(QUERY, near_duplicate_threshold=0.8)
        assert len(baseline.sources) == 5 and len(preview.sources) == 3
        assert container.event_log.list_events() == []
        result = await container.runner.run(
            QUERY, near_duplicate_threshold=0.8, min_evidence_documents=4
        )
        assert result.state == "ERROR" and result.answer is None
        print("Python: 5 -> 3 passages; minimum 4 -> ERROR without generation")

asyncio.run(main())
PY
```

The original library API remains available:

```python
from retrieval.near_duplicate_collapse import NearDuplicateCollapser

survivors = NearDuplicateCollapser(threshold=0.9).collapse(retrieved_results)
```

Library `threshold` still defaults to `0.9` and accepts the original finite
inclusive range from zero to one; `top_k` remains a library-only option.
Direct library calls return the original surviving objects without adding trace
labels. The stricter opt-in API deliberately excludes zero and does not expose
`top_k` or enable collapse when omitted.

## Frozen provenance, exports, and comparisons

The runner constructs a request-local immutable policy and detached observation
before its first await. It retains the requested threshold in the plan,
initial/decision/retrieval events, preview, and completed JSON/Markdown exports.
Concurrent requests cannot change one another's threshold or frozen scope/limits.
`RunConfiguration` retains its four fields and version-one records without the
threshold remain readable; absent policy fields stay absent.

After collapse, each survivor has `retriever="near_duplicate_collapse"` and
`path=[*previous_path, previous_retriever]`. With a quota, its outer retriever is
`diversity_cap_gate` and the path ends in `near_duplicate_collapse`. Full chunks,
text digests, scores, and relative score order remain unchanged.

This trace records **which transformations ran**, not dropped-to-survivor
cluster mappings or pairwise similarity scores. Only final surviving text is
saved in the snapshot. A query's reasoning event retains retrieved candidate IDs,
not every candidate's text. Save a baseline preview if you need to inspect
discarded passages; the demo's `removed_chunk_ids` is a measured baseline diff,
not a new API field.

Completed runs still export through `/runs/{run_id}/export?format=json|markdown`.
Exports revalidate the policy, stage provenance, and final-source collapse
postcondition without consulting today's corpus or generating. Saved comparisons
retain each side's policy and set `changes.evidence_policy_changed` even when
contexts match, for example thresholds `0.9` and `1` on this fixture. A policy
change is not a controlled model-quality comparison.

Under `min_evidence_documents`, collapse can remove the last passage from a
document. An unmet minimum preserves the exact snapshot and typed count
diagnostic in an `ERROR` run without calling generation. Such failed runs are
inspectable through `/runs/{run_id}/events`, **not completed exports**.
Previews retain insufficient evidence and report `evidence_assessment` without
any events. A preview is not a reservation against later corpus changes.

## Errors and extension contracts

| Condition | Outcome |
| --- | --- |
| Invalid threshold | HTTP 422 or Python `ValidationError`, before work |
| Collapse/preparation exception | Query `ERROR`; preview sanitized 500 `context_preparation_failed`, no partial evidence or fallback |
| Scope violation before collapse | Explicit failure, even if collapse could otherwise hide that passage |
| Unmet document minimum | Query `ERROR` with saved snapshot and diagnostic; preview 200 with failed assessment |
| Unsupported custom executor signature or ignored policy/provenance | Explicit failure; never retry without the requested policy |
| Inconsistent saved policy/provenance | Export/comparison 409 `invalid_run_record` |
| LLM-backed HyDE in preview | Existing 409, not substitution with a different retrieval algorithm |

Opted-in `prepare_context`/`answer` overrides must accept `evidence_policy` and
preserve the existing capture callbacks, exact-context, and scope contracts.
The validator checks final-source provenance and reuses the collapser to verify
that final sources already satisfy the threshold and ordering; it does not
silently repair them. With a minimum, custom answer overrides must explicitly
implement the prepared-context generation hook as documented in the
[minimum-evidence guide](MINIMUM_EVIDENCE_DOCUMENTS_GUIDE.md#execution-limits-and-extension-contract).
These are execution-contract checks, not a sandbox against arbitrary extension code.

## Safety, configuration, and workflow inspiration

Keep the service local/trusted: scope is not authorization or tenant isolation.
Full passages, queries, metadata, and failed-run snapshots may be sensitive.
Review permissions before ingesting, sending surviving context to a live model,
or sharing previews, events, GIF inputs, or exports. Jaccard cannot distinguish
paraphrases, scientific contradictions, independent studies, or duplicate paper
identities. It can discard the passage whose small difference matters most.

There is no new environment variable, provider call, dependency, endpoint, model
default, or budget. The stack remains FastAPI, Pydantic, HTTPX, SQLite, and a
hand-written state machine; default retrieval uses lexical hash vectors and BM25,
not learned semantic embeddings or LangGraph. See
[configuration](../../CONFIGURATION.md), [safety](../../SAFETY.md),
[architecture](../../ARCHITECTURE.md), and [provider compatibility](PROVIDER_MODELS_GUIDE.md).

Public prior art checked **2026-10-04** motivates the workflow, not algorithm or
feature equivalence:

- [LlamaIndex node postprocessors](https://developers.llamaindex.ai/python/framework/module_guides/querying/node_postprocessors/node_postprocessors/)
  places filtering/reordering between retrieval and synthesis. Its
  `SimilarityPostprocessor` filters query similarity, **not this cross-chunk
  lexical deduplication algorithm**.
- [PaperQA](https://github.com/Future-House/paper-qa) demonstrates research
  evidence gathering and its retrieval/answer workflow (9,307 stars observed).
- [RAGFlow](https://github.com/infiniflow/ragflow) emphasizes context quality and
  traceable chunk inspection (91,678 stars observed).

No third-party code is copied. Star counts are dated observations, not quality
scores. This change integrates the repository's existing lexical helper rather
than claiming those projects' reasoning, retrieval, or document-understanding features.
