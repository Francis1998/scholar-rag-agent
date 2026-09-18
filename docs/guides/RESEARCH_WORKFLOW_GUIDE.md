# Research workflow and portfolio guide

Build a small, inspectable literature workflow: ingest text you may use, ask
questions, inspect the retrieved evidence, and save a reviewable run artifact.
The same path can demonstrate your engineering work without presenting synthetic
text or an offline placeholder answer as scientific findings.

**This walkthrough uses invented workshop notes, not published papers or measured
results.** It exercises the real local API with the fake model adapter. No model
credentials or remote inference are needed.

## 1. Prepare a reproducible local workspace

Follow [Quickstart steps 1-3](../../QUICKSTART.md) to install, try the local demo,
and start a loopback-only API with a fresh SQLite database and all four provider
keys explicitly empty. Do not ingest the Quickstart example if you want an empty
corpus for the check below. Keep the database path printed by that terminal;
restarting against that same path preserves documents, run events, and snapshots.

In a second terminal, from the repository root:

```bash
export BASE_URL=http://127.0.0.1:8000
export REVIEW_DIR="$(mktemp -d)"
printf 'Review artifacts: %s\n' "$REVIEW_DIR"
git rev-parse HEAD > "$REVIEW_DIR/revision.txt"
cp uv.lock "$REVIEW_DIR/uv.lock"
uv --version > "$REVIEW_DIR/uv-version.txt"
uv run python --version > "$REVIEW_DIR/python-version.txt"
curl --fail-with-body --silent --show-error "$BASE_URL/health"
```

Keep artifacts outside the repository to avoid accidentally committing source
text. Record the source revision and dependency lockfile with your demonstration.
Fresh UUIDs and timestamps will differ between runs; a saved snapshot reproduces
the recorded evidence, not a promise of identical future model output.

On the empty database, this negative control should finish with
`result.answer.ungrounded` set to `true`, an `[UNGROUNDED]` answer, and no citations:

```bash
curl --fail-with-body --silent --show-error "$BASE_URL/query" \
  -H 'Content-Type: application/json' \
  -d '{"query":"What evidence is available in this empty corpus?"}' \
  > "$REVIEW_DIR/empty-corpus.json"
uv run python -m json.tool "$REVIEW_DIR/empty-corpus.json"
```

If the corpus already contains text, use a new database before treating this as
an empty-corpus check. A `DONE` state means the workflow completed, not that its
answer is correct.

## 2. Ingest a bounded corpus

Run these three requests once against the fresh database:

```bash
curl --fail-with-body --silent --show-error "$BASE_URL/ingest/text" \
  -H 'Content-Type: application/json' \
  -d '{
    "title":"Synthetic A - Co-mention retrieval",
    "text":"Synthetic workshop note, not a publication. GraphRAG links co-mentioned entities to expand retrieval across passages. In this invented scenario, expansion can find a related passage. No benchmark or measured improvement is reported.",
    "source":"synthetic:workshop-a"
  }' > "$REVIEW_DIR/ingest-a.json"

curl --fail-with-body --silent --show-error "$BASE_URL/ingest/text" \
  -H 'Content-Type: application/json' \
  -d '{
    "title":"Synthetic B - Keyword baseline",
    "text":"Synthetic workshop note, not a publication. BM25 matches query terms in passages. In this invented scenario, ambiguous entities can introduce irrelevant passages during GraphRAG expansion. GraphRAG does not guarantee better retrieval than BM25. No evaluation is provided.",
    "source":"synthetic:workshop-b"
  }' > "$REVIEW_DIR/ingest-b.json"

curl --fail-with-body --silent --show-error "$BASE_URL/ingest/text" \
  -H 'Content-Type: application/json' \
  -d '{
    "title":"Synthetic C - Evaluation boundaries",
    "text":"Synthetic workshop note, not a publication. Comparing GraphRAG and BM25 requires held-out labeled queries and human review of source passages. A co-mention path is a retrieval aid, not proof that a claim is true. These workshop notes contain no validated scientific findings.",
    "source":"synthetic:workshop-c"
  }' > "$REVIEW_DIR/ingest-c.json"
```

Each response contains `document_id` and `chunk_ids`. The text endpoint accepts
`title`, `text`, and `source`; it is not a PDF-upload endpoint or a general metadata
import API. Repeating ingestion is not a corpus-reset mechanism: use a fresh
database for a clean experiment.

For your own papers, use text you authored or have permission to process. Keep a
separate corpus manifest with title, source identifier, permission/license, text
version, and extraction method. A source URL or open-access flag alone does not
establish redistribution rights. Preserve relevant methods and limitations;
abstract-only input cannot support conclusions requiring the full paper.

The [PDF and source adapter catalog](../README.md#source-adapters-papers-repositories-and-registries)
describes optional Python ingestion paths. The `scholar-rag-ingest` CLI prints a
normalized document; it does **not** populate the running API's SQLite corpus.

## 3. Ask representative questions

Start with comparison and hypothesis requests:

```bash
curl --fail-with-body --silent --show-error "$BASE_URL/query" \
  -H 'Content-Type: application/json' \
  -d '{"query":"Compare GraphRAG and BM25 retrieval in this synthetic corpus."}' \
  > "$REVIEW_DIR/comparison.json"

curl --fail-with-body --silent --show-error "$BASE_URL/query" \
  -H 'Content-Type: application/json' \
  -d '{"query":"What evidence supports or refutes the hypothesis that GraphRAG always improves retrieval?"}' \
  > "$REVIEW_DIR/hypothesis.json"
```

The analyzer uses keyword rules, not a learned intent classifier:

| Question to try | Expected intent and plan |
| --- | --- |
| `What does GraphRAG link in the synthetic notes?` | `factual_lookup`; direct lookup task |
| `Summarize the limitations in the synthetic notes.` | `synthesis`; corpus synthesis task |
| The comparison request above | `comparison`; `comparison-evidence` and `contrast-findings` |
| The hypothesis request above | `hypothesis_validation`; `supporting-evidence` and `counter-evidence` |

Task names describe retrieval intent. Their hits are merged before generation;
the response is not an adjudicated supporting-versus-refuting evidence table.
Keyword precedence can also affect routing, so inspect the returned plan rather
than assuming the wording produced the route you intended.

## 4. Inspect the claims, evidence, and warnings

```bash
uv run python - "$REVIEW_DIR/comparison.json" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text())["result"]
if result["state"] != "DONE":
    raise SystemExit(f"Run failed: {result['error']}")
print("Run:", result["run_id"])
print("Intent:", result["observation"]["intent"])
for task in result["plan"]["tasks"]:
    print("Task:", task["task_id"], task["query"])
answer = result["answer"]
print("Answer:", answer["answer"])
print("Ungrounded:", answer["ungrounded"])
print("Warnings:", answer["warnings"])
for claim in answer["claims"]:
    print("Claim:", claim["text"], claim["grounded"], claim["chunk_ids"])
for citation in answer["citations"]:
    print("Citation:", citation["chunk_id"], citation["title"], citation["snippet"])
PY
```

Repeat with `hypothesis.json`. HTTP success alone is insufficient: `/query` can
return `result.state = "ERROR"` with `result.error`. Invalid request bodies produce
HTTP validation errors. [API examples](../EXAMPLES.md) show the response contract.

**The offline adapter echoes the question with source IDs.** It does not write a
scientific comparison or validate the hypothesis. A useful demonstration here is
the ingestion, plan, source linkage, warning handling, and saved evidence path.
For live drafting, use the [provider model guide](PROVIDER_MODELS_GUIDE.md), check
the chosen provider in the run provenance, and review the text sent off-machine.
Catalog verification is not a live inference or account-entitlement test.

`CitationGrounder` accepts a mapped claim when it shares at least one meaningful
term with a retrieved chunk. Consequently `grounded: true`, a citation, or an
empty warnings list does **not** establish entailment or factual correctness.
Citation snippets are limited previews; read the exact chunk in the export and,
for real research, the complete original source.

Make a separate reviewer note for each material claim:

| Record | Review question |
| --- | --- |
| Claim text and chunk IDs | Do the named passages actually state this claim? |
| Exact passage and source version | Does surrounding context change the meaning? |
| Methods, population, and conditions | Is the claim limited to the study's actual setting? |
| Opposing or missing passages | Were disagreements or important gaps omitted? |
| Decision and rationale | Accept, revise, or reject after reading; do not copy a heuristic label |

## 5. Export the recorded run

Extract the completed comparison run ID, save the event trace, and export both
formats:

```bash
RUN_ID="$(uv run python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["result"]["run_id"])' \
  "$REVIEW_DIR/comparison.json")"

curl --fail-with-body --silent --show-error \
  "$BASE_URL/runs/$RUN_ID/events" > "$REVIEW_DIR/events.json"
curl --fail-with-body --silent --show-error \
  "$BASE_URL/runs/$RUN_ID/export?format=json" > "$REVIEW_DIR/evidence.json"
curl --fail-with-body --silent --show-error \
  "$BASE_URL/runs/$RUN_ID/export?format=markdown" > "$REVIEW_DIR/evidence.md"
uv run python -m json.tool "$REVIEW_DIR/evidence.json"
```

The [evidence export guide](EVIDENCE_EXPORT_GUIDE.md) documents the versioned
schema, model provenance, source chunks, and behavior for legacy or failed runs.
Read `evidence.md` alongside the JSON, not instead of checking source text.
Do not edit the snapshot to make an answer look better; keep corrections in your
reviewer note and create a new run when the corpus or question changes.

To demonstrate persistence, stop the server with Ctrl-C and restart with the
**same database path and the same empty-key settings**, without ingesting again.
Fetch the same export URL. This reads recorded evidence without another model
call, even after restart or later corpus changes. It does not rerun retrieval or
promise that a new LLM call would produce the same answer. Back up the database
and permitted exports before removing a temporary workspace.

## 6. Present a portfolio demonstration

Use a small, explicit scenario rather than a claim about research impact:

1. Identify the corpus as synthetic or explain your permission to use it.
2. Show one ingestion response and the comparison/hypothesis plan.
3. Open a claimed citation, inspect the exact saved passage, and explain the
   lexical grounder's limits.
4. Show the empty-corpus warning and a saved export after restart.
5. Separate what you implemented or changed from the upstream toolkit. Include
   the code revision, run artifacts, and your review notes where sharing is allowed.

The [synthetic demo GIF](../assets/evidence-export.gif) and its
[reproduction instructions](../DEMO.md) show a generated animation from the
synthetic offline workflow, not a live research UI recording. Label any animation
you reuse accordingly. Do not present placeholder answers, invented study results,
or retrieved-source counts as measured scientific quality.

For an actual comparison of retrieval approaches, define a held-out labeled
corpus and an evaluation protocol first. The
[evaluation harness](EVALUATION_HARNESS_GUIDE.md) is opt-in; its lexical answer
metric is not a factuality benchmark. Record your own results only after running
the protocol, including failures and relevant baselines.

## Acceptance checklist

- [ ] The revision, dependency lockfile, runtime version, corpus scope, and text
  permissions are recorded.
- [ ] The API is bound to loopback and the offline run uses no provider credentials.
- [ ] Each input has a saved document/chunk-ID response; no paper or result is
  misrepresented as real.
- [ ] Comparison and hypothesis requests reach `DONE`, with the intended plan
  visible; `ERROR` and warnings are checked explicitly.
- [ ] The empty-corpus run is ungrounded; a successful grounding label is not
  treated as proof.
- [ ] Every material claim has been reviewed against exact source text, with
  disagreements and limitations retained.
- [ ] JSON and Markdown exports are saved, inspectable, and retrievable after
  restart using the same database.
- [ ] Any shared artifact is reviewed for sensitive text and redistribution
  permissions; reviewer decisions remain separate from the recorded run.
- [ ] The presentation distinguishes engineering workflow evidence from
  scientific evaluation or real model synthesis.

## Privacy and boundaries

SQLite, event traces, and exports may contain queries, answers, source text,
document metadata, identifiers, and local paths. Nonsecret model provenance is
not anonymization of the rest of the artifact. Share only a reviewed copy with
permitted content; do not publish `.env`, credentials, or an entire working
database. Turning on a live provider sends retrieved context to that provider.

This is a local-first toolkit, not an authenticated multi-tenant service or a
complete systematic-review platform. There is no built-in PDF-upload UI,
automatic PRISMA decision process, novelty proof, or medical-decision workflow.
Keep human source review in the loop. Read [Safety](../../SAFETY.md) before using
non-synthetic material or making the service reachable beyond your machine.
