# Usage Examples

Use the [Quickstart](../QUICKSTART.md) for installation and an isolated offline
server. All API examples below use synthetic text. Run them from the repository
root in a second terminal while the server is running.

## Run the local demo

```bash
uv run python scripts/demo_local.py
```

The fake adapter emits a cited placeholder, not a scientific summary. This demo's
temporary database is removed on exit and is separate from the API's database.

## Public API

| Endpoint | Response and purpose |
| --- | --- |
| `GET /health` | `{"status":"ok"}`; service liveness, not a model-provider check |
| `POST /ingest/text` | `document_id`, `chunk_ids`; index supplied text |
| `POST /retrieve` | Plan, scoped post-rerank chunks, scores/ranks/paths, context/digest, and effective bounds; no generation or agent events |
| `POST /query` | `{"result": ...}`; plan, answer, citations, warnings, and run status |
| `GET /runs?limit=20&state=DONE` | Bounded query previews and recorded states, with creation-order cursor pagination; `state` is optional |
| `GET /runs/{run_id}/events` | Event array for the run |
| `GET /runs/{run_id}/export?format=json` | Recorded evidence bundle; JSON is the default format |
| `GET /runs/{run_id}/export?format=markdown` | Human-readable rendering of the recorded bundle |

There are no public PDF-upload, corpus-management, authentication, or multi-turn
chat endpoints. FastAPI also exposes its schema and interactive docs at `/docs`.

## Ingest text through the API

```bash
export BASE_URL=http://127.0.0.1:8000
export REVIEW_DIR="$(mktemp -d)"
printf 'Example artifacts: %s\n' "$REVIEW_DIR"

curl --fail-with-body --silent --show-error "$BASE_URL/ingest/text" \
  -H 'Content-Type: application/json' \
  -d '{
    "title":"Synthetic GraphRAG note",
    "text":"Synthetic example, not a publication. GraphRAG connects co-mentioned entities across passages. An entity link is not proof of a scientific claim.",
    "source":"synthetic:api-example"
  }' > "$REVIEW_DIR/ingest.json"
uv run python -m json.tool "$REVIEW_DIR/ingest.json"
```

`title`, `text`, and `source` are the supported request fields; `source` defaults
to `"api"`. The API assigns `source_type="api"` metadata internally.

## Inspect retrieval before generating

```bash
curl --fail-with-body --silent --show-error "$BASE_URL/retrieve" \
  -H 'Content-Type: application/json' \
  -d '{"query":"What does GraphRAG connect?"}' > "$REVIEW_DIR/preview.json"
uv run python -m json.tool "$REVIEW_DIR/preview.json"
```

Pass optional `document_ids` to select ingested papers; omit it for the full corpus.
Explicit null/empty/invalid scope returns 422, while unknown IDs honestly return
empty evidence without widening. The response is a preview directly, not a
`result` run wrapper. It has no run ID, answer, DONE state, or claim provenance.
Runtime failures are explicit 409/500/504 responses, never empty-success fallbacks.
See the [complete retrieval-preview guide](guides/RETRIEVAL_PREVIEW_GUIDE.md) for
Python usage, capture bounds, privacy, error codes, and the measured offline GIF.

## Query the agent

```bash
curl --fail-with-body --silent --show-error "$BASE_URL/query" \
  -H 'Content-Type: application/json' \
  -d '{"query":"What does GraphRAG connect?"}' > "$REVIEW_DIR/query.json"
uv run python -m json.tool "$REVIEW_DIR/query.json"
```

Check `result.state` before using `result.answer`. Runtime errors are returned as
`state="ERROR"` and `error`, not necessarily an HTTP failure. An empty query is
rejected with HTTP 422. A completed response includes the observed intent,
planned tasks, claims, citation snippets, and grounding warnings. The absence of
warnings is not factual verification.

## Read events and export evidence

If you no longer have `query.json`, discover previous IDs with
`GET /runs?limit=20`; follow its `events_url` and `export_url`. These are navigation
links, not exportability promises. The [run-history guide](guides/RUN_HISTORY_GUIDE.md)
shows restart recovery, keyset pagination, filtering, and legacy behavior.

```bash
RUN_ID="$(uv run python - "$REVIEW_DIR/query.json" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text())["result"]
if result["state"] != "DONE":
    raise SystemExit(f"Run failed: {result['error']}")
print(result["run_id"])
PY
)"

curl --fail-with-body --silent --show-error \
  "$BASE_URL/runs/$RUN_ID/events" > "$REVIEW_DIR/events.json"
curl --fail-with-body --silent --show-error \
  "$BASE_URL/runs/$RUN_ID/export?format=json" > "$REVIEW_DIR/evidence.json"
curl --fail-with-body --silent --show-error \
  "$BASE_URL/runs/$RUN_ID/export?format=markdown" > "$REVIEW_DIR/evidence.md"
```

| Export response | Meaning |
| --- | --- |
| HTTP 404, `run_not_found` | No events exist for this run |
| HTTP 409, `snapshot_unavailable` | A legacy run has no saved evidence snapshot |
| HTTP 409, `run_incomplete` or `run_failed` | The run has not completed successfully |
| HTTP 409, `invalid_run_record` | The saved record cannot be safely exported |
| HTTP 422 | Unsupported `format` value |

The 404/409 responses include `detail.code` and `detail.message`. Exporting does
not silently rerun retrieval or generation to fill missing evidence.
See the [export contract](guides/EVIDENCE_EXPORT_GUIDE.md) before treating an old
event trace as a complete evidence bundle. Exports include exact source text;
review permissions and sensitive content before sharing.

## Network connectors

The following **opt-in Python examples make external requests**. They return
normalized `Document` records; they do not populate the running API's corpus by
themselves, and `/query` does not search these services automatically. Only send
content you may process to `/ingest/text`, or explicitly use `IngestionPipeline`
in your own application. These examples are separate from the offline
walkthrough; API availability and service terms can change.

## Fetch A Paper From OpenAlex

```python
import asyncio

from ingestion.openalex import OpenAlexConnector

# `mailto` is optional but routes traffic to the faster OpenAlex "polite" pool.
connector = OpenAlexConnector(mailto="you@example.org")
document = asyncio.run(connector.fetch_work("W2741809807"))
print(document.title)
print(document.text)  # abstract reconstructed from the inverted index
```

The connector normalizes OpenAlex works into the same `Document` shape as the
PDF, arXiv, and Semantic Scholar connectors, reconstructing the abstract from
OpenAlex's `abstract_inverted_index` field.

## Search Crossref By Keyword

```python
import asyncio

from ingestion.crossref import CrossrefConnector

# `mailto` is optional but routes traffic to the faster Crossref "polite" pool.
connector = CrossrefConnector(mailto="you@example.org")
documents = asyncio.run(connector.search("retrieval augmented generation", max_results=5))
for document in documents:
    print(document.metadata["doi"], document.title)
```

Like PubMed, Crossref is a keyword-search connector: one call returns several
works. Each work is normalized into the same `Document` shape, with the DOI and
publication year captured in metadata and any JATS-XML abstract markup stripped
to plain text.

## Search Europe PMC By Keyword

```python
import asyncio

from ingestion.europepmc import EuropePmcConnector

# `email` is optional but identifies polite API traffic to Europe PMC.
connector = EuropePmcConnector(email="you@example.org")
documents = asyncio.run(connector.search("crispr gene therapy", max_results=5))
for document in documents:
    print(document.metadata["pmid"], document.metadata["doi"], document.title)
```

Europe PMC federates PubMed/MEDLINE, PubMed Central, preprints, and patents, so
one keyword search spans many life-sciences sources. Each result is normalized
into the same `Document` shape, with the DOI, publication year, and PMID captured
in metadata and the article's Europe PMC page used as the source URL.

## Evaluate Retrieval

```bash
uv run python scripts/evaluate_retrieval.py
```

This prints ranked chunks for a single fixture. It is a smoke check, not a
retrieval-quality benchmark. For your own labeled cases, use the opt-in
[evaluation harness](guides/EVALUATION_HARNESS_GUIDE.md). For a complete corpus
review and demonstration path, use the
[research workflow](guides/RESEARCH_WORKFLOW_GUIDE.md).
