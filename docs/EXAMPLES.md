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

## Python library imports

Import standalone retrieval helpers directly in a fresh Python interpreter; no
prior agent or API import is required:

```python
from retrieval.citations import CitationGrounder
from retrieval.claim_verification_gate import ClaimVerificationGate
from retrieval.citation_groundedness_score import CitationGroundednessScorer

grounder = CitationGrounder()
claim_gate = ClaimVerificationGate()
citation_scorer = CitationGroundednessScorer()
```

These helpers are local lexical checks, not scientific verification. Importing
`agent` or `agent.models` does not load the runner or executor. The public
`AgentRunner` export loads on first access: `from agent import AgentRunner`,
`agent.AgentRunner`, and `from agent.runner import AgentRunner` resolve to the
same class. Static type checking uses the explicit export, not the runtime
attribute fallback, so unknown or misspelled exports remain type errors.

For application construction in library code, use
`from api.application import create_app`. Importing this factory does not read
runtime settings or initialize storage; calling it creates the app and storage.
By contrast, importing the server entrypoint `api.main` creates its default
`app` using environment/`.env` settings and initializes the configured database.

## Public API

| Endpoint | Response and purpose |
| --- | --- |
| `GET /health` | `{"status":"ok"}`; service liveness, not a model-provider check |
| `POST /ingest/text` | `document_id`, `chunk_ids`; index supplied text |
| `POST /retrieve` | Plan, scoped post-rerank chunks, scores/ranks/paths, context/digest, and effective bounds; no generation or agent events |
| `GET /documents?limit=20` | Bounded titles/sources, stored chunk counts and selectable IDs; optional exact `source`, literal `title` and exclusive ID `cursor` filters |
| `GET /collections/{collection_id}/screening/export?collection_revision=N&format=json\|csv` | All current human-screening results, labels, revisions and counts in one read snapshot; [download/CSV semantics](guides/SCREENING_EXPORT_GUIDE.md) |
| `POST /query` | `{"result": ...}`; plan, answer, citations, warnings, and run status |
| `GET /runs?limit=20&state=DONE` | Bounded query previews and recorded states, with creation-order cursor pagination; `state` is optional |
| `GET /runs/{run_id}/events` | Event array for the run |
| `GET /runs/{run_id}/export?format=json` | Recorded evidence bundle; JSON is the default format |
| `GET /runs/{run_id}/export?format=markdown` | Human-readable rendering of the recorded bundle |
| `GET /runs/{run_id}/export?format=html` | [Self-contained offline reader](guides/OFFLINE_EVIDENCE_READER_GUIDE.md) with local citation links, literal frozen text and a 4 MiB full-render cap |
| `GET /runs/{run_id}/bibliography?format=bibtex` | Default: cited-only frozen-source BibTeX, deduplicated by exact document ID |
| `GET /runs/{run_id}/bibliography?format=json` | Exact document/cited-chunk provenance, selected metadata, BibTeX, and warnings |
| `GET /runs/{run_id}/corpus-drift` | Bounded unchanged/changed/missing findings comparing frozen chunk identities and digests with the persisted corpus; no retrieval or generation |

There are no public PDF-upload, corpus-update/delete, authentication, or multi-turn
chat endpoints. The read-only [document catalog](guides/DOCUMENT_CATALOG_GUIDE.md)
helps select existing papers. FastAPI also exposes its schema and interactive docs at `/docs`.

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

`text` must be a string containing at least one non-whitespace character.
Missing, null, non-string, empty (`""`), or whitespace-only text (including
Unicode such as `"\u00a0\u2003"`) returns HTTP 422 before ingestion, storage,
or indexing. Nonblank text is passed through unchanged for document IDs and
stored document content; existing chunk whitespace normalization still applies.

## Replace a complete document in Python

The Python ingestion boundary replaces the entire current document, not just
matching chunk IDs. This also applies to `PDFConnector.load`: its document ID is
derived from the resolved file path, so reimporting a changed PDF at the same
path must remove obsolete passages. Use a complete document rather than passing
only an appended passage under an existing ID.

```python
import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from api.dependencies import AppContainer
from retrieval.models import Document
from scripts.demo_evidence_export import offline_settings

with TemporaryDirectory(prefix="scholar-reingestion-") as directory:
    settings = offline_settings(Path(directory) / "corpus.sqlite3")
    container = AppContainer(settings)
    original = Document(
        document_id="same-paper",
        title="Synthetic",
        text="Alpha Beta obsolete methodology.",
        source="fixture",
    )
    old = container.ingestion_pipeline.ingest_documents([original])
    revised = original.model_copy(update={"text": "Gamma Delta revised methodology."})
    current = container.ingestion_pipeline.ingest_documents([revised])
    assert old[0].chunk_id != current[0].chunk_id
    assert container.document_store.list_chunks() == current
    hits = asyncio.run(container.hybrid_retriever.retrieve("Alpha", limit=10))
    assert [hit.chunk for hit in hits] == current
    assert AppContainer(settings).document_store.list_chunks() == current

    blank = revised.model_copy(update={"text": ""})
    assert container.ingestion_pipeline.ingest_documents([blank]) == []
    assert container.document_store.list_chunks() == []
    assert container.document_catalog.list_documents().documents[0].chunk_count == 0
```

An identical replay is idempotent; a shorter/blank revision removes all
superseded chunks, mentions, and edges. In a multi-document batch, unrelated IDs
are retained. Repeated IDs in that batch use the **last complete value** in
first-ID-appearance order; earlier versions do not contribute chunks. IDs are
exact and case-sensitive, with no new normalization.

Chunking, extraction, and index preparation happen before destructive writes.
With the default shared database, document/chunk/graph SQL commits together,
then the already prepared dense/BM25 states are published. Preparation, SQL, and
commit errors propagate without publishing the failed replacement. Two writers
sharing a hybrid instance are serialized; in-flight multi-stage retrieval and
separate workers are not one corpus snapshot or a live index-sync mechanism.
Deliberately using separate document and graph files is **not atomic across
files**: a graph commit can succeed before the document commit fails, leaving
the old document/live indexes and the new graph. Retry the same complete batch
explicitly after resolving the error; no fallback rebuild or automatic repair
is attempted.

Custom legacy writers are not silently replaced with the built-in algorithm.
The pipeline rejects an overridden writer without its explicit replacement
opt-in before storage writes:

| Customized writer | Required opt-in |
| --- | --- |
| Dense/BM25/Hybrid `add_chunks` | `replacing_documents(document_ids, chunks)` context manager |
| `SQLiteDocumentStore.add_documents` | `replacing_documents(documents, chunks)` context manager yielding its SQLite connection |
| `GraphRAGBuilder.index_chunks` | `prepare_chunks(chunks)` returning detached chunk/entity/edge triples without writing |
| `SQLiteGraphStore.replace_chunk` | `replace_documents(document_ids, prepared_chunks, connection=...)` |

An opt-in owns its custom algorithm, validation, and side effects. Index contexts
must prepare before yielding and publish only on successful exit. The graph
writer must use a supplied connection without committing it. A custom hook can
explicitly validate and delegate to the corresponding built-in replacement hook
when its semantics are appropriate; unknown subclasses are never cloned.
Custom chunkers must return unique chunk IDs owned by the input document.
An ID collision with an unrelated stored document fails, rather than overwriting
that paper.

Low-level `SQLiteDocumentStore.add_documents([], chunks)` and retriever
`add_chunks` remain incremental exact-chunk upserts; the store alone does not
synchronize retrieval or graphs. Existing stale evidence for an ID is cleaned
only when that complete document is next ingested, not during startup.
Collections retain membership. Completed exports, reviews, and annotations keep
their frozen evidence, even when corpus-drift inspection reports old chunks as
missing. This is not a historical document-version archive or a new HTTP update
endpoint: `/ingest/text` still rejects blank text and changes its document ID
when source/title/text changes.

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
`state="ERROR"` and `error`, not necessarily an HTTP failure. Both `/query` and
`/retrieve` reject non-string, empty, or whitespace-only questions with HTTP 422
before runner execution, model calls, or agent-event writes. This includes JSON
values such as `""`, `" \t\n"`, and `"\u00a0\u2003"`. Validation passes nonblank
strings through unchanged; the existing analyzer still trims their outer
whitespace in observations, plans, and generation prompts.

A completed response includes the observed intent, planned tasks, claims,
citation snippets, and grounding warnings. The absence of warnings is not
factual verification.

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

## Import cited references without another query

```bash
curl --fail-with-body --silent --show-error \
  "$BASE_URL/runs/$RUN_ID/bibliography?format=json" > "$REVIEW_DIR/bibliography.json"
uv run --no-sync python -m json.tool "$REVIEW_DIR/bibliography.json"
curl --fail-with-body --silent --show-error \
  "$BASE_URL/runs/$RUN_ID/bibliography" > "$REVIEW_DIR/bibliography.bib"
```

Review the JSON warnings and citation mapping, then use a reference manager's
BibTeX file-import workflow. Only the saved answer's final citations qualify;
conflicting metadata uses the first cited frozen rank with a warning, and
missing metadata is not recovered from current documents or DOI services.
No live/fake model or event writer is invoked. Both representations must fit
262,144 UTF-8 bytes; failures never yield partial references. This API example
has title-only metadata. See the [full field/error/Python contract and actual-output
GIF](guides/SAVED_BIBLIOGRAPHY_GUIDE.md); do not compile imported LaTeX to inspect it.

## Inspect corpus drift without regenerating

```bash
curl --fail-with-body --silent --show-error \
  "$BASE_URL/runs/$RUN_ID/corpus-drift" > "$REVIEW_DIR/corpus-drift.json"
uv run python -m json.tool "$REVIEW_DIR/corpus-drift.json"
```

The response preserves frozen source order and compares exact text/title/source
digests and canonical chunk metadata. Missing documents, missing chunks, and
reassigned chunk IDs are distinguished. It does not return current source text
or modify the original exports. “Unchanged” is not whole-paper equality or a
scientific-support judgment. Corruption, excessive read sizes, and storage
failures are errors, not partial results. See the complete
[API/Python contract and offline demo](guides/CORPUS_DRIFT_GUIDE.md).

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
