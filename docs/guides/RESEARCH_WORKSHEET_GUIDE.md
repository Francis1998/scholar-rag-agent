# Research evidence worksheets

Inspect the same questions against each selected ingested paper, without
generating answers. `POST /research/worksheet` runs the existing retrieval
preview separately for each question/paper pair and returns a portable JSON or
Markdown worksheet. This prevents a global top-k result list from leaving a
selected paper unrepresented merely because another paper ranks higher.

**A cell contains retrieved passages, not a scientific conclusion.** Lexical
hash vectors, BM25, co-mention traversal, and lexical reranking can return weak
or zero-score matches. Neither passage counts nor scores establish support,
contradiction, answerability, study quality, or completeness.

![Measured synthetic offline research worksheet](../assets/research-worksheet.gif)

This original four-frame illustration is rendered from executed demo output,
not a UI recording or real-model findings.

## Why this workflow

[PaperQA2](https://github.com/Future-House/paper-qa#paperqa2-algorithm) separates
gathering evidence from answer generation.
[kotaemon](https://github.com/Cinnamon/kotaemon#key-features) emphasizes
multi-document QA and inspectable citations with document previews. These
popular open-source projects motivated the evidence-inspection workflow; this
implementation does not copy their code or claim they provide this exact
worksheet.

Scholar RAG already had collection-scoped queries and a single retrieval
preview. The missing step was a bounded question-by-paper view before synthesis.
This feature composes the same preview path instead of adding another source
connector, retrieval algorithm, model dependency, or disconnected scoring helper.

Useful cases include inspecting methods and limitations across a small reading
list, checking whether selected papers appear in a comparison, and preparing
an auditable portfolio demonstration with synthetic notes.

## Reproduce the offline demonstration

From the repository root, after the [Quickstart](../../QUICKSTART.md) install:

```bash
WORKSHEET_DIR="$(mktemp -d)/worksheet-demo"
uv run python -m scripts.demo_research_worksheet --output-dir "$WORKSHEET_DIR"
uv run python -m scripts.create_research_worksheet_gif \
  --transcript "$WORKSHEET_DIR/transcript.txt" \
  --output "$WORKSHEET_DIR/research-worksheet.gif"
```

The demo ingests three synthetic notes into a temporary database and selects
two. It deliberately sets the normal preview source cap to one, so a combined
preview represents one of the two papers while the worksheet inspects both.
Two questions produce four cells, each with one returned passage. It measures
zero live/fake generation calls, zero agent events, Python/HTTP parity, unchanged
database bytes, restart parity on an unchanged corpus, and rejection of unknown
or missing selections. The temporary database is removed.

| Artifact | Contents |
| --- | --- |
| `worksheet.json` | Actual versioned API worksheet |
| `worksheet.md` | Literal Markdown rendering of the same worksheet |
| `checks.json` | Measured counts, statuses, parity checks, and output byte sizes |
| `transcript.txt` | Actual output used to render the GIF |
| `research-worksheet.gif` | Optional generated illustration |

The scripts refuse to overwrite named artifacts. Settings come only from
validated explicit values and defaults: ambient environment, `.env`, provider
keys, and secret-file settings are ignored. No public listener is started.
GIF bytes are reproducible from the same transcript and Pillow environment;
they are not a promise of identical fonts across future dependency versions.

## API workflow

Start the isolated loopback API from the Quickstart and ingest your permitted
text through `/ingest/text`. Recover the returned IDs with `GET /documents`.
Replace `FIRST_DOCUMENT_ID` and `SECOND_DOCUMENT_ID` below with those actual IDs:

```bash
curl --fail-with-body --silent --show-error \
  http://127.0.0.1:8000/research/worksheet \
  -H 'Content-Type: application/json' \
  -d '{
    "questions": ["What retrieval method is described?", "What limitations are discussed?"],
    "document_ids": ["FIRST_DOCUMENT_ID", "SECOND_DOCUMENT_ID"],
    "passages_per_cell": 2
  }'
```

To reuse a saved selection, supply its `collection_id` **instead of**
`document_ids`. See [paper collections](PAPER_COLLECTIONS_GUIDE.md) for creation,
discovery, and revision-checked edits:

```bash
curl --fail-with-body --silent --show-error \
  'http://127.0.0.1:8000/research/worksheet?format=markdown' \
  -H 'Content-Type: application/json' \
  -d '{
    "questions": ["What methods are described?", "What evidence should I inspect?"],
    "collection_id": "REPLACE_WITH_SAVED_COLLECTION_ID",
    "passages_per_cell": 2
  }'
```

The default format is `json`; the only alternative is `markdown`. Successful
responses are attachments named `worksheet.json` or `worksheet.md`, with
`Cache-Control: no-store` and `X-Content-Type-Options: nosniff`. Save only a
successful response; HTTP error bodies are diagnostics, not partial worksheets.
There is no new browser UI, worksheet database table, run ID, or `DONE` event.

### Read the result

`schema_version` is `"1.0"` and `kind` is `"retrieval_passages"`. `documents`
preserves normalized first-seen document order; `rows` preserves question order.
Every row has one cell per selected document, in that same order.

Each cell includes its document ID, effective preview configuration,
`preview_source_count`, `passages_omitted`, and the first requested number of
post-rerank passages. Each passage includes exact document/chunk IDs, rank,
score, retriever, path, full-text SHA-256 and character count, plus bounded title,
source, and excerpt prefixes with explicit truncation flags.

`passages_returned` means a preview returned passages. `no_passages` means that
preview returned none, **not** that the paper lacks evidence or the question is
unanswerable. Scores across different questions/papers are not calibrated
scientific confidence. A digest describes the full recorded passage, not its
possibly truncated excerpt, and is not a signature.

Use a document's `inspection_url` to page through current stored chunks and
match the exact chunk ID. Some path-normalizing identifiers have no safe
inspection URL and return `null`; use Python storage inspection for those.
These links read the current corpus, not frozen historical text.

## Complete offline Python example

This standalone example ignores ambient settings and does not require the API
server. It writes two artifacts into the current directory and refuses existing
filenames. Run it in a fresh working directory with the package installed:

```bash
uv run python - <<'PY'
import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient
from agent.research_worksheet import WorksheetRequest
from api.application import create_app
from scripts.demo_retrieval_preview import offline_settings

async def main():
    for name in ("worksheet.json", "worksheet.md"):
        if Path(name).exists():
            raise FileExistsError(f"Refusing to overwrite {name}")
    with TemporaryDirectory(prefix="worksheet-example-") as temporary:
        app = create_app(offline_settings(Path(temporary) / "corpus.sqlite3"))
        with TestClient(app) as client:
            ids = []
            for title, text in (
                ("Synthetic lexical note", "Synthetic retrieval note: lexical search uses terms."),
                ("Synthetic graph note", "Synthetic retrieval note: graphs connect co-mentions."),
            ):
                response = client.post("/ingest/text", json={
                    "title": title, "text": text, "source": "synthetic:worksheet-guide"
                })
                response.raise_for_status()
                ids.append(response.json()["document_id"])
            worksheet = await app.state.container.worksheets.build(WorksheetRequest(
                questions=("retrieval methods", "graph evidence"),
                document_ids=tuple(ids),
                passages_per_cell=1,
            ))
            print("Cells:", sum(len(row.cells) for row in worksheet.rows))
            print("Events:", len(app.state.container.event_log.list_events()))
            for name, content in (
                ("worksheet.json", worksheet.to_json()),
                ("worksheet.md", worksheet.to_markdown()),
            ):
                with Path(name).open("x", encoding="utf-8") as output:
                    output.write(content)

asyncio.run(main())
PY
```

For an existing `AppContainer`, reuse `container.worksheets.build(...)`.
`WorksheetRequest` also accepts `collection_id`; omit `document_ids` then.
Python input validation raises `ValueError`/Pydantic validation errors.
Operational failures raise `WorksheetError` or the existing `CollectionError`;
external task cancellation propagates instead of producing an empty worksheet.

## Bounds, errors, and consistency

| Bound | Contract |
| --- | --- |
| Questions | 1-5 nonblank strings, at most 500 characters each; order and text retained |
| Selection | Exactly one of `document_ids` or `collection_id`; no whole-corpus default |
| Explicit IDs | 1-10 supplied IDs before deduplication, using the existing scope validator |
| Resolved papers | At most 10, with at most 50 question/paper cells |
| Returned passages | 1-3 per cell, default 2, within the existing preview source cap |
| Excerpts | At most 800 characters each; truncation is explicit |
| Downloads | Each rendering must fit 262,144 UTF-8 bytes; both are checked before success |
| Deadline | 30 seconds overall, including selection, cell previews, and export validation |

The deadline is cooperative, not CPU/process preemption; the existing per-phase
preview timeouts still apply. No shared runner setting is mutated. Every cell
uses the existing planner, scope enforcement, retrieval, reranking, and evidence
capture. Generative HyDE is rejected rather than silently replaced.

| HTTP status | Meaning |
| --- | --- |
| 422 | Invalid/ambiguous/missing selection, unknown explicit document IDs, invalid format or bounds |
| 404 | Unknown saved collection |
| 409 | Invalid/stale collection, selected documents disappeared, or generative retrieval configured |
| 413 | Either serialized rendering is too large; reduce papers, questions, or passages |
| 500 | A cell failed or its provenance was inconsistent; no partial worksheet returned |
| 503 | Selection storage unavailable |
| 504 | Overall or preview phase timeout |

An invalid cell rejects the entire request. Errors expose stable categories and,
where available, zero-based question/document indices, not raw provider or
SQLite diagnostics. Validation errors do not echo request values. Failure never
widens scope or returns an empty success.

Collection membership is resolved once before asynchronous preview work.
All selected IDs must exist; existence is rechecked at completion. This freezes
membership, **not** document contents or indexes. Concurrent corpus updates can
change later cells, and replacing text does not create a corpus snapshot.
Unchanged-corpus restart parity is an offline-demo check, not a guarantee under
mutation. No retrieval model, event log, collection membership, or corpus data is
modified by worksheet construction.

## Privacy, model stack, and portfolio use

This local/trusted API has no authentication or tenant isolation. Questions,
paper identities, excerpts, and labels may be sensitive even without generation.
Protect saved artifacts and the source database. Markdown fences keep untrusted
source text literal; they do not anonymize it. The worksheet is not a systematic
review, medical-decision tool, quality benchmark, or scientific finding.

No live **or fake** model is called. The actual stack is the custom
Observe-Decide-Act planner/preview path, FastAPI, Pydantic, HTTPX, and SQLite,
with deterministic lexical hash vectors, BM25, and co-mention traversal.
It does not require LangGraph, a learned embedding model, or a new provider SDK.

Official catalogs were rechecked on **2026-09-26 America/Los_Angeles**:
GPT-6 Astra, Claude Opus 5.5/Fable 5.1 alongside the compatible Sonnet 5 default,
Gemini 3.8 Flash GA, and Kimi K3. These models concern optional subsequent
generation, not worksheet execution. See the source-linked
[provider guide](PROVIDER_MODELS_GUIDE.md) for configured defaults, endpoint and
thinking-budget limits, and the difference between catalog checks and live API
entitlement.

For a portfolio demonstration, retain the synthetic JSON, Markdown, checks,
transcript, and reproduced GIF together. Explain the global-top-1 versus
per-paper result, identify an exact source chunk, and state what a human must
verify next. Do not present the illustration as a product UI or report these
synthetic checks as research-quality evidence.
