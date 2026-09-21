# Discover and select your local paper corpus

`GET /documents` recovers selectable document IDs from the existing SQLite
corpus, including after a service restart. Browse bounded titles, sources, and
stored chunk counts; filter the list; then pass the selected IDs to `/query`.
You no longer have to preserve every `/ingest/text` response to select papers.

Before selecting a query scope, use the
[stored chunk evidence reader](DOCUMENT_CHUNKS_GUIDE.md) to inspect bounded
passages from one discovered document without running retrieval or generation.

![Measured synthetic corpus discovery](../assets/document-catalog.gif)

This generated illustration uses the offline demo's actual API responses. It is
not a browser UI recording or evidence of scientific accuracy.

## Why this feature

[RAGFlow's document-list API](https://github.com/infiniflow/ragflow/blob/main/docs/references/http_api_reference.md#list-documents)
and [Haystack's DocumentStore protocol](https://docs.haystack.deepset.ai/docs/document-store)
make inspecting an existing corpus a first-class workflow. Scholar previously
offered document-scoped queries but no HTTP discovery path for the saved IDs.
This feature closes that local discovery gap, without adding a connector,
copying either project, or claiming their dataset-management capabilities.

The implementation uses the existing FastAPI/Pydantic/SQLite stack. Browsing
does not call an LLM, run the agent state machine, rebuild indexes, or append
events. The application still initializes its normal in-memory indexes at
startup; the standalone Python reader below avoids that initialization.

## API walkthrough

Start the isolated offline service in the [Quickstart](../../QUICKSTART.md),
ingest some synthetic notes, then run:

```bash
export BASE_URL=http://127.0.0.1:8000
curl --fail-with-body --silent --show-error --get "$BASE_URL/documents" \
  --data-urlencode 'limit=2' | uv run python -m json.tool
```

Each response contains `documents` and `next_cursor`. Every summary has exactly:

| Field | Meaning |
| --- | --- |
| `document_id` | Exact stored ID, suitable for `/query` `document_ids`; never shortened |
| `title` / `title_truncated` | At most 300 characters and whether the original label was longer |
| `source` / `source_truncated` | At most 512 characters and whether the original label was longer |
| `chunk_count` | Number of currently stored chunks belonging to this document |

The list omits document/chunk bodies, arbitrary metadata, vectors, and run data.
A zero chunk count is valid: the document exists but has no searchable chunks.
This count is not a guarantee of retrieval relevance or scientific coverage.

### Filter before paging

```bash
curl --fail-with-body --silent --show-error --get "$BASE_URL/documents" \
  --data-urlencode 'limit=2' \
  --data-urlencode 'source=synthetic:methods' \
  --data-urlencode 'title=graph' | uv run python -m json.tool
```

`source` is an exact, case-sensitive match against the full stored source.
`title` is a literal substring match using SQLite's built-in `lower`; ASCII
case-insensitivity is supported, not full Unicode case folding. `%`, `_`, quotes,
and backslashes are literal characters, not search-language wildcards. Both
filters apply before the page limit, and both must match when supplied together.

Filters are not stripped of whitespace. Empty filters are invalid; omit a filter
instead of passing an empty string. A shortened `source` label is only a preview,
not necessarily a valid exact filter value. Sources longer than the filter's
512-character maximum can still be discovered unfiltered or by title.

### Continue and select

If `next_cursor` is non-null, use that exact value as the next request's `cursor`,
preserving the same filters:

```bash
# Replace the example values with IDs returned by your catalog.
export CURSOR='doc-replace-with-next-cursor'
curl --fail-with-body --silent --show-error --get "$BASE_URL/documents" \
  --data-urlencode 'limit=2' \
  --data-urlencode "cursor=$CURSOR" \
  --data-urlencode 'source=synthetic:methods' \
  --data-urlencode 'title=graph'

curl --fail-with-body --silent --show-error "$BASE_URL/query" \
  -H 'Content-Type: application/json' \
  -d '{"query":"Compare the retrieved methods","document_ids":["doc-replace-with-selected-id"]}'
```

These placeholders are not real IDs. The executable Python example below needs
no substitutions. Always inspect the query's `result.state` and warnings;
unknown IDs match no evidence rather than widening to the whole corpus.

## Python: recover an ID and query after restart

After installing the repository's dev extras, run this complete synthetic example
from the repository root:

```bash
uv run python - <<'PY'
from pathlib import Path
from tempfile import TemporaryDirectory
from fastapi.testclient import TestClient
from api.application import create_app
from scripts.demo_evidence_export import offline_settings
from storage.document_catalog import SQLiteDocumentCatalog

with TemporaryDirectory() as directory:
    path = Path(directory) / "corpus.sqlite3"
    settings = offline_settings(path)
    with TestClient(create_app(settings)) as client:
        client.post("/ingest/text", json={
            "title": "Synthetic Graph methods",
            "text": "Synthetic data only. GraphRAG retrieves connected passages.",
            "source": "synthetic:methods",
        }).raise_for_status()

    # No app, retrieval, or generation is needed to inspect the saved corpus.
    page = SQLiteDocumentCatalog(path).list_documents(
        source="synthetic:methods", title="graph", limit=20
    )
    paper = page.documents[0]
    print("Recovered:", paper.document_id, paper.title)
    print("Stored chunks:", paper.chunk_count)
    with TestClient(create_app(settings)) as restarted:
        response = restarted.post("/query", json={
            "query": "GraphRAG connected passages",
            "document_ids": [paper.document_id],
        })
        response.raise_for_status()
        result = response.json()["result"]
        if result["state"] != "DONE":
            raise RuntimeError(result["error"])
        print("State:", result["state"])
PY
```

The explicit demo settings ignore inherited provider keys, `.env`, and ambient
database settings. The fake adapter used by the final query only exercises the
plumbing. The reader itself never needs a provider. For optional live generation,
the [provider guide](PROVIDER_MODELS_GUIDE.md) documents the current chosen
GPT-6 Astra, Claude Sonnet 5, Gemini 3.8 Flash, and Kimi K3 defaults, newer Claude
alternatives, dated official sources, and account/payload limits.

## Bounds, pagination, errors, and privacy

- `limit` defaults to 20 and is constrained to 1-100. The Python API requires an
  actual integer, not a boolean, floating-point number, or numeric string.
- IDs sort ascending using SQLite's binary text order, **not creation time**.
  `cursor` excludes that ID and all earlier IDs; it need not name a current row.
  IDs are 1-128 characters without surrounding whitespace.
- Each page is one consistent SELECT. Separate requests are not one frozen
  snapshot. Re-ingesting the same ID does not move it. New IDs behind your cursor
  will be missed until you restart browsing; updated labels/counts may change.
- Source/title filters accept 1-512 / 1-300 characters respectively. Invalid
  request parameters return HTTP 422. No matches return an empty page with a
  null cursor. Invalid stored labels or unselectable IDs return HTTP 409 with
  `detail.code="invalid_document_record"`, without echoing private stored data.
  The one-row lookahead is validated too, rather than hiding a bad next row.
- Standalone readers open SQLite read-only and do not create missing databases.
  Database/storage errors are not converted into successful empty responses;
  Python callers receive the underlying exception and API failures remain
  server errors.
- Response size and fetched labels are bounded, including Unicode and embedded
  NUL characters. Filtering may still scan the corpus; chunk counts use an index
  on `chunks(document_id)`. No fixed-latency or corpus-size benchmark is claimed.
- Titles and source labels can themselves be sensitive. HTTP summaries carry
  `Cache-Control: no-store` and `X-Content-Type-Options: nosniff`, but this is not
  authentication, tenant isolation, or permission to share a corpus. Keep the
  API on loopback. Existing evidence exports can retain text independently of
  current corpus contents.

## Reproduce the showcase and present a portfolio demonstration

```bash
CATALOG_DIR="$(mktemp -d)/catalog-demo"
uv run python -m scripts.demo_document_catalog --output-dir "$CATALOG_DIR"
uv run python -m scripts.create_document_catalog_gif \
  --transcript "$CATALOG_DIR/transcript.txt" \
  --output "$CATALOG_DIR/document-catalog.gif"
```

Inspect `page-1.json`, `page-2.json`, `selected.json`, `scoped-evidence.json`, and
`transcript.txt`. The demo executes ingestion, pagination, restart, filtering,
and a scoped fake-model query against temporary SQLite, then removes the
database. Both tools refuse to overwrite named outputs. The GIF is a generated
four-panel rendering of that measured transcript, not a recording of a UI.

For a portfolio, demonstrate recovering IDs without the original ingestion
responses, unchanged pagination after restart, zero browsing-generated events,
and a query whose exported context contains only the selected paper. Explain
why titles are still sensitive, IDs do not prove access rights, paging is not
a cross-request snapshot, and lexical grounding is not scientific verification.
