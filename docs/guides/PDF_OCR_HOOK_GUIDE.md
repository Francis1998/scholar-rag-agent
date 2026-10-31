# Local PDF ingestion and explicit OCR

Extract a permitted local PDF, inspect its text **before persistence**, ingest a
complete document, and recover its actual saved ID after restart. The executable
walkthrough creates its own synthetic PDF with the already-required `pypdf`;
it downloads no paper, runs no OCR engine, and calls no model provider.

## What actually runs

| Surface | Actual behavior |
| --- | --- |
| `PDFConnector.load(path)` | Uses `pypdf.PdfReader`, joins each page's extracted text with newlines, and returns a `Document`. No OCR, chunking, persistence, or API call. |
| `PdfOcrHook.extract_or_ocr(path, extracted_text)` | Standalone, opt-in Python helper. Returns the original text when long enough; otherwise returns the supplied backend's result. **The PDF connector does not call it.** |
| `IngestionPipeline.ingest_documents([document])` | Chunks, persists, and indexes the complete document, replacing existing evidence with the same document ID. |
| `POST /ingest/text` | Accepts supplied JSON text, not a PDF, filename, or upload. Its identity uses source/title/text, unlike the path-based PDF connector. |
| `scholar-rag-ingest` / `scripts/ingest_papers.py` | Reads a **UTF-8 text file** and prints `Document` JSON. Despite the name, it does not parse PDFs, call the API, or persist/index anything. |

There is no PDF-upload endpoint/UI and no bundled Tesseract or PaddleOCR backend.
Installing optional NLP dependencies or choosing a vision-capable model does not
enable OCR. Provider selection is separate; see the dated
[provider catalog and compatibility guide](PROVIDER_MODELS_GUIDE.md).

## 1. Install and create an isolated workspace

Use Python 3.11+ from the repository root. The [Quickstart](../../QUICKSTART.md)
uses `uv sync --extra dev`; this equivalent setup uses an explicit interpreter:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
export PYTHONPATH="$PWD/src:$PWD"
export PDF_DEMO_DIR="$(mktemp -d)"
printf 'PDF demo workspace: %s\n' "$PDF_DEMO_DIR"
```

Keep that printed absolute path and run the following blocks in the same shell.
Each Python block starts a **new interpreter**. Only this fresh workspace is
used, never the default database or an existing corpus. Fixture files are opened
exclusively: rerunning fixture creation requires another fresh directory.

For your own file, use a small PDF you own or have permission to process. Do not
download copyrighted papers just to reproduce this demo. Local processing does
not grant redistribution rights.

## 2. Generate, extract, and inspect without indexing

This creates one page with real, extractable ASCII text. The fixed strings
contain no PDF string escapes; this is a tiny fixture, not a general PDF writer.

```bash
.venv/bin/python - <<'PY'
import os
from pathlib import Path

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from ingestion.pdf import PDFConnector

directory = Path(os.environ["PDF_DEMO_DIR"]).resolve()
pdf_path = directory / "synthetic_note.pdf"
lines = (
    "Synthetic local PDF, not a publication or scientific result.",
    "GraphRAG connects co-mentioned entities in stored passages.",
    "Retrieval scores are not proof; a human must inspect the source.",
)
writer = PdfWriter()
page = writer.add_blank_page(width=612, height=792)
font = DictionaryObject({
    NameObject("/Type"): NameObject("/Font"),
    NameObject("/Subtype"): NameObject("/Type1"),
    NameObject("/BaseFont"): NameObject("/Helvetica"),
})
page[NameObject("/Resources")] = DictionaryObject({
    NameObject("/Font"): DictionaryObject({NameObject("/F1"): font}),
})
contents = DecodedStreamObject()
contents.set_data(
    ("BT /F1 12 Tf 50 740 Td 18 TL "
     + " T* ".join(f"({line}) Tj" for line in lines)
     + " ET").encode("ascii")
)
page.replace_contents(contents)
with pdf_path.open("xb") as output:
    writer.write(output)
writer.close()

document = PDFConnector().load(pdf_path)
with (directory / "extracted.json").open("x", encoding="utf-8") as output:
    output.write(document.model_dump_json(indent=2) + "\n")
print(document.model_dump_json(indent=2))
print(f"Stripped extracted characters: {len(document.text.strip())}")
print("Extraction only: no corpus database has been opened.")
PY
```

Read the **entire text** in the printed JSON, not just its length. Check page
coverage, reading order, columns, tables, symbols, and whether headers drowned out
the useful content. The connector does not preserve page-number/coordinate
provenance or extract author/year/DOI metadata. Its title comes from the filename
stem with underscores replaced by spaces, not the PDF's title metadata.

For a real permitted PDF, skip fixture generation, set `pdf_path` to its absolute
`Path`, then use the same `PDFConnector().load(pdf_path)` and inspection steps.
The saved extraction JSON is a reviewable snapshot: the next step ingests
**that text**, not a silently reread, potentially changed PDF.

## 3. Persist only the inspected complete document, then preview retrieval

Run this block only after approving the extraction. The 40-character check is
an explicit **tutorial safety policy**, not a validation rule added to the
connector or pipeline, and not proof of good extraction. A legitimate short
document needs a deliberate human decision, not an automatic bypass.

```bash
.venv/bin/python - <<'PY'
import asyncio
import os
from pathlib import Path

from api.dependencies import AppContainer
from config import Settings
from ingestion.pipeline import IngestionPipeline
from retrieval.models import Document
from scripts.demo_evidence_export import offline_settings

directory = Path(os.environ["PDF_DEMO_DIR"]).resolve()
document = Document.model_validate_json(
    (directory / "extracted.json").read_text(encoding="utf-8")
)
if len(document.text.strip()) < 40:
    raise SystemExit(
        "STOP: empty/short extraction; inspect or perform real OCR before persistence."
    )

settings: Settings = offline_settings(directory / "corpus.sqlite3")
container = AppContainer(settings)
pipeline: IngestionPipeline = container.ingestion_pipeline
chunks = pipeline.ingest_documents([document])
preview = asyncio.run(
    container.runner.preview("GraphRAG", document_ids=[document.document_id])
)
assert chunks and preview.sources
assert {item.chunk.document_id for item in preview.sources} == {document.document_id}
assert container.event_log.list_events() == []
print(f"Persisted chunks: {len(chunks)}")
print(preview.model_dump_json(indent=2))
print("Agent events: 0; no answer was generated.")
PY
```

The existing [`offline_settings`](../../scripts/demo_evidence_export.py) helper
constructs a `Settings` subclass using **only explicit arguments and defaults**.
It ignores environment, `.env`, and secret-file settings, selects `fake`, and
clears provider keys. Thus an ambient Anthropic reasoning key or a different
database setting cannot change this demo. `Settings(default_model="fake")` alone
is **not** an equivalent offline guarantee.

`AppContainer` explicitly wires the document store, dense/BM25 indexes, graph
builder, and `IngestionPipeline` to this database. Calling only a low-level
store method is not an equivalent ingestion operation. The runner's `preview`
prepares evidence without calling even the fake generator or recording an agent
run; it does not produce an answer, a `DONE` state, or scientific validation.
See the [retrieval-preview contract](RETRIEVAL_PREVIEW_GUIDE.md).

## 4. Restart, discover saved IDs, and use the HTTP preview contract

This new process creates a new application/container against the **same database**.
Its in-process FastAPI client opens no listener and uses no model credentials.
It discovers the ID through `GET /documents`, not by recomputing a hash, reading
the old extraction response, or hard-coding an ID from someone else's machine.

```bash
.venv/bin/python - <<'PY'
import os
from pathlib import Path

from fastapi.testclient import TestClient

from agent.retrieval_preview import RetrievalPreview
from api.application import create_app
from scripts.demo_evidence_export import offline_settings
from storage.document_catalog import DocumentCatalogPage

directory = Path(os.environ["PDF_DEMO_DIR"]).resolve()
database_path = directory / "corpus.sqlite3"
if not database_path.is_file():
    raise SystemExit("STOP: use the existing PDF demo database, not a new empty path.")
application = create_app(offline_settings(database_path))
with TestClient(application) as client:
    response = client.get("/documents", params={"limit": 20})
    response.raise_for_status()
    catalog = DocumentCatalogPage.model_validate_json(response.content)
    print(catalog.model_dump_json(indent=2))
    if len(catalog.documents) != 1 or catalog.next_cursor is not None:
        raise RuntimeError("This fresh synthetic demo must contain exactly one document.")
    selected = catalog.documents[0]
    response = client.post(
        "/retrieve",
        json={"query": "GraphRAG", "document_ids": [selected.document_id]},
    )
    response.raise_for_status()
    preview = RetrievalPreview.model_validate_json(response.content)
    assert preview.sources
    assert {item.chunk.document_id for item in preview.sources} == {selected.document_id}
    print(f"Recovered document ID: {selected.document_id}")
    print(f"Stored chunks after restart: {selected.chunk_count}")
    print(f"Scoped preview sources after restart: {len(preview.sources)}")
    runs = client.get("/runs")
    runs.raise_for_status()
    assert runs.json()["runs"] == []
    print("Saved agent runs: 0")
PY
```

For a larger corpus, page with `next_cursor` and choose the intended row rather
than blindly selecting the first result. Exact `source` and literal `title`
filters help; catalog labels can be truncated. Follow the
[document catalog](DOCUMENT_CATALOG_GUIDE.md) and
[stored passage reader](DOCUMENT_CHUNKS_GUIDE.md) for selection and full text.

To browse with a real local server, use the Quickstart's explicit offline app
setup with this same database path, keep it on loopback, and visit `/explore`.
Restart an already-running server after external Python ingestion: each
container loads its own in-memory indexes, and a shared SQLite path is not a
cross-process index refresh service. Do not call `/ingest/text` again merely to
make the PDF appear; that creates a different identity.

## 5. Explicit OCR wiring: a stub, not an OCR engine

`PdfOcrHook` checks the **whole document's stripped character count**, with a
default `min_chars=40`. At exactly the threshold it preserves the original
text unchanged. Below it, the backend result **replaces**, rather than appends
to, the extracted text. The default `NullOcrBackend` returns `""`, even when the
original extraction contained a few useful characters. No quality check,
page-level fallback, output validation, or automatic persistence follows.

This optional block creates a **blank page**, not a scanned image. It proves
the call boundary with canned synthetic text; the stub never recognizes pixels.
It deliberately does **not** ingest its fabricated result.

```bash
.venv/bin/python - <<'PY'
import os
from pathlib import Path

from pypdf import PdfWriter

from ingestion.pdf import PDFConnector
from ingestion.pdf_ocr import PdfOcrHook


class StubOcrBackend:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def ocr_pdf(self, path: Path | str) -> str:
        self.calls.append(str(path))
        return "Synthetic stub output only; no image recognition or scientific evidence."


directory = Path(os.environ["PDF_DEMO_DIR"]).resolve()
pdf_path = directory / "blank.pdf"
writer = PdfWriter()
writer.add_blank_page(width=612, height=792)
with pdf_path.open("xb") as output:
    writer.write(output)
writer.close()
document = PDFConnector().load(pdf_path)
assert document.text.strip() == ""
assert PdfOcrHook().extract_or_ocr(pdf_path, document.text) == ""

backend = StubOcrBackend()
hook = PdfOcrHook(backend, min_chars=40)
candidate_text = hook.extract_or_ocr(pdf_path, document.text)
if len(candidate_text.strip()) < hook.min_chars:
    raise SystemExit("STOP: OCR output is still empty/short; do not persist it.")
candidate = document.model_copy(update={
    "text": candidate_text,
    "metadata": {**document.metadata, "ocr_backend": "synthetic-stub"},
})
assert backend.calls == [str(pdf_path)]
print(candidate.model_dump_json(indent=2))
print(f"Explicit stub calls: {len(backend.calls)}")
print("Stub result NOT persisted; the original corpus is unchanged.")
PY
```

For **real** OCR, supply your own `OcrBackend.ocr_pdf(path) -> str` implementation
or use an authorized external OCR tool and review its extracted text. The backend
must own PDF rendering, language selection, page ordering, resource limits, and
errors; this repository supplies none of those engine integrations. An external
service also needs separate permission to receive the document.

Load the PDF first, explicitly call the hook with `document.text`, inspect the
**entire returned text**, and only then copy it into the same complete `Document`
as above. Preserve its ID/source and record the real extraction method in
metadata. Persist that reviewed, genuine result with the Step 3 pipeline, not
the stub result. For mixed text/scanned PDFs, the total text can exceed 40
characters while entire pages are missing: the hook will not detect this.
Use page-aware OCR outside the hook when required. An extraction exception is
also not an automatic trigger for OCR.

## Identity, replacement, and privacy

The connector derives `document_id` from the **resolved absolute file path**,
not the PDF bytes or a DOI. Replacing a file at the same resolved path retains
its ID; moving/copying it to a different resolved path normally produces another
ID. Two path spellings resolving to the same file have the same identity, but
the `source` label is `str(Path(path))`, not the resolved identity path.

**Never ingest a failed, blank, or unreviewed short extraction over a saved
document.** Same-ID ingestion replaces the entire current document. A blank
replacement clears current chunks and graph evidence; it does not mean "skip
this update." Identical complete-document replay is idempotent. The existing
[complete-document reingestion example](../EXAMPLES.md#replace-a-complete-document-in-python)
covers shorter revisions, blank clearing, atomicity boundaries, and historical
evidence; do not implement incremental append behavior by reusing the ID.

The original PDF is not copied into SQLite. Extracted document/chunk text,
metadata, and source labels are persisted; previews and later exports can include
them. Here the source is an absolute local path, which can reveal usernames or
private directory names. Parser errors can also contain the input path.
Hashing that path into an ID does not redact the source label. Use a neutral
staging path **before the first import** when appropriate, and inspect JSON,
logs, screenshots, and exports before sharing. Local storage does not make a
later live `/query` private: a configured model provider can receive selected
text. The demonstration's explicit offline settings avoid that.

## Troubleshooting

| Symptom | Meaning and safe next action |
| --- | --- |
| Empty/short text or the tutorial's `STOP` | No persistence has occurred in Step 3. Check all pages; scanned/image-only files need genuine OCR. Keep existing corpus evidence unchanged while investigating. |
| Enough characters but missing pages or garbled tables | Length is only a heuristic. Inspect reading order and mixed scans; pypdf is not layout reconstruction or scientific validation. |
| `ValueError: failed to parse PDF ...` | The connector wraps parse/open failures. Check the path, permissions, and file integrity. It accepts no password parameter; obtain an authorized decrypted copy for a protected PDF that pypdf cannot open. OCR is not a way around encryption. |
| CLI prints JSON, but the catalog stays empty | The text CLI does not ingest. Use the reviewed `Document` and `IngestionPipeline`; never pass PDF bytes to the UTF-8 CLI. |
| Saved ID differs from the example or another computer | Expected: absolute path participates in identity. Recover real IDs through the catalog. |
| Catalog has data but a running server cannot retrieve it | Confirm the database path and restart that container to reload indexes after external ingestion. Do not erase the database or reimport through another identity scheme. |
| Default OCR hook loses short text | `NullOcrBackend` returned empty text. Do not persist it; choose an actual backend or explicitly keep the reviewed extraction. |

## Reproduce the guide and present it honestly

The [guide regression tests](../../tests/test_pdf_ingestion_guide.py) execute the
four Python blocks in separate interpreters with synthetic ambient keys and
network/generation tripwires. They check real pypdf extraction, inspected
snapshot persistence, restart discovery/scoped preview, no stub persistence,
and refusal of blank/short replacements without changing the saved evidence.

```bash
PYTHONPATH=src:. .venv/bin/python -m pytest \
  tests/test_pdf_ingestion_guide.py tests/test_pdf_ocr.py \
  tests/test_document_reingestion.py tests/test_documentation_navigation.py
```

For a portfolio walkthrough, show the printed extraction, one persisted chunk,
the recovered catalog ID, one scoped preview source after restart, and zero
agent runs. IDs and absolute paths are machine-specific; do not invent or paste
fixed successful output. This demonstrates extraction/persistence/retrieval
plumbing, **not real OCR, an LLM quality comparison, or a scientific finding**.
The old illustrative OCR animation is intentionally not used as execution
evidence. The fresh workspace retains the PDFs, extraction JSON, and SQLite
database for local inspection; remove only that known demo directory when done.
