"""Execute the local PDF guide against real synthetic PDFs and isolated storage."""

import json
import os
import re
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import pytest

from ingestion.chunking import stable_id
from retrieval.models import Document

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs/guides/PDF_OCR_HOOK_GUIDE.md"
NO_EXTERNAL_CALLS = """
import httpx
from llm.fake import FakeLLMAdapter
from llm.router import RoutingLLMAdapter

def forbidden(*args, **kwargs):
    raise AssertionError("The PDF guide must not use network or generation.")

httpx.HTTPTransport.handle_request = forbidden
httpx.AsyncHTTPTransport.handle_async_request = forbidden
FakeLLMAdapter.generate = forbidden
RoutingLLMAdapter.generate = forbidden
"""


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / ".env").write_text(
        "SCHOLAR_RAG_DEFAULT_MODEL=invalid-provider\n"
        "SCHOLAR_RAG_MAX_SOURCE_DOCS=invalid-count\n"
        "ANTHROPIC_API_KEY=unused-synthetic-dotenv-key\n",
        encoding="utf-8",
    )
    return tmp_path


def run_example(index: int, directory: Path) -> subprocess.CompletedProcess[str]:
    examples = re.findall(
        r"^\.venv/bin/python - <<'PY'\n(.*?)\nPY$",
        GUIDE.read_text(encoding="utf-8"),
        flags=re.DOTALL | re.MULTILINE,
    )
    assert len(examples) == 4
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(ROOT / "src"), str(ROOT))),
        "PDF_DEMO_DIR": str(directory),
        "SCHOLAR_RAG_DATABASE_PATH": str(directory / "must-not-use-ambient.sqlite3"),
        "SCHOLAR_RAG_MAX_HOPS": "invalid-ambient-count",
        **dict.fromkeys(
            (
                "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY",
                "GEMINI_API_KEY",
                "MOONSHOT_API_KEY",
                "SEMANTIC_SCHOLAR_API_KEY",
            ),
            "unused-synthetic-key",
        ),
    }
    guard = NO_EXTERNAL_CALLS
    if index != 3:
        guard += (
            "\nfrom ingestion.pdf_ocr import PdfOcrHook\nPdfOcrHook.extract_or_ocr = forbidden\n"
        )
    result = subprocess.run(
        [sys.executable, "-"],
        input=guard + examples[index],
        cwd=directory,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=45,
    )
    assert not (directory / "must-not-use-ambient.sqlite3").exists()
    assert "unused-synthetic" not in result.stdout
    return result


def database_dump(path: Path) -> str:
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as connection:
        return "\n".join(connection.iterdump())


def test_guide_extracts_persists_recovers_and_keeps_stub_out_of_corpus(workspace: Path) -> None:
    extracted = run_example(0, workspace)
    assert extracted.returncode == 0, extracted.stderr
    assert not (workspace / "corpus.sqlite3").exists()
    document = Document.model_validate_json((workspace / "extracted.json").read_text())
    assert document.document_id == stable_id(str(workspace / "synthetic_note.pdf"), "doc")
    assert document.title == "synthetic note"
    assert document.source == str(workspace / "synthetic_note.pdf")
    assert document.metadata == {"source_type": "pdf"}
    assert document.text.splitlines() == [
        "Synthetic local PDF, not a publication or scientific result.",
        "GraphRAG connects co-mentioned entities in stored passages.",
        "Retrieval scores are not proof; a human must inspect the source.",
    ]
    (workspace / "synthetic_note.pdf").write_bytes(b"Changed after the extraction snapshot.")

    persisted = run_example(1, workspace)
    assert persisted.returncode == 0, persisted.stderr
    assert "Persisted chunks: 1" in persisted.stdout
    assert "Agent events: 0" in persisted.stdout
    before = database_dump(workspace / "corpus.sqlite3")

    restarted = run_example(2, workspace)
    assert restarted.returncode == 0, restarted.stderr
    assert f"Recovered document ID: {document.document_id}" in restarted.stdout
    assert "Stored chunks after restart: 1" in restarted.stdout
    assert "Scoped preview sources after restart: 1" in restarted.stdout
    assert "Saved agent runs: 0" in restarted.stdout

    stub = run_example(3, workspace)
    assert stub.returncode == 0, stub.stderr
    assert "Explicit stub calls: 1" in stub.stdout
    assert "Stub result NOT persisted" in stub.stdout
    assert database_dump(workspace / "corpus.sqlite3") == before


@pytest.mark.parametrize("text", ["", " \t\n", "\u00a0\u2003", "A short extraction."])
@pytest.mark.parametrize("existing", [False, True], ids=["new-corpus", "saved-evidence"])
def test_guide_stops_short_output_before_any_corpus_change(
    workspace: Path, text: str, existing: bool
) -> None:
    extracted = run_example(0, workspace)
    assert extracted.returncode == 0, extracted.stderr
    database = workspace / "corpus.sqlite3"
    if existing:
        persisted = run_example(1, workspace)
        assert persisted.returncode == 0, persisted.stderr
    before = database_dump(database) if existing else None
    snapshot = workspace / "extracted.json"
    document = json.loads(snapshot.read_text(encoding="utf-8"))
    document["text"] = text
    snapshot.write_text(json.dumps(document), encoding="utf-8")

    refused = run_example(1, workspace)
    assert refused.returncode != 0
    assert "STOP: empty/short extraction" in refused.stderr
    if existing:
        assert database_dump(database) == before
    else:
        assert not database.exists()
