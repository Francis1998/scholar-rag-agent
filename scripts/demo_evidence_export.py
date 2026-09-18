"""Save and inspect real API evidence exports using only synthetic offline data."""

import argparse
import sqlite3
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent.evidence import EvidenceBundle
from api.dependencies import AppContainer
from api.main import app
from config import Settings

DEMO_TEXT = (
    "This is synthetic test data, not a real paper or a scientific finding. "
    "GraphRAG connects entities mentioned in synthetic research passages. "
    "The local retrieval example uses deterministic hashed vectors, BM25, and lexical reranking. "
    "The passage deliberately extends beyond the old 240-character citation snippet so a "
    "reader can inspect the full saved evidence rather than infer missing context. "
    "Unicode survives in the export: caf\u00e9, \u03b2, \u7814\u7a76. "
    "A matching token is not proof of a conclusion."
)
DEMO_QUERY = "How does GraphRAG connect synthetic research evidence?"


def offline_settings(database_path: Path) -> Settings:
    """Disable all live model credentials, including environment and dotenv values."""
    return Settings(
        _env_file=None,
        database_path=database_path,
        default_model="fake",
        OPENAI_API_KEY="",
        ANTHROPIC_API_KEY="",
        GEMINI_API_KEY="",
        MOONSHOT_API_KEY="",
    )


def run_demo(output_dir: Path) -> str:
    """Create inspectable files and prove exports outlive this demo's source corpus."""
    output_dir.mkdir(parents=True, exist_ok=True)
    names = ("demo.sqlite3", "bundle.json", "bundle.md", "transcript.txt")
    for name in names:
        if (output_dir / name).exists():
            raise FileExistsError(
                f"Refusing to overwrite {output_dir / name}; choose a new directory."
            )
    database_path = output_dir / "demo.sqlite3"
    application = FastAPI()
    application.include_router(app.router)
    container = AppContainer(offline_settings(database_path))
    application.state.container = container
    with TestClient(application) as client:
        ingested = client.post(
            "/ingest/text",
            json={
                "title": "Synthetic evidence note",
                "source": "synthetic:demo",
                "text": DEMO_TEXT,
            },
        )
        ingested.raise_for_status()
        query = client.post("/query", json={"query": DEMO_QUERY})
        query.raise_for_status()
        result = query.json()["result"]
        if result["state"] != "DONE":
            raise RuntimeError(f"Synthetic query failed: {result['error']}")
        export_path = f"/runs/{result['run_id']}/export"
        json_response = client.get(export_path, params={"format": "json"})
        markdown_response = client.get(export_path, params={"format": "markdown"})
        json_response.raise_for_status()
        markdown_response.raise_for_status()
        bundle = EvidenceBundle.model_validate_json(json_response.content)
        if (
            bundle.generation.provider != "fake"
            or len(bundle.snapshot.sources) != 1
            or bundle.snapshot.sources[0].chunk.text != DEMO_TEXT
        ):
            raise RuntimeError("The offline demo did not capture its exact synthetic passage.")

        # Only this newly created, synthetic demo database is modified.
        with sqlite3.connect(database_path) as connection:
            connection.execute("DELETE FROM entity_edges")
            connection.execute("DELETE FROM entity_mentions")
            connection.execute("DELETE FROM graph_chunks")
            connection.execute("DELETE FROM chunks")
            connection.execute("DELETE FROM documents")
        restarted = AppContainer(offline_settings(database_path))
        application.state.container = restarted
        for fmt, before in (("json", json_response), ("markdown", markdown_response)):
            after = client.get(export_path, params={"format": fmt})
            after.raise_for_status()
            if before.content != after.content:
                raise RuntimeError(f"{fmt} export changed after source deletion and restart.")
        remaining = len(restarted.document_store.list_chunks())
        if remaining != 0:
            raise RuntimeError("The demo source corpus was not removed.")

    (output_dir / "bundle.json").write_bytes(json_response.content)
    (output_dir / "bundle.md").write_bytes(markdown_response.content)
    transcript = (
        "\n\n".join(
            [
                "\n".join(
                    [
                        "1. Run a synthetic, offline research query",
                        f"POST /ingest/text -> {ingested.status_code}",
                        f"POST /query -> {query.status_code}; state={bundle.status}",
                        f"provider={bundle.generation.provider}; "
                        f"model={bundle.generation.model_name}",
                        f"question: {bundle.query}",
                    ]
                ),
                "\n".join(
                    [
                        "2. Download the saved evidence, not a new answer",
                        f"GET /runs/<run_id>/export?format=json -> {json_response.status_code}",
                        "GET /runs/<run_id>/export?format=markdown -> "
                        f"{markdown_response.status_code}",
                        f"schema_version={bundle.schema_version}; events={len(bundle.events)}",
                        f"full passage={len(bundle.snapshot.sources[0].chunk.text)} characters",
                        "old citation snippet="
                        f"{len(bundle.answer.citations[0].snippet)} characters",
                        f"context SHA-256: {bundle.snapshot.context_sha256}",
                    ]
                ),
                "\n".join(
                    [
                        "3. Inspect associations without mistaking them for proof",
                        f"Claim 1 -> evidence ranks {bundle.claim_evidence[0].evidence_ranks}",
                        f"Missing IDs: {bundle.claim_evidence[0].missing_chunk_ids}",
                        f"Answer ungrounded={bundle.answer.ungrounded}",
                        "The recorded grounded flag checks token overlap, not entailment.",
                        "Only final-context passages are captured; not the entire corpus.",
                    ]
                ),
                "\n".join(
                    [
                        "4. Delete the demo corpus, reopen SQLite, export again",
                        f"Remaining corpus chunks: {remaining}",
                        f"Saved evidence passages: {len(bundle.snapshot.sources)}",
                        "JSON after restart: byte-identical",
                        "Markdown after restart: byte-identical",
                        "Saved: bundle.json, bundle.md, demo.sqlite3, transcript.txt",
                        "Review before sharing. No deterministic LLM replay or signed proof.",
                    ]
                ),
            ]
        )
        + "\n"
    )
    (output_dir / "transcript.txt").write_text(transcript, encoding="utf-8")
    return transcript


def main() -> None:
    """Run with a fresh output directory; no server, API key, or real paper required."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("evidence-demo"))
    arguments = parser.parse_args()
    print(run_demo(arguments.output_dir), end="")


if __name__ == "__main__":
    main()
