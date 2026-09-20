"""Discover a synthetic corpus after restart, then use a recovered document ID."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

from agent.evidence import EvidenceBundle
from api.application import create_app
from api.dependencies import AppContainer
from scripts.demo_evidence_export import offline_settings
from storage.document_catalog import DocumentCatalogPage

PANEL_TITLES = (
    "1. Ingest a synthetic corpus, then forget its IDs",
    "2. Browse the persisted corpus without generation",
    "3. Recover a selected paper after restart",
    "4. Use the recovered ID in an inspectable query",
)


def run_demo(output_dir: Path) -> str:
    """Save measured API responses; never read ambient credentials or a user's corpus."""
    output_dir.mkdir(parents=True, exist_ok=True)
    names = (
        "page-1.json",
        "page-2.json",
        "selected.json",
        "scoped-evidence.json",
        "transcript.txt",
    )
    for name in names:
        if (output_dir / name).exists():
            raise FileExistsError(
                f"Refusing to overwrite {output_dir / name}; choose a new directory."
            )
    with TemporaryDirectory(prefix="scholar-document-catalog-") as temporary:
        settings = offline_settings(Path(temporary) / "corpus.sqlite3")
        application = create_app(settings)
        with TestClient(application) as client:
            for title, source, text in (
                (
                    "Synthetic Graph methods",
                    "synthetic:methods",
                    "Synthetic data, not a publication. GraphRAG retrieves connected passages.",
                ),
                (
                    "Synthetic BM25 baseline",
                    "synthetic:methods",
                    "Synthetic data, not a publication. BM25 ranks matching lexical terms.",
                ),
                (
                    "Synthetic background",
                    "synthetic:background",
                    "Synthetic data only. Retrieved evidence still requires human review.",
                ),
            ):
                client.post(
                    "/ingest/text", json={"title": title, "source": source, "text": text}
                ).raise_for_status()
            first_response = client.get("/documents", params={"limit": 2})
            first_response.raise_for_status()
            first = DocumentCatalogPage.model_validate_json(first_response.content)
            if first.next_cursor is None:
                raise RuntimeError("The synthetic catalog must have a second page.")
            second_response = client.get(
                "/documents", params={"limit": 2, "cursor": first.next_cursor}
            )
            second_response.raise_for_status()
            second = DocumentCatalogPage.model_validate_json(second_response.content)
            identifiers = [d.document_id for d in [*first.documents, *second.documents]]
            if len(identifiers) != 3 or len(set(identifiers)) != 3 or second.next_cursor:
                raise RuntimeError("Catalog pagination lost or repeated a synthetic document.")
            container: AppContainer = application.state.container
            if container.event_log.list_events():
                raise RuntimeError("Browsing must not create an agent run.")

            application.state.container = AppContainer(settings)
            reloaded = client.get("/documents", params={"limit": 2})
            reloaded.raise_for_status()
            if reloaded.content != first_response.content:
                raise RuntimeError("The first catalog page changed after reopening SQLite.")
            selected_response = client.get(
                "/documents", params={"source": "synthetic:methods", "title": "graph"}
            )
            selected_response.raise_for_status()
            selected = DocumentCatalogPage.model_validate_json(selected_response.content)
            if len(selected.documents) != 1 or selected.next_cursor:
                raise RuntimeError("The synthetic filter must select exactly one paper.")
            chosen = selected.documents[0]
            query = client.post(
                "/query",
                json={"query": "GraphRAG connected passages", "document_ids": [chosen.document_id]},
            )
            query.raise_for_status()
            result = query.json()["result"]
            if result["state"] != "DONE":
                raise RuntimeError(f"Synthetic scoped query failed: {result['error']}")
            evidence_response = client.get(f"/runs/{result['run_id']}/export")
            evidence_response.raise_for_status()
            evidence = EvidenceBundle.model_validate_json(evidence_response.content)
            if evidence.generation.provider != "fake" or {
                item.chunk.document_id for item in evidence.snapshot.sources
            } != {chosen.document_id}:
                raise RuntimeError("The recovered ID did not select the expected offline evidence.")

    artifacts = (first_response, second_response, selected_response, evidence_response)
    for name, response in zip(names[:-1], artifacts, strict=True):
        (output_dir / name).write_bytes(response.content)
    transcript = (
        "\n\n".join(
            [
                "\n".join(
                    [
                        PANEL_TITLES[0],
                        f"Ingested synthetic papers: {len(identifiers)}",
                        "Original ingestion responses are not needed for discovery.",
                        "Database: isolated temporary SQLite; public listener: none",
                    ]
                ),
                "\n".join(
                    [
                        PANEL_TITLES[1],
                        f"Page 1: {len(first.documents)} papers; "
                        f"page 2: {len(second.documents)} paper",
                        f"Unique discovered IDs: {len(set(identifiers))}; repeated IDs: 0",
                        "Returned: IDs, bounded titles/sources, stored chunk counts",
                        "Agent events created by browsing: 0",
                        "Document bodies and arbitrary metadata are not returned.",
                    ]
                ),
                "\n".join(
                    [
                        PANEL_TITLES[2],
                        "First page after reopening SQLite: byte-identical",
                        "Filter: source=synthetic:methods; title contains 'graph'",
                        f"Selected papers: {len(selected.documents)}",
                        f"Title: {chosen.title}; stored chunks: {chosen.chunk_count}",
                        f"Recovered document ID: {chosen.document_id}",
                    ]
                ),
                "\n".join(
                    [
                        PANEL_TITLES[3],
                        f"Scoped query: {evidence.status}; "
                        f"provider: {evidence.generation.provider}",
                        f"Selected evidence passages: {len(evidence.snapshot.sources)}",
                        "Unselected documents in evidence: 0",
                        "Saved: two pages, selection, scoped evidence, transcript",
                        "Fake output demonstrates plumbing, not scientific findings.",
                        "Temporary SQLite removed; review artifacts before sharing.",
                    ]
                ),
            ]
        )
        + "\n"
    )
    (output_dir / "transcript.txt").write_text(transcript, encoding="utf-8")
    return transcript


def main() -> None:
    """Run the synthetic API demonstration without a public listener."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("document-catalog-demo"))
    arguments = parser.parse_args()
    print(run_demo(arguments.output_dir), end="")


if __name__ == "__main__":
    main()
