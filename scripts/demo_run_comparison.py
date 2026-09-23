"""Compare two actual offline synthetic runs and retain only reviewable output artifacts."""

import argparse
import sqlite3
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

from agent.comparison_models import RunComparison
from agent.evidence import EvidenceBundle
from agent.models import AgentState
from api.application import create_app
from api.dependencies import AppContainer
from api.schemas import IngestResponse, QueryResponse
from scripts.demo_evidence_export import offline_settings

PANEL_TITLES = (
    "1. Save two synthetic offline research runs",
    "2. Compare frozen evidence, not new answers",
    "3. Review exact changes, not quality scores",
    "4. Delete the corpus and reopen the database",
)


def run_demo(output_dir: Path) -> str:
    """Use real API routes; remove the temporary database and refuse artifact collisions."""
    names = ("baseline.json", "candidate.json", "comparison.json", "transcript.txt")
    for name in names:
        if (output_dir / name).exists():
            raise FileExistsError(
                f"Refusing to overwrite {output_dir / name}; choose a new directory."
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="scholar-run-comparison-") as temporary:
        path = Path(temporary) / "demo.sqlite3"
        application = create_app(offline_settings(path))
        container: AppContainer = application.state.container
        with TestClient(application) as client:
            documents = []
            for label in ("baseline-only", "shared", "candidate-only"):
                response = client.post(
                    "/ingest/text",
                    json={
                        "title": f"Synthetic {label} note",
                        "source": f"synthetic:comparison:{label}",
                        "text": (
                            f"Synthetic {label} evidence, not a scientific finding. "
                            "GraphRAG connects synthetic research evidence. "
                            f"This {label} fixture demonstrates saved passage inspection."
                        ),
                    },
                )
                response.raise_for_status()
                documents.append(IngestResponse.model_validate_json(response.content))
            runs = []
            for query, selected in (
                ("What does GraphRAG connect?", documents[:2]),
                ("What synthetic evidence does GraphRAG connect?", documents[1:]),
            ):
                response = client.post(
                    "/query",
                    json={
                        "query": query,
                        "document_ids": [document.document_id for document in selected],
                    },
                )
                response.raise_for_status()
                run = QueryResponse.model_validate_json(response.content).result
                if run.state != AgentState.DONE:
                    raise RuntimeError(f"Synthetic query did not complete: {run.error}")
                runs.append(run)
            baseline_response = client.get(f"/runs/{runs[0].run_id}/export")
            candidate_response = client.get(f"/runs/{runs[1].run_id}/export")
            baseline_response.raise_for_status()
            candidate_response.raise_for_status()
            bundles = [
                EvidenceBundle.model_validate_json(response.content)
                for response in (baseline_response, candidate_response)
            ]
            if any(bundle.generation.provider != "fake" for bundle in bundles):
                raise RuntimeError("The offline demo did not use the fake provider.")
            before_events = container.event_log.list_events()
            comparison_path = f"/runs/{runs[0].run_id}/compare/{runs[1].run_id}"
            comparison_response = client.get(comparison_path)
            comparison_response.raise_for_status()
            result = RunComparison.model_validate_json(comparison_response.content)
            counts = result.evidence.statistics
            if (
                counts.baseline_count != 2
                or counts.candidate_count != 2
                or counts.shared_count != 1
                or counts.added_count != 1
                or counts.removed_count != 1
                or not result.changes.query_changed
                or not result.changes.document_scope_changed
                or not result.changes.answer_text_changed
            ):
                raise RuntimeError("The real comparison did not match the synthetic run inputs.")
            self_response = client.get(f"/runs/{runs[0].run_id}/compare/{runs[0].run_id}")
            self_response.raise_for_status()
            unchanged = RunComparison.model_validate_json(self_response.content)
            if unchanged.any_changes:
                raise RuntimeError("Comparing a saved run to itself reported changes.")
            with closing(sqlite3.connect(path)) as connection:
                connection.execute("DELETE FROM entity_edges")
                connection.execute("DELETE FROM entity_mentions")
                connection.execute("DELETE FROM graph_chunks")
                connection.execute("DELETE FROM chunks")
                connection.execute("DELETE FROM documents")
                connection.commit()
            restarted = AppContainer(offline_settings(path))
            application.state.container = restarted
            remaining = len(restarted.document_store.list_chunks())
            after = client.get(comparison_path)
            after.raise_for_status()
            if remaining or after.content != comparison_response.content:
                raise RuntimeError("The comparison changed after corpus deletion and restart.")
            after_events = restarted.event_log.list_events()
            if after_events != before_events:
                raise RuntimeError("Comparison or recreation modified saved events.")
            event_writes = len(after_events) - len(before_events)
    database_removed = not path.exists()
    if not database_removed:
        raise RuntimeError("The temporary demo database was not removed.")
    for name, response in (
        ("baseline.json", baseline_response),
        ("candidate.json", candidate_response),
        ("comparison.json", comparison_response),
    ):
        (output_dir / name).write_bytes(response.content)
    transcript = (
        "\n\n".join(
            [
                "\n".join(
                    [
                        PANEL_TITLES[0],
                        f"POST /ingest/text -> 200; synthetic documents={len(documents)}",
                        f"POST /query -> 200; completed runs={len(runs)}",
                        "Saved providers: "
                        + ", ".join(bundle.generation.provider for bundle in bundles),
                        f"Baseline scope: {len(result.baseline.document_ids or ())} documents",
                        f"Candidate scope: {len(result.candidate.document_ids or ())} documents",
                        "Fake output is a plumbing example, not a scientific finding.",
                    ]
                ),
                "\n".join(
                    [
                        PANEL_TITLES[1],
                        "GET /runs/<baseline>/compare/<candidate> -> "
                        f"{comparison_response.status_code}",
                        f"schema_version={result.schema_version}; any_changes={result.any_changes}",
                        f"Saved chunks: baseline={counts.baseline_count}, "
                        f"candidate={counts.candidate_count}",
                        f"Identity counts: shared={counts.shared_count}, "
                        f"added={counts.added_count}, removed={counts.removed_count}",
                        "Identity means (chunk ID, document ID), not a support measurement.",
                        f"Same-run comparison: any_changes={unchanged.any_changes}",
                    ]
                ),
                "\n".join(
                    [
                        PANEL_TITLES[2],
                        f"Exact query changed: {result.changes.query_changed}",
                        f"Document scope changed: {result.changes.document_scope_changed}",
                        f"Answer text changed: {result.changes.answer_text_changed}",
                        f"Generation context changed: {result.changes.context_changed}",
                        "Different inputs/scopes are not a fair model A/B test.",
                        "Follow the two exports for full passages, citations, and warnings.",
                    ]
                ),
                "\n".join(
                    [
                        PANEL_TITLES[3],
                        f"Remaining corpus chunks: {remaining}",
                        "Comparison after corpus deletion/restart: byte-identical",
                        f"Comparison event writes: {event_writes}",
                        f"Temporary demo database removed: {database_removed}",
                        "Saved: baseline.json, candidate.json, comparison.json, transcript.txt",
                        "Synthetic/offline inspection only; no live model or research UI.",
                    ]
                ),
            ]
        )
        + "\n"
    )
    (output_dir / "transcript.txt").write_text(transcript, encoding="utf-8")
    return transcript


def main() -> None:
    """Require an explicit output directory; never load ambient deployment settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    print(run_demo(arguments.output_dir), end="")


if __name__ == "__main__":
    main()
