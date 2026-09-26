"""Measure a model-free research worksheet over an isolated synthetic corpus."""

import argparse
import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi.testclient import TestClient

from agent.research_worksheet import ResearchWorksheet, WorksheetRequest
from agent.retrieval_preview import RetrievalPreview
from api.application import create_app
from api.dependencies import AppContainer
from api.schemas import IngestResponse
from llm.fake import FakeLLMAdapter
from llm.router import RoutingLLMAdapter
from scripts.demo_retrieval_preview import offline_settings

QUESTIONS = ("retrieval", "graph evidence")
PANEL_TITLES = (
    "1. Compare selected papers",
    "2. Inspect retrieved passages",
    "3. Keep the workflow offline",
    "4. Save portable review artifacts",
)
PAPERS = (
    (
        "Synthetic lexical retrieval note",
        "Synthetic fixture, not a publication. Lexical retrieval ranks shared terms. "
        "Repeated terms can raise a lexical score. This note reports no experimental results.",
    ),
    (
        "Synthetic graph retrieval note",
        "Synthetic fixture, not a publication. Graph retrieval follows entity co-mentions. "
        "Shared concepts help discover passages, but these links are not proof of support.",
    ),
    ("Synthetic excluded note", "EXCLUDED_MARKER Unselected synthetic retrieval evidence."),
)


def _ingest(client: TestClient, title: str, text: str) -> str:
    response = client.post(
        "/ingest/text",
        json={"title": title, "text": text, "source": "synthetic:worksheet"},
    )
    response.raise_for_status()
    return IngestResponse.model_validate_json(response.content).document_id


def run_demo(output_dir: Path) -> str:
    """Save measured downloads, checks, and a transcript; never overwrite named artifacts."""
    names = ("worksheet.json", "worksheet.md", "checks.json", "transcript.txt")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        if (output_dir / name).exists():
            raise FileExistsError(
                f"Refusing to overwrite {output_dir / name}; choose a new directory."
            )
    with (
        TemporaryDirectory(prefix="scholar-worksheet-") as temporary,
        patch.object(
            RoutingLLMAdapter, "generate", side_effect=AssertionError("No worksheet generation.")
        ) as routed,
        patch.object(
            FakeLLMAdapter, "generate", side_effect=AssertionError("No fake generation either.")
        ) as fake,
    ):
        database = Path(temporary) / "worksheet.sqlite3"
        settings = offline_settings(database).model_copy(update={"max_source_docs": 1})
        application = create_app(settings)
        container: AppContainer = application.state.container
        with TestClient(application) as client:
            identifiers = [_ingest(client, title, text) for title, text in PAPERS]
            selected = identifiers[:2]
            payload = WorksheetRequest(
                questions=QUESTIONS, document_ids=tuple(selected), passages_per_cell=1
            )
            request_body = payload.model_dump(mode="json", exclude_none=True)
            before = database.read_bytes()
            baseline_response = client.post(
                "/retrieve", json={"query": QUESTIONS[0], "document_ids": selected}
            )
            baseline_response.raise_for_status()
            baseline = RetrievalPreview.model_validate_json(baseline_response.content)
            response = client.post("/research/worksheet", json=request_body)
            response.raise_for_status()
            worksheet = ResearchWorksheet.model_validate_json(response.content)
            markdown_response = client.post(
                "/research/worksheet?format=markdown", json=request_body
            )
            markdown_response.raise_for_status()
            python_result = asyncio.run(container.worksheets.build(payload))
            unknown = client.post(
                "/research/worksheet",
                json={"questions": QUESTIONS, "document_ids": ["not-ingested"]},
            )
            unscoped = client.post("/research/worksheet", json={"questions": QUESTIONS})
            events = container.event_log.list_events()

        with TestClient(create_app(settings)) as client:
            reopened = client.post("/research/worksheet", json=request_body)
            reopened.raise_for_status()
            restarted = ResearchWorksheet.model_validate_json(reopened.content)
        json_text = worksheet.to_json()
        markdown_text = worksheet.to_markdown()
        checks: dict[str, bool | int] = {
            "global_preview_papers": len({source.chunk.document_id for source in baseline.sources}),
            "worksheet_papers": len(
                {
                    passage.document_id
                    for row in worksheet.rows
                    for cell in row.cells
                    for passage in cell.passages
                }
            ),
            "generation_calls": routed.call_count + fake.call_count,
            "agent_events": len(events),
            "python_http_equal": python_result == worksheet,
            "restart_equal": restarted == worksheet,
            "database_unchanged": before == database.read_bytes(),
            "unknown_selection_status": unknown.status_code,
            "unscoped_status": unscoped.status_code,
            "json_bytes": len(json_text.encode("utf-8")),
            "markdown_bytes": len(markdown_text.encode("utf-8")),
        }
        if (
            checks["global_preview_papers"] != 1
            or checks["worksheet_papers"] != 2
            or checks["generation_calls"] != 0
            or checks["agent_events"] != 0
            or unknown.status_code != 422
            or unscoped.status_code != 422
            or "EXCLUDED_MARKER" in json_text
            or markdown_response.text != markdown_text
            or response.text != json_text
        ):
            raise RuntimeError("The measured worksheet or offline isolation contract failed.")
        for name in ("python_http_equal", "restart_equal", "database_unchanged"):
            if not checks[name]:
                raise RuntimeError(f"The measured {name} contract failed.")
    checks["temporary_database_removed"] = not database.exists()
    if not checks["temporary_database_removed"]:
        raise RuntimeError("The synthetic worksheet database was not removed.")
    cell_count = sum(len(row.cells) for row in worksheet.rows)
    transcript = (
        "\n\n".join(
            [
                "\n".join(
                    [
                        PANEL_TITLES[0],
                        f"Questions: {len(worksheet.rows)}; selected papers: "
                        f"{len(worksheet.documents)}; cells: {cell_count}",
                        f"Global top-1 represents {checks['global_preview_papers']}/2 selected "
                        f"papers; worksheet represents {checks['worksheet_papers']}/2.",
                        "Every cell uses the same preview path, scoped to one selected paper.",
                        "Retrieved passages are not answers or scientific findings.",
                    ]
                ),
                "\n".join(
                    [
                        PANEL_TITLES[1],
                        f"Question 1: {worksheet.rows[0].question}",
                        f"Paper 1: {worksheet.rows[0].cells[0].passages[0].title}",
                        f"Paper 2: {worksheet.rows[0].cells[1].passages[0].title}",
                        "One passage per cell; exact IDs, ranks, scores and digests retained.",
                        "EXCLUDED_MARKER in worksheet: " + str("EXCLUDED_MARKER" in json_text),
                        "Scores describe retrieval, not the strength of evidence.",
                    ]
                ),
                "\n".join(
                    [
                        PANEL_TITLES[2],
                        f"Live/fake generation calls: {checks['generation_calls']}; "
                        f"agent events: {checks['agent_events']}",
                        f"Unknown selection: HTTP {unknown.status_code}; "
                        f"missing selection: HTTP {unscoped.status_code}",
                        f"Database unchanged: {checks['database_unchanged']}",
                        "No public listener; ambient provider keys and settings are ignored.",
                    ]
                ),
                "\n".join(
                    [
                        PANEL_TITLES[3],
                        f"Python / HTTP equal: {checks['python_http_equal']}",
                        f"Unchanged-corpus restart equal: {checks['restart_equal']}",
                        f"JSON: {checks['json_bytes']} bytes; "
                        f"Markdown: {checks['markdown_bytes']} bytes",
                        f"Temporary database removed: {checks['temporary_database_removed']}",
                        "Save the downloads for human review; they are not a frozen corpus.",
                    ]
                ),
            ]
        )
        + "\n"
    )
    artifacts = {
        "worksheet.json": json_text,
        "worksheet.md": markdown_text,
        "checks.json": json.dumps(checks, indent=2, sort_keys=True) + "\n",
        "transcript.txt": transcript,
    }
    for name, content in artifacts.items():
        with (output_dir / name).open("x", encoding="utf-8") as destination:
            destination.write(content)
    return transcript


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    print(run_demo(arguments.output_dir), end="")


if __name__ == "__main__":
    main()
