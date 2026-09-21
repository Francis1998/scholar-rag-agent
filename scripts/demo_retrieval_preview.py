"""Measure generation-free retrieval previews over an isolated synthetic corpus."""

import argparse
import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi.testclient import TestClient
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource

from agent.retrieval_preview import RetrievalPreview
from api.application import create_app
from api.dependencies import AppContainer
from api.schemas import IngestResponse
from config import Settings
from llm.fake import FakeLLMAdapter
from llm.router import RoutingLLMAdapter
from retrieval.hyde import HyDEExpander

QUERY = "retrieval"
PANEL_TITLES = (
    "1. Inspect evidence without generating an answer",
    "2. Keep selection through hybrid and graph retrieval",
    "3. Distinguish empty matches from invalid requests",
    "4. Compare Python, HTTP, and a reopened corpus",
)


class _OfflineSettings(Settings):
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (init_settings,)


def offline_settings(database_path: Path) -> Settings:
    """Validate only explicit values/defaults, never ambient .env, keys, or secret files."""
    return _OfflineSettings.model_validate(
        {
            "database_path": database_path,
            "default_model": "fake",
            "max_source_docs": 2,
            "max_hops": 3,
            "OPENAI_API_KEY": "",
            "ANTHROPIC_API_KEY": "",
            "GEMINI_API_KEY": "",
            "MOONSHOT_API_KEY": "",
            "SEMANTIC_SCHOLAR_API_KEY": "",
        }
    )


def _ingest(client: TestClient, title: str, text: str) -> IngestResponse:
    response = client.post(
        "/ingest/text",
        json={"title": title, "text": text, "source": "synthetic:retrieval-preview"},
    )
    response.raise_for_status()
    return IngestResponse.model_validate_json(response.content)


def _preview(
    client: TestClient, document_ids: list[str] | None = None, *, query: str = QUERY
) -> RetrievalPreview:
    payload: dict[str, object] = {"query": query}
    if document_ids is not None:
        payload["document_ids"] = document_ids
    response = client.post("/retrieve", json=payload)
    response.raise_for_status()
    return RetrievalPreview.model_validate_json(response.content)


def _documents(preview: RetrievalPreview) -> set[str]:
    return {source.chunk.document_id for source in preview.sources}


def run_demo(output_dir: Path) -> str:
    """Save measured API/Python results with zero generator calls and zero agent events."""
    names = ("unscoped.json", "scoped.json", "graph.json", "unknown.json", "transcript.txt")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        if (output_dir / name).exists():
            raise FileExistsError(
                f"Refusing to overwrite {output_dir / name}; choose a new directory."
            )
    with (
        TemporaryDirectory(prefix="scholar-retrieval-preview-") as temporary,
        patch.object(
            RoutingLLMAdapter,
            "generate",
            side_effect=AssertionError("A preview must never generate."),
        ) as routed,
        patch.object(
            FakeLLMAdapter,
            "generate",
            side_effect=AssertionError("A preview must never call even the fake generator."),
        ) as fake,
    ):
        settings = offline_settings(Path(temporary) / "preview.sqlite3")
        application = create_app(settings)
        container: AppContainer = application.state.container
        with TestClient(application) as client:
            expanded = asyncio.run(HyDEExpander().expand(QUERY))
            papers = [
                _ingest(client, f"Synthetic distractor {index}", expanded) for index in range(4)
            ]
            first = _ingest(
                client,
                "Synthetic selected methods",
                "Alpha selected retrieval. " + "unrelated " * 50,
            )
            second = _ingest(
                client,
                "Synthetic selected followup",
                "Beta Gamma followup retrieval. " + "unrelated " * 50,
            )
            bridge = _ingest(
                client, "Synthetic excluded bridge", "Alpha Beta EXCLUDED_MARKER retrieval."
            )
            papers.extend([first, second, bridge])
            selected = [first.document_id, second.document_id]
            baseline = _preview(client)
            scoped = _preview(client, [f" {selected[0]} ", selected[1], selected[0]])
            graph = _preview(client, selected, query="compare Alpha")
            unknown = _preview(client, ["not-ingested"])
            empty = client.post("/retrieve", json={"query": QUERY, "document_ids": []})
            null = client.post("/retrieve", json={"query": QUERY, "document_ids": None})
            if _documents(baseline) & set(selected) or _documents(scoped) != set(selected):
                raise RuntimeError(
                    "Expected selected papers below global top-2, recovered by scope."
                )
            if (
                _documents(graph) != set(selected)
                or {source.path[-1] for source in graph.sources} != {"rrf", "multihop"}
                or "EXCLUDED_MARKER" in graph.context
            ):
                raise RuntimeError("Hybrid and graph retrieval must both respect the selection.")
            if (
                unknown.sources
                or unknown.context
                or empty.status_code != 422
                or null.status_code != 422
            ):
                raise RuntimeError("Empty matches and invalid scope must have distinct responses.")
            python_preview = asyncio.run(container.runner.preview(QUERY, document_ids=selected))
            python_equal = python_preview == scoped
            events = container.event_log.list_events()
            if events or not python_equal:
                raise RuntimeError("Python/HTTP parity or the no-event-write contract failed.")

        reopened = create_app(settings)
        with TestClient(reopened) as client:
            reopened_preview = _preview(client, selected)
            reopened_equal = reopened_preview == scoped
            events = reopened.state.container.event_log.list_events()
            if not reopened_equal or events or client.get("/runs").json()["runs"]:
                raise RuntimeError("Unchanged corpus previews must survive restart without runs.")
        generation_calls = routed.call_count + fake.call_count
        if generation_calls:
            raise RuntimeError("The demo called a generator.")

    for name, preview in (
        ("unscoped.json", baseline),
        ("scoped.json", scoped),
        ("graph.json", graph),
        ("unknown.json", unknown),
    ):
        (output_dir / name).write_text(preview.model_dump_json(indent=2) + "\n", encoding="utf-8")
    transcript = (
        "\n\n".join(
            [
                "\n".join(
                    [
                        PANEL_TITLES[0],
                        f"Synthetic papers: {len(papers)}; selected papers: {len(selected)}",
                        "Global top-2 selected papers: "
                        f"{len(_documents(baseline) & set(selected))}; "
                        f"scoped top-2: {len(_documents(scoped))}",
                        f"Live/fake generation calls: {generation_calls}",
                        f"Persisted agent events: {len(events)}",
                        "No public listener, answer, claim verdict, or DONE state.",
                    ]
                ),
                "\n".join(
                    [
                        PANEL_TITLES[1],
                        f"Comparison tasks: {len(graph.plan.tasks)}; "
                        f"final chunks: {len(graph.sources)}",
                        "Winning retrieval paths: "
                        + ", ".join(sorted({s.path[-1] for s in graph.sources})),
                        "Excluded bridge text in context: "
                        + str("EXCLUDED_MARKER" in graph.context),
                        f"Effective source cap: {graph.configuration.max_source_docs}; "
                        f"hop cap: {graph.configuration.max_hops}",
                        "Scores describe lexical/co-mention retrieval, not scientific support.",
                    ]
                ),
                "\n".join(
                    [
                        PANEL_TITLES[2],
                        f"Unknown ID retained: {list(unknown.plan.observation.document_ids or ())}",
                        f"Unknown ID -> HTTP 200; sources: {len(unknown.sources)}; "
                        f"context bytes: {len(unknown.context.encode('utf-8'))}",
                        f"Empty scope: HTTP {empty.status_code}; "
                        f"null scope: HTTP {null.status_code}",
                        "No fallback to the full corpus; failures are not empty successes.",
                    ]
                ),
                "\n".join(
                    [
                        PANEL_TITLES[3],
                        f"Python / HTTP evidence identical: {python_equal}",
                        f"Reopened corpus preview identical: {reopened_equal}",
                        f"Context SHA-256: {scoped.context_sha256}",
                        "Saved exact chunks, ranks, scores, paths, scope, and limits as JSON.",
                        "Temporary SQLite removed; changed corpora need not match this preview.",
                    ]
                ),
            ]
        )
        + "\n"
    )
    (output_dir / "transcript.txt").write_text(transcript, encoding="utf-8")
    return transcript


def main() -> None:
    """Execute the offline demonstration without opening a public network listener."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("retrieval-preview-demo"))
    arguments = parser.parse_args()
    print(run_demo(arguments.output_dir), end="")


if __name__ == "__main__":
    main()
