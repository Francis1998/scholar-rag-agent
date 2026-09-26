"""Measure request-level document quotas through the real, explicitly offline API."""

import argparse
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient
from pydantic import BaseModel

from agent.comparison_models import RunComparison
from agent.evidence import EvidenceBundle, EvidenceSource
from agent.retrieval_preview import RetrievalPreview
from api.application import create_app
from api.dependencies import AppContainer
from api.schemas import QueryResponse
from retrieval.models import Chunk, Document
from scripts.demo_evidence_export import offline_settings

QUERY = "retrieval evidence"
PANEL_TITLES = (
    "1. Compare the same bounded candidate pool",
    "2. Combine a quota with a saved selection",
    "3. Generate after inspecting the exact evidence",
    "4. Preserve policies and saved evidence",
)
OUTPUT_NAMES = (
    "baseline.json",
    "limited.json",
    "collection.json",
    "unknown.json",
    "bundle.json",
    "bundle.md",
    "comparison.json",
    "transcript.txt",
)


def seed_corpus(container: AppContainer) -> None:
    """Index six explicitly chunked synthetic passages with identical titles/source labels."""
    documents, chunks = [], []
    for document_id, count in (("paper-a", 3), ("paper-b", 2), ("paper-c", 1)):
        passages = [
            Chunk(
                chunk_id=f"{document_id}-{index}",
                document_id=document_id,
                title="Identical synthetic title",
                text=f"GraphRAG retrieval evidence {index} for {document_id}; synthetic only.",
                source="synthetic:per-paper-demo",
                metadata={"license": "synthetic-only"},
            )
            for index in range(count)
        ]
        documents.append(
            Document(
                document_id=document_id,
                title=passages[0].title,
                text="\n".join(chunk.text for chunk in passages),
                source=passages[0].source,
            )
        )
        chunks.extend(passages)
    container.document_store.add_documents(documents, chunks)
    container.hybrid_retriever.add_chunks(chunks)
    container.graph_builder.index_chunks(chunks)


def _preview(client: TestClient, **options: object) -> RetrievalPreview:
    response = client.post("/retrieve", json={"query": QUERY, **options})
    response.raise_for_status()
    return RetrievalPreview.model_validate_json(response.content)


def _query(client: TestClient, **options: object) -> str:
    response = client.post("/query", json={"query": QUERY, **options})
    response.raise_for_status()
    result = QueryResponse.model_validate_json(response.content).result
    if result.state != "DONE":
        raise RuntimeError(f"Synthetic query failed: {result.error}")
    return result.run_id


def _counts(sources: list[EvidenceSource]) -> dict[str, int]:
    return dict(sorted(Counter(source.chunk.document_id for source in sources).items()))


def _count_text(sources: list[EvidenceSource]) -> str:
    return ", ".join(f"{document_id}={count}" for document_id, count in _counts(sources).items())


def run_demo(output_dir: Path) -> str:
    """Save actual API outputs; never use ambient credentials or overwrite caller artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_NAMES:
        path = output_dir / name
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite {path}; choose a new directory.")
    with (
        TemporaryDirectory(prefix="scholar-per-paper-") as temporary,
        patch.object(
            httpx.HTTPTransport,
            "handle_request",
            side_effect=AssertionError("The synthetic demo must not use the network."),
        ) as sync_http,
        patch.object(
            httpx.AsyncHTTPTransport,
            "handle_async_request",
            side_effect=AssertionError("The synthetic demo must not use the network."),
        ) as async_http,
    ):
        settings = offline_settings(Path(temporary) / "demo.sqlite3").model_copy(
            update={"max_source_docs": 6}
        )
        application = create_app(settings)
        container: AppContainer = application.state.container
        seed_corpus(container)
        executor = container.runner._executor
        with (
            TestClient(application) as client,
            patch.object(
                container.hybrid_retriever, "retrieve", wraps=container.hybrid_retriever.retrieve
            ) as hybrid,
            patch.object(
                executor._multihop_retriever,
                "retrieve",
                wraps=executor._multihop_retriever.retrieve,
            ) as graph,
            patch.object(container.llm, "generate", wraps=container.llm.generate) as generate,
            patch.object(
                container.event_log, "append_event", wraps=container.event_log.append_event
            ) as event_writes,
        ):
            baseline = _preview(client)
            baseline_calls = (hybrid.await_count, graph.await_count)
            baseline_arguments = (list(hybrid.await_args_list), list(graph.await_args_list))
            hybrid.reset_mock()
            graph.reset_mock()
            limited = _preview(client, max_chunks_per_document=1)
            capped_calls = (hybrid.await_count, graph.await_count)
            if (
                _counts(baseline.sources) != {"paper-a": 3, "paper-b": 2, "paper-c": 1}
                or _counts(limited.sources) != {"paper-a": 1, "paper-b": 1, "paper-c": 1}
                or baseline_arguments != (list(hybrid.await_args_list), list(graph.await_args_list))
            ):
                raise RuntimeError("The quota must only filter the same bounded candidate pool.")
            saved = client.post(
                "/collections",
                json={"name": "Synthetic selection", "document_ids": ["paper-a", "paper-b"]},
            )
            saved.raise_for_status()
            collection = _preview(
                client, collection_id=saved.json()["collection_id"], max_chunks_per_document=1
            )
            unknown = _preview(client, document_ids=["not-ingested"], max_chunks_per_document=1)
            null = client.post("/retrieve", json={"query": QUERY, "max_chunks_per_document": None})
            boolean = client.post("/query", json={"query": QUERY, "max_chunks_per_document": True})
            preview_generations, preview_writes = generate.await_count, event_writes.call_count
            if (
                _counts(collection.sources) != {"paper-a": 1, "paper-b": 1}
                or unknown.sources
                or unknown.plan.observation.document_ids != ("not-ingested",)
                or null.status_code != 422
                or boolean.status_code != 422
                or preview_generations
                or preview_writes
                or container.event_log.list_events()
            ):
                raise RuntimeError(
                    "Preview validation, scope, or no-generation/no-event contract failed."
                )

            run_id = _query(client, max_chunks_per_document=1)
            export_path = f"/runs/{run_id}/export"
            json_response = client.get(export_path)
            markdown_response = client.get(export_path, params={"format": "markdown"})
            json_response.raise_for_status()
            markdown_response.raise_for_status()
            bundle = EvidenceBundle.model_validate_json(json_response.content)
            policy = bundle.plan.observation.evidence_policy
            same_context = (
                bundle.snapshot.sources == limited.sources
                and bundle.snapshot.request.context == limited.context
                and bundle.snapshot.context_sha256 == limited.context_sha256
            )
            if (
                not same_context
                or policy is None
                or policy.max_chunks_per_document != 1
                or policy != limited.plan.observation.evidence_policy
                or bundle.generation.provider != "fake"
                or "max_chunks_per_document=1" not in markdown_response.text
            ):
                raise RuntimeError(
                    "Saved query evidence must match the preview and requested policy."
                )
            first = _query(client, document_ids=["paper-c"], max_chunks_per_document=1)
            second = _query(client, document_ids=["paper-c"], max_chunks_per_document=2)
            comparison_path = f"/runs/{first}/compare/{second}"
            comparison_response = client.get(comparison_path)
            comparison_response.raise_for_status()
            comparison = RunComparison.model_validate_json(comparison_response.content)
            if [key for key, changed in comparison.changes.model_dump().items() if changed] != [
                "evidence_policy_changed"
            ] or not comparison.any_changes:
                raise RuntimeError("Equal evidence must not hide different requested quotas.")
            query_generations = generate.await_count - preview_generations
            if query_generations != 3:
                raise RuntimeError(
                    "Expected exactly three fake query generations, with no retries."
                )
            before_events = container.event_log.list_events()

        reopened = create_app(settings)
        with TestClient(reopened) as client:
            requests = (
                (export_path, json_response.content),
                (export_path + "?format=markdown", markdown_response.content),
                (comparison_path, comparison_response.content),
            )
            for path, before in requests:
                after = client.get(path)
                after.raise_for_status()
                if after.content != before:
                    raise RuntimeError("Saved exports/comparison changed after restart.")
            if reopened.state.container.event_log.list_events() != before_events:
                raise RuntimeError("Reading saved artifacts must not append agent events.")
        live_attempts = sync_http.call_count + async_http.call_count
        if live_attempts:
            raise RuntimeError("The offline demo attempted an external HTTP call.")

    artifacts: tuple[tuple[str, BaseModel], ...] = (
        ("baseline.json", baseline),
        ("limited.json", limited),
        ("collection.json", collection),
        ("unknown.json", unknown),
    )
    for name, model in artifacts:
        with (output_dir / name).open("x", encoding="utf-8") as stream:
            stream.write(model.model_dump_json(indent=2) + "\n")
    for name, content in (
        ("bundle.json", json_response.content),
        ("bundle.md", markdown_response.content),
        ("comparison.json", comparison_response.content),
    ):
        with (output_dir / name).open("xb") as stream:
            stream.write(content)
    transcript = (
        "\n\n".join(
            [
                "\n".join(
                    [
                        PANEL_TITLES[0],
                        f"Synthetic papers={len(_counts(baseline.sources))}; "
                        f"chunks={len(baseline.sources)}; "
                        f"shared titles={len({s.chunk.title for s in baseline.sources})}.",
                        f"Omitted: {_count_text(baseline.sources)}",
                        f"Cap 1: {_count_text(limited.sources)}",
                        f"Hybrid / graph calls: omitted={baseline_calls[0]}/{baseline_calls[1]}; "
                        f"capped={capped_calls[0]}/{capped_calls[1]}",
                        f"Kept chunks={len(limited.sources)} of max_source_docs="
                        f"{limited.configuration.max_source_docs}; no refill.",
                    ]
                ),
                "\n".join(
                    [
                        PANEL_TITLES[1],
                        f"Collection paper-a + paper-b -> {len(collection.sources)} chunks.",
                        f"Unknown selected ID -> {len(unknown.sources)} chunks (scope retained).",
                        f"Explicit null -> HTTP {null.status_code}; "
                        f"boolean -> HTTP {boolean.status_code}.",
                        f"Preview generation calls={preview_generations}; "
                        f"preview event writes={preview_writes}.",
                        "The quota does not guarantee paper coverage or independent evidence.",
                    ]
                ),
                "\n".join(
                    [
                        PANEL_TITLES[2],
                        f"Query state={bundle.status}; provider={bundle.generation.provider}.",
                        f"Preview / query sources and context identical={same_context}.",
                        "Policy in plan + initial event + export: cap "
                        f"{policy.max_chunks_per_document}.",
                        f"Context SHA-256: {limited.context_sha256}",
                        "The fake answer is plumbing, not a scientific conclusion.",
                    ]
                ),
                "\n".join(
                    [
                        PANEL_TITLES[3],
                        "Cap 1 vs 2 on paper-c: equal context="
                        f"{not comparison.changes.context_changed}.",
                        "Comparison: evidence_policy_changed="
                        f"{comparison.changes.evidence_policy_changed}; "
                        f"any_changes={comparison.any_changes}.",
                        "Exports and comparison after restart: byte-identical=True.",
                        f"Query fake generations={query_generations}; "
                        f"live HTTP attempts={live_attempts}.",
                        "Inspect bundle.json, bundle.md, comparison.json before sharing.",
                    ]
                ),
            ]
        )
        + "\n"
    )
    with (output_dir / "transcript.txt").open("x", encoding="utf-8") as stream:
        stream.write(transcript)
    return transcript


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    print(run_demo(arguments.output_dir), end="")


if __name__ == "__main__":
    main()
