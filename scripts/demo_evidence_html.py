"""Measure an offline HTML reader using actual synthetic query/export routes and no live HTTP."""

import argparse
import hashlib
import json
import socket
import sqlite3
from contextlib import ExitStack, closing
from html.parser import HTMLParser
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import httpx
from fastapi.testclient import TestClient

from agent.evidence import EvidenceBundle
from api.application import create_app
from api.dependencies import AppContainer
from api.schemas import QueryResponse
from scripts.demo_evidence_export import offline_settings

PANEL_TITLES = (
    "1. Save synthetic evidence and a fake answer",
    "2. Open the downloaded reader offline",
    "3. Follow references to frozen passages",
    "4. Remove corpus, restart, export unchanged",
)
OUTPUT_NAMES = (
    "evidence.json",
    "evidence.md",
    "evidence.html",
    "restarted.html",
    "checks.json",
    "transcript.txt",
)
DEMO_TEXT = (
    "GraphRAG connects synthetic research evidence. This original workshop note is not a paper "
    "or a scientific finding. The saved reader retains complete captured passages, not only "
    "the first 240 characters of a citation. A second note supplies a different passage. "
    "Literal markup must stay text: <script>synthetic example only</script> and &lt;b&gt;. "
    "Unicode remains readable: caf\u00e9, \u7814\u7a76, \U0001f52c. "
    "The placeholder answer demonstrates provenance, not correctness."
)


class _ReaderLinks(HTMLParser):
    def __init__(self, html: str) -> None:
        super().__init__(convert_charrefs=True)
        self.identifiers: set[str] = set()
        self.links: list[str] = []
        self.styles = 0
        self.feed(html)
        self.close()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag in {"script", "link", "img", "base", "iframe", "object", "embed", "form", "svg"}:
            raise RuntimeError("The reader contains an active or external-resource element.")
        if "style" in attributes or any(name.startswith("on") for name in attributes):
            raise RuntimeError("The reader contains inline behavior or styling.")
        if tag == "style":
            self.styles += 1
        if identifier := attributes.get("id"):
            if identifier in self.identifiers:
                raise RuntimeError("Duplicate reader fragment ID.")
            self.identifiers.add(identifier)
        if tag == "a":
            href = attributes.get("href")
            if href is None or not href.startswith("#"):
                raise RuntimeError("The reader contains a non-local link.")
            self.links.append(href)


def _block_export_work(stack: ExitStack, container: AppContainer) -> list[Mock]:
    return [
        stack.enter_context(
            patch.object(
                component,
                method,
                side_effect=AssertionError(
                    "HTML export attempted agent/corpus work or an event write."
                ),
            )
        )
        for component, method in (
            (container.runner, "run"),
            (container.runner, "preview"),
            (container.llm, "generate"),
            (container.hybrid_retriever, "retrieve"),
            (container.document_store, "list_chunks"),
            (container.document_store, "add_documents"),
            (container.graph_store, "chunks_for_entities"),
            (container.event_log, "append_event"),
            (container.event_log, "append_transition"),
        )
    ]


def run_demo(output_dir: Path) -> str:
    """Save actual downloads/checks; never open an ambient DB or overwrite named artifacts."""
    for name in OUTPUT_NAMES:
        path = output_dir / name
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite {path}; choose a new directory.")
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, bytes] = {}
    with TemporaryDirectory(prefix="scholar-html-reader-") as temporary, ExitStack() as stack:
        network = [
            stack.enter_context(
                patch.object(
                    target,
                    method,
                    side_effect=AssertionError(
                        "The synthetic HTML-reader demo forbids networking."
                    ),
                )
            )
            for target, method in (
                (httpx.HTTPTransport, "handle_request"),
                (httpx.AsyncHTTPTransport, "handle_async_request"),
                (socket, "create_connection"),
                (socket.socket, "connect"),
            )
        ]
        database_path = Path(temporary) / "reader.sqlite3"
        settings = offline_settings(database_path)
        application = create_app(settings)
        container: AppContainer = application.state.container
        with TestClient(application) as client:
            for title, text in (
                ("Synthetic markup and Unicode note", DEMO_TEXT),
                (
                    "Synthetic comparison note",
                    "GraphRAG connects a second synthetic evidence passage. "
                    "This note records a different example, "
                    "not independent scientific confirmation.",
                ),
            ):
                ingested = client.post(
                    "/ingest/text",
                    json={
                        "title": title,
                        "text": text,
                        "source": "synthetic:offline-reader",
                    },
                )
                ingested.raise_for_status()
            queried = client.post("/query", json={"query": "How does GraphRAG connect evidence?"})
            queried.raise_for_status()
            result = QueryResponse.model_validate_json(queried.content).result
            if result.state != "DONE":
                raise RuntimeError(f"The synthetic query failed: {result.error}")
            path = f"/runs/{result.run_id}/export"
            events_before = container.event_log.list_events()
            blocked = _block_export_work(stack, container)
            saved_bytes = database_path.read_bytes()
            for fmt, name in (
                ("json", "evidence.json"),
                ("markdown", "evidence.md"),
                ("html", "evidence.html"),
            ):
                response = client.get(path, params={"format": fmt})
                response.raise_for_status()
                artifacts[name] = response.content
            html_response = response
            bundle = EvidenceBundle.model_validate_json(artifacts["evidence.json"])
            if bundle.generation.provider != "fake" or len(bundle.snapshot.sources) != 2:
                raise RuntimeError("Expected fake generation with two captured synthetic passages.")
            reader = _ReaderLinks(artifacts["evidence.html"].decode("utf-8"))
            all_links_resolve = all(link[1:] in reader.identifiers for link in reader.links)
            ranks = {source.chunk.chunk_id: source.rank for source in bundle.snapshot.sources}
            expected_links = [
                f"#source-{ranks[chunk_id]}"
                for link in bundle.claim_evidence
                for chunk_id in [*link.proposed_chunk_ids, *link.grounded_chunk_ids]
                if chunk_id in ranks
            ] + [
                f"#source-{link.evidence_rank}"
                for link in bundle.citation_evidence
                if link.evidence_rank is not None
            ]
            source_links = [link for link in reader.links if link.startswith("#source-")]
            if (
                not all_links_resolve
                or sorted(source_links) != sorted(expected_links)
                or reader.styles != 1
            ):
                raise RuntimeError("The actual reader has inconsistent references or styling.")
            if database_path.read_bytes() != saved_bytes:
                raise RuntimeError("Export wrote to the synthetic database.")
            # Only this newly created synthetic database is changed, outside export requests.
            with closing(sqlite3.connect(database_path)) as connection, connection:
                connection.execute("DELETE FROM entity_edges")
                connection.execute("DELETE FROM entity_mentions")
                connection.execute("DELETE FROM graph_chunks")
                connection.execute("DELETE FROM chunks")
                connection.execute("DELETE FROM documents")
                remaining_chunks = connection.execute("SELECT count(*) FROM chunks").fetchone()[0]
        restarted = create_app(settings)
        blocked.extend(_block_export_work(stack, restarted.state.container))
        saved_bytes = database_path.read_bytes()
        format_matches = {}
        with TestClient(restarted) as client:
            for fmt, name in (
                ("json", "evidence.json"),
                ("markdown", "evidence.md"),
                ("html", "evidence.html"),
            ):
                after = client.get(path, params={"format": fmt})
                after.raise_for_status()
                format_matches[fmt] = after.content == artifacts[name]
                if not format_matches[fmt]:
                    raise RuntimeError(f"The saved {fmt} export changed after corpus removal.")
                if fmt == "html":
                    artifacts["restarted.html"] = after.content
        events_after = restarted.state.container.event_log.list_events()
        work_calls = sum(guard.call_count for guard in blocked)
        network_calls = sum(guard.call_count for guard in network)
        if (
            work_calls
            or network_calls
            or remaining_chunks != 0
            or events_before != events_after
            or database_path.read_bytes() != saved_bytes
        ):
            raise RuntimeError("Reader export performed unexpected work or changed saved state.")
    database_removed = not database_path.exists()
    if not database_removed:
        raise RuntimeError("The temporary synthetic database was not removed.")
    checks = {
        "captured_sources": len(bundle.snapshot.sources),
        "final_citations": len(bundle.answer.citations),
        "source_links": len(source_links),
        "all_fragment_links_resolve": all_links_resolve,
        "html_bytes": len(artifacts["evidence.html"]),
        "html_sha256": {
            "before": hashlib.sha256(artifacts["evidence.html"]).hexdigest(),
            "after": hashlib.sha256(artifacts["restarted.html"]).hexdigest(),
        },
        "content_security_policy": html_response.headers["content-security-policy"],
        "all_formats_unchanged": all(format_matches.values()),
        "events_unchanged": events_before == events_after,
        "export_agent_or_corpus_calls": work_calls,
        "export_event_writes": len(events_after) - len(events_before),
        "network_calls": network_calls,
        "remaining_chunks": remaining_chunks,
        "temporary_database_removed": database_removed,
    }
    artifacts["checks.json"] = (json.dumps(checks, sort_keys=True, indent=2) + "\n").encode("utf-8")
    claim = bundle.claim_evidence[0]
    transcript = (
        "\n\n".join(
            [
                "\n".join(
                    [
                        PANEL_TITLES[0],
                        f"POST /ingest/text -> {ingested.status_code}; "
                        "two original synthetic notes",
                        f"POST /query -> {queried.status_code}; saved state={bundle.status}",
                        f"Saved provider: {bundle.generation.provider}; "
                        f"model: {bundle.generation.model_name}",
                        f"Captured passages: {len(bundle.snapshot.sources)}; "
                        f"final citations: {len(bundle.answer.citations)}",
                        "Ambient credentials, settings and .env ignored; no real papers used.",
                        "Fake output is a workflow placeholder, not scientific evidence.",
                    ]
                ),
                "\n".join(
                    [
                        PANEL_TITLES[1],
                        f"GET /runs/<run_id>/export?format=html -> {html_response.status_code}",
                        f"Content type: {html_response.headers['content-type']}",
                        f"HTML bytes: {len(artifacts['evidence.html'])}; "
                        f"fixed stylesheets: {reader.styles}",
                        "CSP: default-src 'none'; only hashed fixed CSS is allowed.",
                        "Literal markup stays text; every link is a local fragment.",
                        "Use browser-native links, details, Find and Print; no live app.",
                    ]
                ),
                "\n".join(
                    [
                        PANEL_TITLES[2],
                        "Claim 1 proposed ranks: "
                        f"{[ranks[c] for c in claim.proposed_chunk_ids if c in ranks]}",
                        "Claim 1 accepted ranks: "
                        f"{[ranks[c] for c in claim.grounded_chunk_ids if c in ranks]}",
                        f"Missing IDs in this fixture: {claim.missing_chunk_ids}",
                        f"Source links checked: {len(source_links)}; every target resolves",
                        "Full passages, scores, paths, metadata and SHA-256 stay inspectable.",
                        "A resolved reference and token overlap do not validate a claim.",
                    ]
                ),
                "\n".join(
                    [
                        PANEL_TITLES[3],
                        f"Remaining corpus chunks: {remaining_chunks}; "
                        "HTML after restart: byte-identical",
                        "JSON and Markdown after restart: byte-identical",
                        f"Export agent/corpus calls: {work_calls}; "
                        f"event writes: {checks['export_event_writes']}",
                        f"External HTTP/socket calls: {network_calls}; saved events unchanged",
                        f"Temporary database removed: {database_removed}; "
                        "digests saved in checks.json",
                        "Actual offline output; illustration, not a screen recording.",
                    ]
                ),
            ]
        )
        + "\n"
    )
    artifacts["transcript.txt"] = transcript.encode("utf-8")
    for name, content in artifacts.items():
        with (output_dir / name).open("xb") as output:
            output.write(content)
    return transcript


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    print(run_demo(arguments.output_dir), end="")


if __name__ == "__main__":
    main()
