"""Measure the native comparison workflow on original offline synthetic saved runs."""

import argparse
import json
import socket
import sqlite3
from contextlib import ExitStack, closing
from html import escape
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

import httpx
from fastapi.testclient import TestClient

from agent.comparison_models import RunComparison
from agent.models import AgentState
from api.application import create_app
from api.schemas import IngestResponse, QueryResponse
from scripts.demo_corpus_explorer import _no_work, browsing_guards
from scripts.demo_evidence_export import offline_settings
from scripts.demo_passage_search import SearchPage
from scripts.demo_saved_run_browser import _get
from storage.run_comparison import SavedRunComparator

PANEL_TITLES = (
    "1. Choose an explicit saved baseline",
    "2. Inspect recorded outputs side by side",
    "3. Reverse direction and open frozen sources",
    "4. Restart after deleting the synthetic corpus",
)


class ComparisonPage(SearchPage):
    """Read literal fields and native controls from actual HTML, not simulated UI data."""

    def __init__(self, html: str) -> None:
        self.values: dict[str, list[str]] = {}
        self._classes: list[str] = []
        self._value = ""
        super().__init__(html)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        super().handle_starttag(tag, attrs)
        if tag == "pre":
            self._classes = str(dict(attrs).get("class", "")).split()
            self._value = ""

    def handle_data(self, data: str) -> None:
        super().handle_data(data)
        if self._classes:
            self._value += data

    def handle_endtag(self, tag: str) -> None:
        super().handle_endtag(tag)
        if tag == "pre":
            for css_class in self._classes:
                self.values.setdefault(css_class, []).append(self._value)
            self._classes = []


def _verify(html: str, comparison: RunComparison) -> ComparisonPage:
    page = ComparisonPage(html)
    if [json.loads(value) for value in page.values["compared-run-id"]] != [
        comparison.baseline.run_id,
        comparison.candidate.run_id,
    ] or json.loads(page.values["evidence-statistics"][0]) != (
        comparison.evidence.statistics.model_dump(mode="json")
    ):
        raise RuntimeError("Native sides or evidence statistics disagree with saved comparison.")
    for name, changed in comparison.changes.model_dump().items():
        label = "Changed" if changed else "Unchanged"
        if f'id="{name}">{label}</td>' not in html:
            raise RuntimeError("Native change flags disagree with saved comparison.")
    return page


def run_demo(output_dir: Path) -> str:
    """Save executed responses, remove the temporary fixture DB, and refuse overwrites."""
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {output_dir}; choose a new directory.")
    output_dir.mkdir(parents=True)
    artifacts: dict[str, bytes] = {}
    with (
        TemporaryDirectory(prefix="scholar-comparison-workbench-") as temporary,
        ExitStack() as stack,
    ):
        network = [
            stack.enter_context(patch.object(target, method, side_effect=_no_work))
            for target, method in (
                (socket, "create_connection"),
                (socket.socket, "connect"),
                (httpx.HTTPTransport, "handle_request"),
                (httpx.AsyncHTTPTransport, "handle_async_request"),
            )
        ]
        database = Path(temporary) / "synthetic.sqlite3"
        settings = offline_settings(database)
        app = create_app(settings)
        container = app.state.container
        with TestClient(app) as client:
            documents = []
            for label in ("baseline-only", "shared", "candidate-only"):
                response = client.post(
                    "/ingest/text",
                    json={
                        "title": f"Synthetic {label} comparison note",
                        "source": "synthetic:comparison-workbench",
                        "text": f"GraphRAG connects synthetic {label} research evidence. "
                        "This original workshop fixture is not a scientific finding. "
                        "Literal markup stays text: <script>not executable</script> & context.",
                    },
                )
                response.raise_for_status()
                documents.append(IngestResponse.model_validate_json(response.content).document_id)
            runs = []
            for question, scope in (
                ("How does GraphRAG connect synthetic evidence?", documents[:2]),
                ("What does the synthetic GraphRAG evidence omit?", documents[1:]),
            ):
                response = client.post("/query", json={"query": question, "document_ids": scope})
                response.raise_for_status()
                result = QueryResponse.model_validate_json(response.content).result
                if result.state != AgentState.DONE:
                    raise RuntimeError("An isolated fake-adapter fixture did not complete.")
                runs.append(result.run_id)
            baseline, candidate = runs
            comparator = SavedRunComparator(container.saved_run_evidence)
            expected = comparator.compare(baseline, candidate)
            events = container.event_log.list_events()
            before = database.read_bytes()
            with browsing_guards(container) as guards:
                api_result = _get(client, f"/runs/{baseline}/compare/{candidate}")
                if RunComparison.model_validate_json(api_result.content) != expected:
                    raise RuntimeError("Existing JSON API and bounded comparison disagree.")
                with (
                    patch.object(container.evidence_exporter, "export", side_effect=_no_work),
                    patch.object(container.event_log, "list_events", side_effect=_no_work),
                ):
                    first = _get(client, "/explore/runs?state=DONE&limit=1")
                    history_url = SearchPage(first.text).links["next-page"]
                    history = _get(client, history_url)
                    chooser = _get(client, SearchPage(history.text).links["compare-run-1"])
                    controls = SearchPage(chooser.text)
                    params = {**controls.form_values, "candidate": json.dumps(candidate)}
                    response = client.get(controls.form_action, params=params)
                    response.raise_for_status()
                    result_url = response.url.raw_path.decode("ascii")
                    result_page = _verify(response.text, expected)
                    same = _get(client, result_page.links["same-run-control"])
                    _verify(same.text, comparator.compare(baseline, baseline))
                    swapped = _get(client, result_page.links["swap-comparison"])
                    _verify(swapped.text, comparator.compare(candidate, baseline))
                    source_url = result_page.links["shared-1-candidate-source"]
                    source = _get(client, source_url)
                    parts = urlsplit(source_url)
                    selected = container.saved_run_evidence.export(
                        parse_qs(parts.query)["run_id"][0]
                    ).snapshot.sources[int(parts.fragment.removeprefix("source-")) - 1]
                    if (
                        f'id="{parts.fragment}"' not in source.text
                        or escape(selected.chunk.text).replace("\r", "&#13;") not in source.text
                        or SearchPage(source.text).links["back-to-runs"] != history_url
                        or _get(client, result_page.links["back-to-runs"]).content
                        != history.content
                    ):
                        raise RuntimeError("Exact frozen source or history return was lost.")
                    missing = client.get(
                        controls.form_action,
                        params={
                            **params,
                            "candidate": json.dumps("synthetic-missing-run"),
                        },
                    )
                    if (
                        missing.status_code != 404
                        or "Candidate evidence unavailable" not in missing.text
                    ):
                        raise RuntimeError(
                            "Missing candidate was not rejected as a whole comparison."
                        )
                counts = {name: guard.call_count for name, guard in guards.items()}
            if database.read_bytes() != before or container.event_log.list_events() != events:
                raise RuntimeError("Read-only comparison changed SQLite bytes or saved events.")
            for name, content in (
                ("history.html", history.content),
                ("chooser.html", chooser.content),
                ("comparison.html", response.content),
                ("same.html", same.content),
                ("swapped.html", swapped.content),
                ("source.html", source.content),
                ("missing.html", missing.content),
                ("comparison.json", api_result.content),
            ):
                artifacts[name] = content
            for name, run_id in (("baseline", baseline), ("candidate", candidate)):
                artifacts[name + ".json"] = (
                    container.saved_run_evidence.export(run_id).model_dump_json(indent=2).encode()
                )
        with closing(sqlite3.connect(database)) as connection, connection:
            for table in ("entity_edges", "entity_mentions", "graph_chunks", "chunks", "documents"):
                connection.execute(f"DELETE FROM {table}")  # noqa: S608
            remaining = connection.execute("SELECT count(*) FROM chunks").fetchone()[0]
        restarted = create_app(settings)
        before = database.read_bytes()
        with TestClient(restarted) as client, browsing_guards(restarted.state.container) as guards:
            with patch.object(
                restarted.state.container.evidence_exporter, "export", side_effect=_no_work
            ):
                after = _get(client, result_url)
                frozen_after = _get(client, source_url)
            if after.content != response.content or frozen_after.content != source.content:
                raise RuntimeError(
                    "Corpus deletion/restart changed the comparison or frozen source."
                )
            counts.update({f"restart:{name}": guard.call_count for name, guard in guards.items()})
        after_events = restarted.state.container.event_log.list_events()
        if (
            before != database.read_bytes()
            or events != after_events
            or remaining
            or any(counts.values())
            or any(guard.call_count for guard in network)
        ):
            raise RuntimeError("Measured offline/no-write invariants failed.")
        checks = {
            "fixture": "saved-run-comparison-workbench-synthetic-v1",
            "native_python_api_parity": True,
            "exact_history_return": True,
            "restart_comparison_identical": True,
            "restart_source_identical": True,
            "database_bytes_unchanged_during_reads": True,
            "counts": expected.evidence.statistics.model_dump(mode="json"),
            "changed_fields": sum(expected.changes.model_dump().values()),
            "same_run_changes": comparator.compare(baseline, baseline).any_changes,
            "events_before": len(events),
            "events_after": len(after_events),
            "remaining_corpus_chunks": remaining,
            "missing_candidate_status": missing.status_code,
            "forbidden_call_counts": counts,
            "external_calls": sum(guard.call_count for guard in network),
            "source_rank": selected.rank,
        }
    checks["temporary_database_removed"] = not database.exists()
    if not checks["temporary_database_removed"]:
        raise RuntimeError("Temporary fixture database was not removed.")
    stats = expected.evidence.statistics
    transcript = (
        "\n\n".join(
            (
                "\n".join(
                    (
                        PANEL_TITLES[0],
                        f"Two actual offline fake-adapter queries saved {len(events)} events.",
                        "History -> baseline picker -> native GET form: "
                        f"HTTP {response.status_code}.",
                        "JSON-string choices preserve exact IDs; no automatic baseline.",
                        "Original synthetic fixtures; no real papers, credentials or live models.",
                    )
                ),
                "\n".join(
                    (
                        PANEL_TITLES[1],
                        f"Frozen source counts: baseline={stats.baseline_count}; "
                        f"candidate={stats.candidate_count}.",
                        f"Shared={stats.shared_count}; added={stats.added_count}; "
                        f"removed={stats.removed_count}.",
                        f"Changed saved fields={checks['changed_fields']}; "
                        "native/Python/JSON API agree.",
                        f"Same-run any_changes={checks['same_run_changes']}; "
                        "no quality or release verdict.",
                    )
                ),
                "\n".join(
                    (
                        PANEL_TITLES[2],
                        f"Swap: HTTP {swapped.status_code}; "
                        "baseline and candidate explicitly reverse.",
                        f"Exact candidate frozen source: HTTP {source.status_code}; "
                        f"rank={selected.rank}.",
                        "Source reader and comparison retain the original filtered history page.",
                        f"Missing candidate: HTTP {missing.status_code}; no partial comparison.",
                    )
                ),
                "\n".join(
                    (
                        PANEL_TITLES[3],
                        f"Remaining corpus chunks={remaining}; "
                        f"events before/after={len(events)}/{len(after_events)}.",
                        "Restarted comparison and source HTML are byte-identical.",
                        "Guarded generation/retrieval/writes/external calls="
                        f"{sum(counts.values()) + checks['external_calls']}.",
                        "Temporary fixture database removed; inspectable responses retained.",
                        "Measured-output illustration, not browser footage "
                        "or a fair A/B experiment.",
                    )
                ),
            )
        )
        + "\n"
    )
    artifacts["transcript.txt"] = transcript.encode()
    artifacts["checks.json"] = (json.dumps(checks, indent=2) + "\n").encode()
    for name, content in artifacts.items():
        with (output_dir / name).open("xb") as output:
            output.write(content)
    return transcript


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()
    print(run_demo(arguments.output_dir), end="")


if __name__ == "__main__":
    main()
