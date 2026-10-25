"""Execute the native human-screening workflow using original offline synthetic notes."""

import argparse
import json
import socket
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from urllib.parse import urlencode, urlsplit

from fastapi.testclient import TestClient

from api.application import create_app
from api.dependencies import AppContainer
from retrieval.models import Chunk, Document
from scripts.demo_corpus_explorer import browsing_guards
from scripts.demo_evidence_export import offline_settings
from scripts.demo_passage_search import SearchPage
from storage.paper_screening import ScreeningHistory

SELECTED_ID = "paper/../synthetic\r\nid\0?draft#v1"
DECISIONS = ("include", "exclude", "unsure")
REASONS = (
    "Synthetic human judgment: methods appear relevant.",
    '  Synthetic exclusion: <script>literal markup</script> & "quotes".\r\nNot a finding.  ',
    "Synthetic reassessment: read the source again before choosing evidence.",
)
PANEL_TITLES = (
    "1. Discover and inspect an exact member",
    "2. Explicitly record a human opinion",
    "3. Reject unsafe or stale submissions",
    "4. Restart and inspect retained history",
)


def run_demo(output_dir: Path) -> str:
    """Save real HTML/JSON measurements; never pretend these are browser recordings."""
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {output_dir}; choose a new directory.")
    output_dir.mkdir(parents=True)
    artifacts: dict[str, bytes] = {}
    forbidden_counts: dict[str, int] = {}
    with TemporaryDirectory(prefix="human-screening-demo-") as temporary:
        database = Path(temporary) / "synthetic.sqlite3"
        settings = offline_settings(database)
        app = create_app(settings)
        container: AppContainer = app.state.container
        documents = [
            Document(
                document_id=identifier,
                title=title,
                source="synthetic:human-screening",
                text="Original synthetic fixture, not a publication or scientific finding.",
            )
            for identifier, title in (
                (SELECTED_ID, "Synthetic methods <note>"),
                ("background", "Synthetic background note"),
            )
        ]
        container.document_store.add_documents(
            documents,
            [
                Chunk(
                    **document.model_dump(exclude={"metadata"}),
                    chunk_id=f"chunk-{index}",
                    metadata={"chunk_index": "0"},
                )
                for index, document in enumerate(documents)
            ],
        )
        collection = container.paper_collections.create(
            name="Synthetic human review",
            document_ids=[document.document_id for document in documents],
        )
        before_reads = database.read_bytes()
        with (
            TestClient(app, root_path="/review") as client,
            browsing_guards(container) as guards,
            ExitStack() as stack,
        ):
            extras = {
                name: stack.enter_context(
                    patch.object(target, method, side_effect=AssertionError("Forbidden demo work."))
                )
                for name, target, method in (
                    ("transitions", container.event_log, "append_transition"),
                    ("worksheets", container.worksheets, "build"),
                    ("socket_create", socket, "create_connection"),
                    ("socket_connect", socket.socket, "connect"),
                )
            }
            submit = stack.enter_context(
                patch.object(
                    container.paper_screening, "submit", wraps=container.paper_screening.submit
                )
            )

            def get(name: str, url: str) -> SearchPage:
                parts = urlsplit(url)
                if parts.scheme or parts.netloc or not parts.path.startswith("/review/"):
                    raise RuntimeError("Navigation left the local mounted workflow.")
                response = client.get(url)
                response.raise_for_status()
                if response.headers["cache-control"] != "no-store":
                    raise RuntimeError("Native read lost its no-store header.")
                artifacts[name] = response.content
                return SearchPage(response.text)

            catalog = get("catalog.html", "/review/explore")
            listed = get("collections.html", catalog.links["saved-collections"])
            detail = get("collection.html", listed.links["inspect-collection-1"])
            form_url = detail.links["screen-member-1"]
            form = get("form.html", form_url)
            paper = get("paper.html", form.links["screening-paper"])
            source = get("source.html", paper.links["source-context-1"])
            if (
                source.documents != [SELECTED_ID]
                or source.chunks != ["chunk-0"]
                or json.loads(form.form_values["selection"])["document_id"] != SELECTED_ID
                or form.form_values["decision"] != ""
                or database.read_bytes() != before_reads
            ):
                raise RuntimeError("Exact native selection/source or read-only boundary failed.")
            initial = {**form.form_values, "decision": DECISIONS[0], "reason": REASONS[0]}
            redirect_statuses = []
            location = ""
            for index, (decision, reason) in enumerate(zip(DECISIONS, REASONS, strict=True), 1):
                response = client.post(
                    form.form_action,
                    data={**form.form_values, "decision": decision, "reason": reason},
                    headers={"origin": "http://testserver", "sec-fetch-site": "same-origin"},
                    follow_redirects=False,
                )
                redirect_statuses.append(response.status_code)
                if response.status_code != 303 or "reason=" in response.headers["location"]:
                    raise RuntimeError("Submission did not return a local reason-free 303.")
                location = response.headers["location"]
                form = get(f"saved-{index}.html", location)
                if form.form_values["expected_decision_revision"] != str(index):
                    raise RuntimeError("The form did not read the persisted decision revision.")
            before_rejections = database.read_bytes()
            rejected = {
                "cross_origin": client.post(
                    form.form_action, data=initial, headers={"origin": "https://external.invalid"}
                ).status_code,
                "invalid": client.post(
                    form.form_action,
                    content=urlencode(initial) + "&reason=duplicate",
                    headers={
                        "origin": "http://testserver",
                        "content-type": "application/x-www-form-urlencoded",
                    },
                ).status_code,
                "stale": client.post(
                    form.form_action, data=initial, headers={"origin": "http://testserver"}
                ).status_code,
            }
            unchanged_on_rejection = database.read_bytes() == before_rejections
            history_page = get("history.html", form.links["screening-history"])
            get("history.json", history_page.links["history-json"])
            history = ScreeningHistory.model_validate_json(artifacts["history.json"])
            python_history = container.paper_screening.list_history(
                collection.collection_id, SELECTED_ID, collection_revision=1
            )
            if history != python_history:
                raise RuntimeError("Python and JSON history disagree.")
            forbidden_counts.update({name: mock.call_count for name, mock in guards.items()})
            forbidden_counts.update({name: mock.call_count for name, mock in extras.items()})
            submit_attempts = submit.call_count
        before_restart = database.read_bytes()
        restarted = create_app(settings)
        with (
            TestClient(restarted, root_path="/review") as client,
            browsing_guards(restarted.state.container) as guards,
        ):
            response = client.get(location)
            response.raise_for_status()
            artifacts["restarted.html"] = response.content
            forbidden_counts.update(
                {f"restart:{name}": mock.call_count for name, mock in guards.items()}
            )
        checks = {
            "successful_posts": len(DECISIONS),
            "redirect_statuses": redirect_statuses,
            "submit_service_attempts": submit_attempts,
            "rejected_statuses": rejected,
            "rejected_posts_database_unchanged": unchanged_on_rejection,
            "get_database_unchanged": True,
            "python_api_history_equal": history == python_history,
            "decisions_newest_first": [review.decision for review in history.items],
            "reasons_exact": [review.reason for review in history.items] == list(reversed(REASONS)),
            "revisions_newest_first": [review.revision for review in history.items],
            "unretained_revisions": history.unretained_decision_revisions,
            "collection_unchanged": container.paper_collections.get(collection.collection_id)
            == collection,
            "restart_html_identical": artifacts["saved-3.html"] == artifacts["restarted.html"],
            "restart_database_unchanged": database.read_bytes() == before_restart,
            "exact_opaque_source": True,
            "events": len(container.event_log.list_events()),
            "forbidden_call_counts": forbidden_counts,
            "output_bytes": {name: len(content) for name, content in artifacts.items()},
        }
        if (
            redirect_statuses != [303, 303, 303]
            or rejected != {"cross_origin": 403, "invalid": 422, "stale": 409}
            or submit_attempts != 4
            or checks["revisions_newest_first"] != [3, 2, 1]
            or checks["events"] != 0
            or any(forbidden_counts.values())
            or not all(
                checks[key]
                for key in (
                    "rejected_posts_database_unchanged",
                    "reasons_exact",
                    "collection_unchanged",
                    "restart_html_identical",
                    "restart_database_unchanged",
                )
            )
        ):
            raise RuntimeError("Measured native screening contract failed.")
    checks["temporary_database_removed"] = not database.exists()
    transcript = (
        "\n\n".join(
            (
                f"{PANEL_TITLES[0]}\n"
                "Executed catalog -> collection -> native screening form under /review.\n"
                "Read the exact paper/source; slash, CRLF and NUL ID remained unchanged.\n"
                "GET database bytes unchanged; decision and reason not preselected.",
                f"{PANEL_TITLES[1]}\n"
                f"Explicit include, exclude, unsure submissions: {len(DECISIONS)}.\n"
                f"Local redirect statuses: {redirect_statuses}; reasons absent from URLs.\n"
                "Persisted reasons retain spaces, CRLF and escaped literal markup.\n"
                "Collection membership unchanged; no retrieval selection changed.",
                f"{PANEL_TITLES[2]}\n"
                f"Cross-origin: {rejected['cross_origin']}; "
                f"duplicate field: {rejected['invalid']}; "
                f"stale decision: {rejected['stale']}.\n"
                f"Database bytes unchanged by rejected writes: {unchanged_on_rejection}.\n"
                f"Submit service attempts: {submit_attempts}; "
                f"successful decisions: {len(DECISIONS)}.\n"
                "No automatic retry or unconditional overwrite.",
                f"{PANEL_TITLES[3]}\n"
                f"Retained revisions, newest first: {checks['revisions_newest_first']}.\n"
                f"Python/API history equal: {checks['python_api_history_equal']}; "
                f"restart HTML identical: {checks['restart_html_identical']}.\n"
                f"Events: {checks['events']}; forbidden calls: {sum(forbidden_counts.values())}.\n"
                "Temporary synthetic database removed; no network, retrieval or generation.\n"
                "Measured-output illustration, not browser footage or scientific validation.",
            )
        )
        + "\n"
    )
    artifacts["checks.json"] = (json.dumps(checks, indent=2, ensure_ascii=True) + "\n").encode()
    artifacts["transcript.txt"] = transcript.encode()
    for name, content in artifacts.items():
        (output_dir / name).write_bytes(content)
    return transcript


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()
    print(run_demo(arguments.output_dir), end="")


if __name__ == "__main__":
    main()
