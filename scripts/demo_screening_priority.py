"""Execute offline screening prioritization: cold start, learned order, browser queue, simulation.

Writes only a new temporary database and the requested output directory. The simulated
reviewer records fixture labels through the real JSON API; the application never decides.
"""

import argparse
import hashlib
import json
import re
import socket
import statistics
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch
from urllib.parse import quote, urlsplit

from fastapi.testclient import TestClient

from api.application import create_app
from api.dependencies import AppContainer
from retrieval.models import Document
from scripts.demo_corpus_explorer import _no_work, browsing_guards
from scripts.demo_evidence_export import offline_settings
from scripts.demo_passage_search import SearchPage
from scripts.screening_priority_fixtures import (
    RELEVANT_PAPERS,
    TOTAL_PAPERS,
    collection_order,
    synthetic_papers,
)
from storage.screening_priority import (
    ScreeningPriority,
    ScreeningSimulation,
    SimulatedPaper,
    prioritize_screening,
    simulate_screening,
)

MAIN_SEED = 0
SIMULATION_SEEDS = tuple(range(10))
API_DECISIONS = 12
PANEL_TITLES = (
    "1. Cold start keeps your saved order",
    "2. Learn only from your include/exclude decisions",
    "3. Opt-in browser queue beside the decision forms",
    "4. Simulated reviewer: prioritized vs saved order",
)
OUTPUT_NAMES = (
    "cold.json",
    "ready.json",
    "collection.html",
    "priority.html",
    "form.html",
    "simulation.json",
    "checks.json",
    "transcript.txt",
)
SIMULATED_REASON = "Simulated reviewer: synthetic fixture label for the offline demo only."


def _simulations(papers: list[SimulatedPaper]) -> dict[int, dict[str, ScreeningSimulation]]:
    results = {}
    for seed in SIMULATION_SEEDS:
        ordered = collection_order(papers, seed)
        results[seed] = {
            strategy: simulate_screening(ordered, strategy=strategy)
            for strategy in ("most_likely_relevant_first", "collection_order")
        }
    return results


def _summary(values: list[int]) -> str:
    return f"{statistics.median(values):g} ({min(values)}-{max(values)})"


def _terms(item: dict[str, Any]) -> str:
    return ", ".join(term["term"] for term in item["top_terms"][:3]) or "none"


def run_demo(output_dir: Path) -> str:
    """Save real response bytes; setup writes only touch a new temporary database."""
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {output_dir}; choose a new directory.")
    output_dir.mkdir(parents=True)
    artifacts: dict[str, bytes] = {}

    def save(name: str, content: bytes) -> None:
        with (output_dir / name).open("xb") as output:
            output.write(content)
        artifacts[name] = content

    papers = synthetic_papers()
    labels = {paper.document_id: paper.relevant for paper in papers}
    ordered = collection_order(papers, MAIN_SEED)
    simulations = _simulations(papers)
    main = simulations[MAIN_SEED]
    counts: dict[str, int] = {}
    with (
        TemporaryDirectory(prefix="scholar-screening-priority-") as temporary,
        ExitStack() as stack,
    ):
        network = {
            method: stack.enter_context(patch.object(target, method, side_effect=_no_work))
            for target, method in ((socket, "create_connection"), (socket.socket, "connect"))
        }
        database = Path(temporary) / "synthetic.sqlite3"
        settings = offline_settings(database)
        app = create_app(settings)
        container: AppContainer = app.state.container
        container.document_store.add_documents(
            [
                Document(
                    document_id=paper.document_id,
                    title=paper.title,
                    text=paper.text,
                    source="synthetic:screening-priority",
                )
                for paper in ordered
            ],
            [],
        )
        collection = container.paper_collections.create(
            name="Synthetic grounded-answer review",
            document_ids=[paper.document_id for paper in ordered],
        )
        endpoint = f"/collections/{collection.collection_id}/screening-priority"
        with TestClient(app) as client:

            def priority(name: str | None = None) -> dict[str, Any]:
                response = client.get(endpoint, params={"collection_revision": 1})
                response.raise_for_status()
                if response.headers["cache-control"] != "no-store":
                    raise RuntimeError("The ordering lost its no-store header.")
                if name is not None:
                    save(name, response.content)
                result: dict[str, Any] = response.json()
                return result

            cold = priority("cold.json")
            if (
                cold["model_status"] != "needs_include_and_exclude"
                or [item["document_id"] for item in cold["items"]]
                != [paper.document_id for paper in ordered]
                or any(item["score"] is not None for item in cold["items"])
            ):
                raise RuntimeError("Cold start did not keep saved order without scores.")
            screened = []
            for _ in range(API_DECISIONS):
                chosen = priority()["items"][0]["document_id"]
                response = client.put(
                    f"/collections/{collection.collection_id}/screening/{quote(chosen, safe='')}",
                    json={
                        "collection_revision": 1,
                        "expected_decision_revision": 0,
                        "decision": "include" if labels[chosen] else "exclude",
                        "reason": SIMULATED_REASON,
                    },
                )
                response.raise_for_status()
                screened.append(chosen)
            if (
                screened
                != list(main["most_likely_relevant_first"].screened_document_ids)[:API_DECISIONS]
            ):
                raise RuntimeError("The API-driven queue diverged from the pure simulation.")
            unsure_id = priority()["items"][1]["document_id"]
            client.put(
                f"/collections/{collection.collection_id}/screening/{quote(unsure_id, safe='')}",
                json={
                    "collection_revision": 1,
                    "expected_decision_revision": 0,
                    "decision": "unsure",
                    "reason": "Simulated reviewer defers this paper; not a training label.",
                },
            ).raise_for_status()
            ready = priority("ready.json")
            before = database.read_bytes()
            with browsing_guards(container) as guards, ExitStack() as reads:
                extra = {
                    name: reads.enter_context(patch.object(target, method, side_effect=_no_work))
                    for name, target, method in (
                        ("submit", container.paper_screening, "submit"),
                        ("transitions", container.event_log, "append_transition"),
                    )
                }

                def get(name: str, url: str) -> SearchPage:
                    parts = urlsplit(url)
                    if parts.scheme or parts.netloc or not parts.path.startswith("/explore"):
                        raise RuntimeError("Navigation left the local read-only workflow.")
                    response = client.get(url)
                    response.raise_for_status()
                    save(name, response.content)
                    return SearchPage(response.text)

                detail = get(
                    "collection.html",
                    "/explore/collection?"
                    + f"collection_id={collection.collection_id}&collection_revision=1&limit=3",
                )
                page = get("priority.html", detail.links["screening-priority"])
                form = get("form.html", page.links["screen-member-1"])
                again = priority()
                counts.update({name: guard.call_count for name, guard in guards.items()})
                counts.update({name: guard.call_count for name, guard in extra.items()})
            reads_unchanged = database.read_bytes() == before
            python = prioritize_screening(
                container.paper_screening, collection.collection_id, collection_revision=1
            )
        restarted = create_app(settings)
        with TestClient(restarted) as client:
            response = client.get(endpoint, params={"collection_revision": 1})
            restart_identical = response.content == artifacts["ready.json"]
        counts.update({f"socket:{name}": guard.call_count for name, guard in network.items()})
    ranked = [item["document_id"] for item in ready["items"]]
    html = artifacts["priority.html"].decode("utf-8")
    card = html[html.index('<article class="priority-item">') : html.index("</article>")]
    shown = re.search(
        r'class="priority-score">([^<]+)</strong>; log-odds <span class="priority-log-odds">'
        r"([^<]+)</span>",
        card,
    )
    terms = re.findall(
        r'class="top-term" translate="no">([^<]+)</code> <span class="term-contribution">'
        r"([^<]+)</span>",
        card,
    )
    if shown is None or not terms:
        raise RuntimeError("The rank-1 browser card lost its score or explanation.")
    checks: dict[str, Any] = {
        "synthetic_only": True,
        "papers": TOTAL_PAPERS,
        "relevant_papers": RELEVANT_PAPERS,
        "cold_status": cold["model_status"],
        "api_decisions": API_DECISIONS,
        "api_matches_simulation": True,
        "ready_status": ready["model_status"],
        "training_relevant": ready["training_relevant"],
        "training_irrelevant": ready["training_irrelevant"],
        "unsure_items": len(ready["unsure_items"]),
        "feature_terms": ready["feature_terms"],
        "collection_page_saved_order": detail.documents
        == [paper.document_id for paper in ordered][:3],
        "browser_matches_api": page.documents[: len(ranked)] == ranked,
        "form_decision_blank": form.form_values.get("decision") == ""
        and json.loads(form.form_values["selection"])["document_id"] == ranked[0],
        "python_matches_api": python == ScreeningPriority.model_validate(ready),
        "repeat_read_identical": again == ready,
        "reads_database_unchanged": reads_unchanged,
        "restart_identical": restart_identical,
        "database_removed": not database.exists(),
        "forbidden_call_counts": counts,
    }
    if not all(
        checks[key]
        for key in (
            "collection_page_saved_order",
            "browser_matches_api",
            "form_decision_blank",
            "python_matches_api",
            "repeat_read_identical",
            "reads_database_unchanged",
            "restart_identical",
            "database_removed",
        )
    ) or any(counts.values()):
        raise RuntimeError("Measured read-only prioritization checks failed.")
    table = {
        str(seed): {
            strategy: {
                "papers_to_recall_target": result.papers_to_recall_target,
                "papers_to_find_all_relevant": result.papers_to_find_all_relevant,
                "work_saved_over_sampling": result.work_saved_over_sampling,
                "longest_irrelevant_run_before_last_relevant": (
                    result.longest_irrelevant_run_before_last_relevant
                ),
            }
            for strategy, result in results.items()
        }
        for seed, results in simulations.items()
    }
    save(
        "simulation.json",
        (
            json.dumps(
                {
                    "recall_target": 0.95,
                    "relevant_target": main["collection_order"].relevant_target,
                    "seeds": table,
                    "main_seed_recall_curves": {
                        strategy: list(result.relevant_found) for strategy, result in main.items()
                    },
                },
                indent=2,
            )
            + "\n"
        ).encode("utf-8"),
    )
    prioritized = [simulations[seed]["most_likely_relevant_first"] for seed in SIMULATION_SEEDS]
    default = [simulations[seed]["collection_order"] for seed in SIMULATION_SEEDS]
    checks["simulation"] = {
        "main_seed": MAIN_SEED,
        "prioritized_to_recall_target": main["most_likely_relevant_first"].papers_to_recall_target,
        "default_to_recall_target": main["collection_order"].papers_to_recall_target,
        "median_prioritized": statistics.median(r.papers_to_recall_target for r in prioritized),
        "median_default": statistics.median(r.papers_to_recall_target for r in default),
    }
    if not all(
        fast.papers_to_recall_target < slow.papers_to_recall_target
        for fast, slow in zip(prioritized, default, strict=True)
    ):
        raise RuntimeError("Prioritization did not beat saved order on every synthetic shuffle.")
    first = ready["items"][:3]
    target = main["collection_order"].relevant_target
    panels = [
        f"{PANEL_TITLES[0]}\n"
        "GET /collections/{id}/screening-priority?collection_revision=1 -> HTTP 200\n"
        f'"model_status": "{cold["model_status"]}", "ordering": "{cold["ordering"]}"\n'
        f'"counts": unscreened {cold["counts"]["unscreened"]}, include 0, exclude 0\n'
        "First items: "
        + ", ".join(item["document_id"].removeprefix("synthetic-") for item in cold["items"][:2])
        + '; "score": null\n'
        "Saved collection order is kept; no score is invented before one include\n"
        "and one exclude exist.",
        f"{PANEL_TITLES[1]}\n"
        f"{API_DECISIONS} simulated fixture labels recorded explicitly via PUT, plus 1 unsure.\n"
        f'"model_status": "{ready["model_status"]}", "training_relevant": '
        f'{ready["training_relevant"]}, "training_irrelevant": {ready["training_irrelevant"]}\n'
        f'"feature_terms": {ready["feature_terms"]}; the unsure paper is not a label and is '
        "listed apart.\n"
        + "\n".join(
            f"#{item['rank']} {item['document_id'].removeprefix('synthetic-')} "
            f"score {item['score']:.3f}: {_terms(item)}"
            for item in first
        )
        + f"\nWith {ready['training_irrelevant']} exclude, near misses sharing query words rank "
        "high; each\nexclude you record retrains the next order. Nothing was decided for you.",
        f"{PANEL_TITLES[2]}\n"
        "/explore/collection keeps saved order; its opt-in link opened the queue page.\n"
        f'Rank 1 card: "{page.documents[0]}", status unscreened,\n'
        f"score {shown.group(1)}, log-odds {shown.group(2)}, terms "
        + ", ".join(f"{term} {value}" for term, value in terms[:2])
        + "\n"
        "Its link opens the existing /explore/screening form; decision blank: "
        f"{checks['form_decision_blank']}.\n"
        f"Browser order equals JSON order: {checks['browser_matches_api']}; "
        f"reads changed database bytes: {not reads_unchanged}.\n"
        f"Forbidden calls: {sum(counts.values())}; restarted app returned identical JSON: "
        f"{restart_identical}.",
        f"{PANEL_TITLES[3]}\n"
        f"Synthetic templated set: {TOTAL_PAPERS} papers, {RELEVANT_PAPERS} relevant "
        "(easier than real reviews).\n"
        f"Seed {MAIN_SEED}: {target}/{RELEVANT_PAPERS} relevant after "
        f"{main['most_likely_relevant_first'].papers_to_recall_target} papers prioritized vs "
        f"{main['collection_order'].papers_to_recall_target} in saved order.\n"
        f"WSS@95: {main['most_likely_relevant_first'].work_saved_over_sampling:.2f} vs "
        f"{main['collection_order'].work_saved_over_sampling:.2f}.\n"
        f"{len(SIMULATION_SEEDS)} shuffled orders, median (range) to {target}/{RELEVANT_PAPERS}: "
        f"{_summary([r.papers_to_recall_target for r in prioritized])} vs "
        f"{_summary([r.papers_to_recall_target for r in default])}.\n"
        "Longest exclude run before the last relevant (prioritized): up to "
        f"{max(r.longest_irrelevant_run_before_last_relevant for r in prioritized)}.\n"
        "A simulation on synthetic labels, not a recall guarantee for your review.",
    ]
    transcript = "\n\n".join(panels) + "\n"
    save("transcript.txt", transcript.encode("utf-8"))
    checks["output_bytes"] = {name: len(content) for name, content in artifacts.items()}
    checks["sha256"] = {
        name: hashlib.sha256(content).hexdigest() for name, content in artifacts.items()
    }
    save("checks.json", (json.dumps(checks, indent=2) + "\n").encode("utf-8"))
    return transcript


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(run_demo(args.output_dir), end="")


if __name__ == "__main__":
    main()
