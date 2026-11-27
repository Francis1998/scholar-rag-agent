"""Execute offline screening stopping rules: SAFE phases 1-2, a seeded sample and buscar.

Writes only a new temporary database and the requested output directory. A simulated
reviewer records synthetic fixture labels through the real JSON API, choosing papers from
the real prioritized queue; the application never decides, excludes or stops anything.
The collection ID is fixed so the seeded sample order, which hashes that ID, is
reproducible in the committed transcript.
"""

import argparse
import hashlib
import json
import re
import socket
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch
from urllib.parse import quote, urlencode
from uuid import UUID

import httpx
from fastapi.testclient import TestClient

from api.application import create_app
from api.dependencies import AppContainer
from retrieval.models import Document
from scripts.demo_corpus_explorer import _no_work, browsing_guards
from scripts.demo_evidence_export import offline_settings
from scripts.screening_priority_fixtures import (
    RELEVANT_PAPERS,
    TOTAL_PAPERS,
    collection_order,
    synthetic_papers,
)
from storage import paper_collections

MAIN_SEED = 5
SAMPLE_SEED = 20261124
PILOT_SIZE = 10
FIXED_COLLECTION = UUID("2026112400000000000000000000c0de")
PANEL_TITLES = (
    "1. SAFE phases 1-2: random pilot, then prioritized screening",
    "2. A seeded random sample of the unscreened remainder",
    "3. buscar: exact p-value after each sampled decision",
    "4. Advisory only: evidence for your decision, never a stop",
)
OUTPUT_NAMES = (
    "checkpoint.json",
    "sample-start.json",
    "sample-final.json",
    "stopping.html",
    "checks.json",
    "transcript.txt",
)
REASON = "Simulated reviewer: synthetic fixture label for the offline demo only."


def _json(response: httpx.Response) -> dict[str, Any]:
    response.raise_for_status()
    if response.headers["cache-control"] != "no-store":
        raise RuntimeError("A stopping or screening response lost its no-store header.")
    result: dict[str, Any] = response.json()
    return result


def _fraction(text: str) -> str:
    return "0" if text.startswith("0/") else "1" if text == "1/1" else text


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
    ordered = [paper.document_id for paper in collection_order(papers, MAIN_SEED)]
    counts: dict[str, int] = {}
    with (
        TemporaryDirectory(prefix="scholar-screening-stopping-") as temporary,
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
                    source="synthetic:screening-stopping",
                )
                for paper in papers
            ],
            [],
        )
        with patch.object(paper_collections, "uuid4", return_value=FIXED_COLLECTION):
            collection = container.paper_collections.create(
                name="Synthetic grounded-answer review", document_ids=ordered
            )
        base = f"/collections/{collection.collection_id}"
        keys = sorted(identifier for identifier in ordered if labels[identifier])[:2]
        with TestClient(app) as client, browsing_guards(container) as guards, ExitStack() as extra:
            forbidden = {
                name: extra.enter_context(patch.object(target, method, side_effect=_no_work))
                for name, target, method in (
                    ("transitions", container.event_log, "append_transition"),
                )
            }

            def decide(identifier: str) -> None:
                _json(
                    client.put(
                        f"{base}/screening/{quote(identifier, safe='')}",
                        json={
                            "collection_revision": 1,
                            "expected_decision_revision": 0,
                            "decision": "include" if labels[identifier] else "exclude",
                            "reason": REASON,
                        },
                    )
                )

            for identifier in ordered[:PILOT_SIZE]:
                decide(identifier)
            pilot_relevant = sum(labels[identifier] for identifier in ordered[:PILOT_SIZE])
            estimate = pilot_relevant * TOTAL_PAPERS // PILOT_SIZE
            safe = [("estimated_relevant", str(estimate))] + [
                ("key_document_id", key) for key in keys
            ]

            def assess(extra_pairs: tuple[tuple[str, str], ...] = ()) -> httpx.Response:
                query = urlencode([("collection_revision", "1"), *safe, *extra_pairs])
                return client.get(f"{base}/screening-stopping?{query}")

            while True:
                checkpoint_response = assess()
                checkpoint = _json(checkpoint_response)
                if checkpoint["safe_checkpoint"]["status"] == "met":
                    break
                priority = _json(
                    client.get(f"{base}/screening-priority", params={"collection_revision": 1})
                )
                decide(priority["items"][0]["document_id"])
            save("checkpoint.json", checkpoint_response.content)
            start = checkpoint["recall_test"]
            sample = (
                ("sample_seed", str(SAMPLE_SEED)),
                ("sample_after", str(start["sample_after"])),
                ("sample_urn_sha256", start["urn_sha256"]),
            )
            before_read = database.read_bytes()
            drawn_response = assess(sample)
            drawn = _json(drawn_response)["recall_test"]
            unchanged_by_read = database.read_bytes() == before_read
            save("sample-start.json", drawn_response.content)
            progression = [(0, 0, drawn["p_value_exact"])]
            final_response = drawn_response
            for identifier in drawn["next_document_ids"]:
                decide(identifier)
                final_response = assess(sample)
                test = _json(final_response)["recall_test"]
                progression.append(
                    (test["sample_size"], test["sample_relevant"], test["p_value_exact"])
                )
                if test["status"] == "met":
                    break
            save("sample-final.json", final_response.content)
            final = _json(final_response)
            page_query = urlencode(
                [
                    ("collection_id", collection.collection_id),
                    ("collection_revision", "1"),
                    ("estimated_relevant", str(estimate)),
                    ("key_document_ids", "\r\n".join(keys)),
                    *sample,
                ]
            )
            page = client.get(f"/explore/screening-stopping?{page_query}")
            page.raise_for_status()
            save("stopping.html", page.content)
            posted = client.post(f"{base}/screening-stopping?collection_revision=1")
            counts.update({name: guard.call_count for name, guard in guards.items()})
            counts.update({name: guard.call_count for name, guard in forbidden.items()})
        restarted = create_app(settings)
        with TestClient(restarted) as client:
            query = urlencode([("collection_revision", "1"), *safe, *sample])
            again = client.get(f"{base}/screening-stopping?{query}")
            restart_identical = again.content == final_response.content
        counts.update({f"socket:{name}": guard.call_count for name, guard in network.items()})
    return _report(
        output_dir,
        save,
        artifacts,
        labels,
        pilot_relevant,
        estimate,
        checkpoint,
        drawn,
        progression,
        final,
        {
            "unchanged_by_read": unchanged_by_read,
            "restart_identical": restart_identical,
            "database_removed": not database.exists(),
            "post_status": [posted.status_code, posted.json()["detail"]["code"]],
            "forbidden_call_counts": counts,
        },
    )


def _report(
    output_dir: Path,
    save: Callable[[str, bytes], None],
    artifacts: dict[str, bytes],
    labels: dict[str, bool],
    pilot_relevant: int,
    estimate: int,
    checkpoint: dict[str, Any],
    drawn: dict[str, Any],
    progression: list[tuple[int, int, str]],
    final: dict[str, Any],
    measured: dict[str, Any],
) -> str:
    html = artifacts["stopping.html"].decode("utf-8")
    shown = {
        name: re.search(rf'id="{name}">([^<]+)<', html).group(1)  # type: ignore[union-attr]
        for name in ("consecutive-status", "safe-status", "recall-status", "p-value-exact")
    }
    test = final["recall_test"]
    rule = checkpoint["consecutive_irrelevant"]
    criteria = {item["name"]: item["status"] for item in checkpoint["safe_checkpoint"]["criteria"]}
    found = final["counts"]["include"]
    screened_before = checkpoint["timeline"]["screened"]
    prioritized_found = checkpoint["counts"]["include"]
    total_screened = final["timeline"]["screened"]
    checks: dict[str, Any] = {
        "synthetic_only": True,
        "papers": len(labels),
        "relevant_papers": sum(labels.values()),
        "pilot": [PILOT_SIZE, pilot_relevant, estimate],
        "checkpoint_screened": screened_before,
        "checkpoint_found": prioritized_found,
        "checkpoint_run": [rule["current_run"], rule["threshold"]],
        "checkpoint_criteria": criteria,
        "sample": [drawn["sample_after"], drawn["urn_size"], drawn["h0_min_relevant_in_urn"]],
        "projection": drawn["additional_irrelevant_needed"],
        "progression": progression,
        "final_status": test["status"],
        "final_p": test["p_value_exact"],
        "final_bounds": [test["recall_estimate"], test["recall_lower_bound"]],
        "screened_total": total_screened,
        "true_recall": found / RELEVANT_PAPERS,
        "order": [final["timeline"]["order_reliable"], final["timeline"]["tied_positions"]],
        "browser_matches_api": shown
        == {
            "consecutive-status": final["consecutive_irrelevant"]["status"],
            "safe-status": final["safe_checkpoint"]["status"],
            "recall-status": test["status"],
            "p-value-exact": test["p_value_exact"],
        },
        **measured,
    }
    required = (
        checkpoint["safe_checkpoint"]["status"] == "met",
        drawn["status"] == "not_met" and drawn["sample_size"] == 0,
        test["status"] == "met",
        test["recall_lower_bound"] <= found / RELEVANT_PAPERS,
        checks["browser_matches_api"],
        checks["order"] == [True, 0],
        measured["unchanged_by_read"],
        measured["restart_identical"],
        measured["database_removed"],
        measured["post_status"] == [405, "screening_stopping_method_not_allowed"],
        not any(measured["forbidden_call_counts"].values()),
    )
    if not all(required):
        raise RuntimeError(f"Measured stopping-rule checks failed: {checks}")
    sampled = test["sample_size"]
    shown_steps = [step for step in progression if step[0] % 10 == 0 or step[0] == sampled]
    panels = [
        f"{PANEL_TITLES[0]}\n"
        f"{len(labels)} synthetic papers ({sum(labels.values())} relevant); a simulated reviewer "
        "PUTs fixture labels.\n"
        f"Phase 1: first {PILOT_SIZE} in saved (shuffled) order -> {pilot_relevant} relevant; "
        f"RR_T = {pilot_relevant}/{PILOT_SIZE} x {len(labels)} = {estimate}.\n"
        "Phase 2: GET .../screening-priority rank 1 -> PUT decision -> GET "
        ".../screening-stopping\n"
        f"SAFE checkpoint met after {screened_before} decisions: {prioritized_found} of "
        f"{sum(labels.values())} relevant found.\n"
        f"  key papers {criteria['key_papers_included']}; {screened_before} >= 2 x {estimate} "
        f"{criteria['screened_twice_estimated_relevant']}; >= 10% "
        f"{criteria['minimum_screened_percent']}\n"
        f"  {rule['current_run']} consecutive excludes >= threshold {rule['threshold']} "
        f"(10% of {len(labels)}): {rule['status']}\n"
        f"Nothing was excluded: {checkpoint['timeline']['pending']} members are still "
        "unscreened.",
        f"{PANEL_TITLES[1]}\n"
        f"GET .../screening-stopping -> recall_test not_started: sample_after="
        f"{drawn['sample_after']}, urn {drawn['urn_size']}\n"
        f"Seed {SAMPLE_SEED} chosen before viewing; urn_sha256 {drawn['urn_sha256'][:16]}...\n"
        "Sample order = SHA-256 of [version, seed, collection, revision, document ID]\n"
        "Screen next: " + ", ".join(drawn["next_document_ids"][:3]) + ", ...\n"
        f"H0 'recall < 0.95': >= {drawn['h0_min_relevant_in_urn']} relevant left in the urn "
        f"(K_hat); p = {_fraction(drawn['p_value_exact'])} before sampling\n"
        f"Projection: if the next {drawn['additional_irrelevant_needed']} sampled records are "
        "irrelevant, p < 0.05\n"
        f"Reading the sample left the database unchanged: {measured['unchanged_by_read']}",
        f"{PANEL_TITLES[2]}\n"
        "Sampled  Relevant  p = P(X <= k | N_urn, K_hat, n), exact\n"
        + "\n".join(
            f"{n:>7}  {k:>8}  {_fraction(p)} = {float(sum_fraction(p)):.3f}"
            for n, k, p in shown_steps
        )
        + f"\nMet at n = {sampled}: p < 0.05, reject H0 'recall < 0.95' (95% confidence).\n"
        f"Recall estimate {test['recall_estimate']:.3f}; lower bound "
        f"{test['recall_lower_bound']:.3f}; screened {total_screened} of {len(labels)}.",
        f"{PANEL_TITLES[3]}\n"
        "Order: first recorded decision time at this revision; reliable "
        f"{final['timeline']['order_reliable']}, ties {final['timeline']['tied_positions']}\n"
        f"Browser page statuses and exact p equal the JSON: {checks['browser_matches_api']}\n"
        f"POST .../screening-stopping -> HTTP {measured['post_status'][0]} "
        f'"{measured["post_status"][1]}"\n'
        f"Restarted app returned identical JSON: {measured['restart_identical']}\n"
        f"Demo-only ground truth: {found}/{RELEVANT_PAPERS} relevant found, true recall "
        f"{found / RELEVANT_PAPERS:.2f} >= bound {test['recall_lower_bound']:.3f}\n"
        f"{len(labels) - total_screened} papers never screened; deciding to stop stays "
        "a human, documented choice.\n"
        f"Forbidden calls: {sum(measured['forbidden_call_counts'].values())} (no models, "
        "retrieval or network). Synthetic data only.",
    ]
    transcript = "\n\n".join(panels) + "\n"
    save("transcript.txt", transcript.encode("utf-8"))
    checks["output_bytes"] = {name: len(content) for name, content in artifacts.items()}
    checks["sha256"] = {
        name: hashlib.sha256(content).hexdigest() for name, content in artifacts.items()
    }
    save("checks.json", (json.dumps(checks, indent=2) + "\n").encode("utf-8"))
    return transcript


def sum_fraction(text: str) -> float:
    numerator, denominator = text.split("/")
    return int(numerator) / int(denominator)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(run_demo(args.output_dir), end="")


if __name__ == "__main__":
    main()
