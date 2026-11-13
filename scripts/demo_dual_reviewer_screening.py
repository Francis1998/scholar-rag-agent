"""Execute offline dual-reviewer screening: blind decisions, conflicts, resolutions and kappa.

Writes only a new temporary database and the requested output directory. Two simulated
reviewers and a lead record fixture decisions through the real JSON API; the application
never decides, suggests or resolves anything itself.
"""

import argparse
import csv
import hashlib
import io
import json
import re
import socket
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch
from urllib.parse import quote, urlsplit

import httpx
from fastapi.testclient import TestClient

from api.application import create_app
from api.dependencies import AppContainer
from retrieval.models import Document
from scripts.demo_corpus_explorer import _no_work, browsing_guards
from scripts.demo_evidence_export import offline_settings
from storage.dual_screening import PairAgreement, agreement_line

PANEL_TITLES = (
    "1. Two reviewers screen the same papers blind",
    "2. Blinding lifts per paper, then the team unblinds",
    "3. An append-only resolution becomes the final decision",
    "4. Agreement report: Cohen's kappa, CI and PABAK",
)
OUTPUT_NAMES = (
    "r2-queue-blind.json",
    "conflicts-blind.json",
    "r2-member-blind.html",
    "conflicts.json",
    "resolution.json",
    "final-decisions.csv",
    "agreement.json",
    "agreement.md",
    "agreement.csv",
    "report.json",
    "team.html",
    "checks.json",
    "transcript.txt",
)
# (document ID, title, r1 decision, r2 decision); all original synthetic fixtures.
PAPERS = (
    ("grounded-qa-01", "Citation-grounded answers to clinical questions", "include", "include"),
    ("grounded-qa-02", "Retrieved passages cited inline by a QA system", "include", "include"),
    ("grounded-qa-03", "Evidence-linked summaries of trial reports", "include", "include"),
    ("grounded-qa-04", "Answer attribution to retrieved scientific text", "include", "include"),
    ("near-miss-05", "Citation recommendation for draft manuscripts", "include", "exclude"),
    ("off-topic-06", "Battery cathode coatings and capacity fade", "exclude", "exclude"),
    ("off-topic-07", "Protein folding with coarse-grained models", "exclude", "exclude"),
    ("off-topic-08", "Traffic forecasting with graph networks", "exclude", "exclude"),
    ("off-topic-09", "Music genre retrieval from audio features", "exclude", "exclude"),
    ("off-topic-10", "Crop yield estimation from satellite images", "exclude", "exclude"),
    ("near-miss-11", "Closed-book science QA with source checks", "exclude", "include"),
    ("near-miss-12", "Hallucination audits of translation models", "unsure", "exclude"),
)
REASONS = {
    "include": "Grounds answers in retrieved evidence: meets criterion C1.",
    "exclude": "E1: does not ground generated answers in cited evidence.",
    "unsure": "Abstract unclear about whether sources are cited; needs full text.",
}
RESOLUTIONS = (
    ("near-miss-05", "exclude", "E1 after discussion: recommends citations, answers nothing."),
    ("near-miss-11", "include", "Methods section confirms answers cite checked sources."),
)


def _post_json(response: httpx.Response) -> dict[str, Any]:
    response.raise_for_status()
    if response.headers["cache-control"] != "no-store":
        raise RuntimeError("A team or reviewer response lost its no-store header.")
    result: dict[str, Any] = response.json()
    return result


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

    counts: dict[str, int] = {}
    with TemporaryDirectory(prefix="scholar-dual-review-") as temporary, ExitStack() as stack:
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
                    document_id=identifier,
                    title=title,
                    text=f"Synthetic abstract for {title.lower()}.",
                    source="synthetic:dual-review",
                )
                for identifier, title, _, _ in PAPERS
            ],
            [],
        )
        collection = container.paper_collections.create(
            name="Synthetic grounded-answer review",
            document_ids=[identifier for identifier, *_ in PAPERS],
        )
        base = f"/collections/{collection.collection_id}"
        with TestClient(app) as client, browsing_guards(container) as guards, ExitStack() as extra:
            forbidden = {
                name: extra.enter_context(patch.object(target, method, side_effect=_no_work))
                for name, target, method in (
                    ("legacy_submit", container.paper_screening, "submit"),
                    ("transitions", container.event_log, "append_transition"),
                )
            }

            def decide(reviewer: str, identifier: str, decision: str) -> dict[str, Any]:
                return _post_json(
                    client.put(
                        f"{base}/reviewer-screening/{quote(identifier, safe='')}",
                        json={
                            "reviewer": reviewer,
                            "collection_revision": 1,
                            "expected_decision_revision": 0,
                            "decision": decision,
                            "reason": REASONS[decision],
                        },
                    )
                )

            def queue(reviewer: str) -> httpx.Response:
                response = client.get(
                    f"{base}/reviewer-screening",
                    params={"reviewer": reviewer, "collection_revision": 1, "limit": 100},
                )
                response.raise_for_status()
                return response

            r1 = [decide("r1", identifier, first) for identifier, _, first, _ in PAPERS]
            blind_queue = queue("r2")
            save("r2-queue-blind.json", blind_queue.content)
            blind_items = blind_queue.json()["items"]
            blocked = client.get(f"{base}/screening-conflicts", params={"collection_revision": 1})
            save("conflicts-blind.json", blocked.content)
            request_unblind = client.get(
                f"{base}/reviewer-screening",
                params={"reviewer": "r2", "collection_revision": 1, "blind": "false"},
            )
            member = client.get(
                "/explore/reviewer-screening",
                params={
                    "collection_id": collection.collection_id,
                    "collection_revision": 1,
                    "reviewer": "r2",
                    "document_id": "near-miss-05",
                },
            )
            member.raise_for_status()
            save("r2-member-blind.html", member.content)
            r2 = [decide("r2", PAPERS[0][0], PAPERS[0][3])]
            lifted = queue("r2").json()["items"]
            r2 += [decide("r2", identifier, second) for identifier, _, _, second in PAPERS[1:]]
            protocol = _post_json(
                client.put(
                    f"{base}/screening-protocol",
                    json={
                        "blind": False,
                        "exclusion_reasons_must_match": False,
                        "expected_protocol_revision": 0,
                        "changed_by": "lead",
                    },
                )
            )
            conflicts_response = client.get(
                f"{base}/screening-conflicts", params={"collection_revision": 1}
            )
            save("conflicts.json", conflicts_response.content)
            conflicts = _post_json(conflicts_response)
            before = _post_json(
                client.get(f"{base}/screening-agreement", params={"collection_revision": 1})
            )
            items = {item["document_id"]: item for item in conflicts["items"]}
            resolutions = []
            for identifier, decision, rationale in RESOLUTIONS:
                response = client.put(
                    f"{base}/screening-resolutions/{quote(identifier, safe='')}",
                    json={
                        "resolver": "lead",
                        "collection_revision": 1,
                        "expected_resolution_revision": items[identifier]["resolution_revision"],
                        "expected_basis": items[identifier]["expected_basis"],
                        "decision": decision,
                        "rationale": rationale,
                    },
                )
                if not resolutions:
                    save("resolution.json", response.content)
                resolutions.append(_post_json(response))
            downloads = {}
            for name, route, format in (
                ("final-decisions.csv", "screening-final-decisions", "csv"),
                ("agreement.json", "screening-agreement", "json"),
                ("agreement.md", "screening-agreement", "markdown"),
                ("agreement.csv", "screening-agreement", "csv"),
                ("report.json", "screening-report", "json"),
            ):
                response = client.get(
                    f"{base}/{route}", params={"collection_revision": 1, "format": format}
                )
                response.raise_for_status()
                save(name, response.content)
                downloads[name] = response
            team = client.get(
                "/explore/screening-conflicts",
                params={"collection_id": collection.collection_id, "collection_revision": 1},
            )
            team.raise_for_status()
            if urlsplit(str(team.url)).path != "/explore/screening-conflicts":
                raise RuntimeError("Navigation left the local team page.")
            save("team.html", team.content)
            r2_after = queue("r2").json()["items"]
            counts.update({name: guard.call_count for name, guard in guards.items()})
            counts.update({name: guard.call_count for name, guard in forbidden.items()})
        restarted = create_app(settings)
        with TestClient(restarted) as client:
            response = client.get(f"{base}/screening-agreement", params={"collection_revision": 1})
            restart_identical = response.content == artifacts["agreement.json"]
        counts.update({f"socket:{name}": guard.call_count for name, guard in network.items()})
    agreement = json.loads(artifacts["agreement.json"])
    report = json.loads(artifacts["report.json"])["dual_review"]
    primary, binary = agreement["pairs"]
    rows = list(csv.DictReader(io.StringIO(artifacts["final-decisions.csv"].decode("utf-8"))))
    resolved_rows = [row for row in rows if row["document_id"] == "'near-miss-05"]
    member_html = artifacts["r2-member-blind.html"].decode("utf-8")
    team_html = artifacts["team.html"].decode("utf-8")
    lines = re.findall(r'class="agreement-line">([^<]+)<', team_html)
    r1_reasons = {review["reason"] for review in r1}
    original_decisions = {
        (review["reviewer"], review["document_id"], review["revision"]) for review in r1 + r2
    }
    after_decisions = {
        (entry["reviewer"], item["document_id"], entry["revision"])
        for item in r2_after
        for entry in item["other_decisions"]
    } | {("r2", item["document_id"], item["review"]["revision"]) for item in r2_after}
    checks: dict[str, Any] = {
        "synthetic_only": True,
        "papers": len(PAPERS),
        "blind_items_hidden": sum(item["others_hidden"] for item in blind_items),
        "blind_queue_has_r1_reasons": any(
            reason in artifacts["r2-queue-blind.json"].decode("utf-8") for reason in r1_reasons
        ),
        "team_view_while_blind": [blocked.status_code, blocked.json()["detail"]["code"]],
        "request_unblind": [request_unblind.status_code, request_unblind.json()["detail"]["code"]],
        "member_page_hidden": 'id="others-hidden"' in member_html
        and not any(reason in member_html for reason in r1_reasons),
        "lifted_after_first_decision": sum(not item["others_hidden"] for item in lifted),
        "r2_blinded_flags": sorted({review["blinded"] for review in r2}),
        "protocol_revision": protocol["revision"],
        "conflicts": [
            [item["document_id"], *(entry["decision"] for entry in item["decisions"])]
            for item in conflicts["items"]
        ],
        "resolution_revisions": [resolution["revision"] for resolution in resolutions],
        "decisions_unchanged": original_decisions == after_decisions,
        "kappa_unchanged_by_resolutions": before["pairs"] == agreement["pairs"],
        "primary_kappa": primary["statistics"]["kappa"],
        "binary_kappa": binary["statistics"]["kappa"],
        "final_counts": agreement["final_counts"],
        "conflict_counts": agreement["conflicts"],
        "report_matches_agreement": report["final_counts"] == agreement["final_counts"]
        and report["conflicts"] == agreement["conflicts"],
        "resolved_csv": [
            [
                _cell(row[field])
                for field in ("reviewer", "reviewer_decision", "final_status", "final_decision")
            ]
            for row in resolved_rows
        ],
        "browser_matches_api": lines == [agreement_line_text(pair) for pair in agreement["pairs"]],
        "restart_identical": restart_identical,
        "database_removed": not database.exists(),
        "forbidden_call_counts": counts,
    }
    required = (
        checks["blind_items_hidden"] == len(PAPERS),
        not checks["blind_queue_has_r1_reasons"],
        checks["team_view_while_blind"] == [409, "screening_blind_mode_on"],
        checks["request_unblind"] == [409, "screening_blind_mode_on"],
        checks["member_page_hidden"],
        checks["lifted_after_first_decision"] == 1,
        checks["r2_blinded_flags"] == [True],
        checks["decisions_unchanged"],
        checks["kappa_unchanged_by_resolutions"],
        checks["report_matches_agreement"],
        checks["browser_matches_api"],
        checks["restart_identical"],
        checks["database_removed"],
        len(checks["conflicts"]) == 3,
        not any(counts.values()),
    )
    if not all(required):
        raise RuntimeError(f"Measured dual-review checks failed: {checks}")
    stats, binary_stats = primary["statistics"], binary["statistics"]
    final = agreement["final_counts"]
    panels = [
        f"{PANEL_TITLES[0]}\n"
        f"{len(PAPERS)} synthetic papers; r1 records {len(r1)} decisions via PUT "
        ".../reviewer-screening/{id}.\n"
        'GET .../reviewer-screening?reviewer=r2 -> "blind": true, '
        f'{checks["blind_items_hidden"]}/{len(PAPERS)} "others_hidden": true\n'
        '"other_decisions": null, "final": null; r1 reasons in r2\'s JSON: '
        f"{checks['blind_queue_has_r1_reasons']}\n"
        f"GET .../screening-conflicts while blind -> HTTP {blocked.status_code} "
        f'"{blocked.json()["detail"]["code"]}"\n'
        f"A blind=false request -> HTTP {request_unblind.status_code}; r2's browser form "
        "says other\nreviewers are hidden until r2 records a decision on that paper.",
        f"{PANEL_TITLES[1]}\n"
        f"After r2's first decision: {checks['lifted_after_first_decision']} member shows "
        f"r1's decision, {len(PAPERS) - checks['lifted_after_first_decision']} stay hidden.\n"
        f'r2 records {len(r2)} decisions, each "blinded": true (first decision, blind mode '
        "on).\n"
        'PUT .../screening-protocol {"blind": false, "changed_by": "lead"} -> revision '
        f"{protocol['revision']}\n"
        f"GET .../screening-conflicts -> {conflicts['counts']['disagreements']} "
        f"disagreements, {conflicts['counts']['resolved']} resolved:\n"
        + "\n".join(
            f"  {identifier}: r1 {first} / r2 {second}"
            for identifier, first, second in checks["conflicts"]
        ),
        f"{PANEL_TITLES[2]}\n"
        "PUT .../screening-resolutions/near-miss-05 by lead: exclude -> revision "
        f"{resolutions[0]['revision']}\n"
        "Basis: "
        + ", ".join(
            f"{entry['reviewer']} rev {entry['revision']} {entry['decision']}"
            for entry in resolutions[0]["basis"]
        )
        + f"; reviewer decisions unchanged: {checks['decisions_unchanged']}\n"
        "near-miss-11 resolved include; near-miss-12 (r1 unsure / r2 exclude) left "
        "unresolved.\n"
        "Final CSV, near-miss-05: "
        + "; ".join(f"{reviewer} {decision}" for reviewer, decision, _, _ in checks["resolved_csv"])
        + f" -> {checks['resolved_csv'][0][2]} {checks['resolved_csv'][0][3]}\n"
        f"Final include {final['final_include']}, final exclude {final['final_exclude']}, "
        f"needs review {final['needs_review']} ({agreement['conflicts']['unresolved']} "
        "unresolved conflict)\n"
        "Existing progress report gains a dual_review section with the same counts: "
        f"{checks['report_matches_agreement']}",
        f"{PANEL_TITLES[3]}\n"
        f"r1 vs r2, include/exclude/unsure, n {stats['compared']}: p_o "
        f"{stats['observed_agreement']:.3f}, p_e {stats['expected_agreement']:.3f}\n"
        f"kappa {stats['kappa']:.3f} (95% CI {stats['kappa_ci95_low']:.3f} to "
        f"{stats['kappa_ci95_high']:.3f}), {stats['landis_koch_band']} (Landis-Koch, "
        "arbitrary)\n"
        f"PABAK {stats['pabak']:.3f}; unsure counted as include: kappa "
        f"{binary_stats['kappa']:.3f}, PABAK {binary_stats['pabak']:.3f}\n"
        "SE: Fleiss-Cohen-Everitt 1969 (Wald CI); resolutions changed kappa: "
        f"{not checks['kappa_unchanged_by_resolutions']}\n"
        f"Unresolved conflicts: {agreement['conflicts']['unresolved']}; restarted app "
        f"returned identical JSON: {restart_identical}\n"
        f"Forbidden calls: {sum(counts.values())} (no models, retrieval or network). "
        "Synthetic data only.",
    ]
    transcript = "\n\n".join(panels) + "\n"
    save("transcript.txt", transcript.encode("utf-8"))
    checks["output_bytes"] = {name: len(content) for name, content in artifacts.items()}
    checks["sha256"] = {
        name: hashlib.sha256(content).hexdigest() for name, content in artifacts.items()
    }
    save("checks.json", (json.dumps(checks, indent=2) + "\n").encode("utf-8"))
    return transcript


def agreement_line_text(pair: dict[str, Any]) -> str:
    """The exact line the browser renders for one pair, rebuilt from the JSON report."""
    return agreement_line(PairAgreement.model_validate(pair))


def _cell(value: str) -> str:
    """Undo the reversible apostrophe-prefix-v1 encoding of one CSV text cell."""
    return value.removeprefix("'")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(run_demo(args.output_dir), end="")


if __name__ == "__main__":
    main()
