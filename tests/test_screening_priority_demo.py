"""Published prioritization illustrations and numbers must come from measured offline runs."""

import hashlib
import json
import re
import statistics
import sys
from pathlib import Path

import pytest
from PIL import Image
from scripts.create_screening_priority_gif import create_preview
from scripts.demo_screening_priority import OUTPUT_NAMES, SIMULATION_SEEDS, run_demo
from scripts.screening_priority_fixtures import collection_order, synthetic_papers

from storage.screening_priority import ScreeningPriority, simulate_screening

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs/assets"
GUIDE = ROOT / "docs/guides/SCREENING_PRIORITIZATION_GUIDE.md"


def test_demo_measures_read_only_ordering_and_renders_four_frames(tmp_path: Path) -> None:
    output = tmp_path / "demo"
    transcript = run_demo(output)
    assert {path.name for path in output.iterdir()} == set(OUTPUT_NAMES)
    assert transcript == (output / "transcript.txt").read_text()
    checks = json.loads((output / "checks.json").read_text())
    cold = ScreeningPriority.model_validate_json((output / "cold.json").read_bytes())
    ready = ScreeningPriority.model_validate_json((output / "ready.json").read_bytes())
    assert cold.model_status == "needs_include_and_exclude" and cold.feature_terms == 0
    assert ready.model_status == "ready" and len(ready.unsure_items) == 1
    assert ready.training_documents == checks["api_decisions"] == 12
    assert checks["api_matches_simulation"] and checks["python_matches_api"]
    assert checks["browser_matches_api"] and checks["collection_page_saved_order"]
    assert checks["form_decision_blank"] and checks["repeat_read_identical"]
    assert checks["reads_database_unchanged"] and checks["restart_identical"]
    assert checks["database_removed"]
    assert not any(checks["forbidden_call_counts"].values())
    simulation = checks["simulation"]
    assert simulation["prioritized_to_recall_target"] < simulation["default_to_recall_target"]
    assert simulation["median_prioritized"] < simulation["median_default"]
    for name, size in checks["output_bytes"].items():
        content = (output / name).read_bytes()
        assert len(content) == size
        assert checks["sha256"][name] == hashlib.sha256(content).hexdigest()
    html = (output / "priority.html").read_text()
    assert "READ-ONLY / SCREENING PRIORITIZATION" in html and "<form" not in html
    gif, still = tmp_path / "preview.gif", tmp_path / "preview.png"
    create_preview(output / "transcript.txt", gif, still)
    with Image.open(gif) as animation, Image.open(still) as preview:
        assert animation.n_frames == 4 and animation.info["loop"] == 0
        assert animation.size == preview.size == (1120, 540)
        assert animation.convert("RGB").tobytes() == preview.convert("RGB").tobytes()
        frames = set()
        for index in range(4):
            animation.seek(index)
            assert animation.info["duration"] == 3500
            frames.add(hashlib.sha256(animation.convert("RGB").tobytes()).hexdigest())
        assert len(frames) == 4
    with pytest.raises(FileExistsError):
        create_preview(output / "transcript.txt", gif, still)
    assert (ASSETS / "screening-priority.txt").read_text() == transcript


@pytest.mark.parametrize("kind", ["directory", "file", "symlink"])
def test_demo_never_overwrites_user_output(tmp_path: Path, kind: str) -> None:
    target = tmp_path / "existing"
    if kind == "directory":
        target.mkdir()
    elif kind == "file":
        target.write_text("Keep this")
    else:
        target.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        run_demo(target)
    if kind == "file":
        assert target.read_text() == "Keep this"


def test_demo_ignores_ambient_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = tmp_path / "user.sqlite3"
    sentinel.write_bytes(b"DO NOT OPEN")
    monkeypatch.setenv("SCHOLAR_RAG_DATABASE_PATH", str(sentinel))
    monkeypatch.setenv("SCHOLAR_RAG_DEFAULT_MODEL", "not-a-provider")
    monkeypatch.setenv("OPENAI_API_KEY", "unused-private-fixture")
    output = tmp_path / "demo"
    run_demo(output)
    assert sentinel.read_bytes() == b"DO NOT OPEN"
    assert all(b"unused-private-fixture" not in path.read_bytes() for path in output.iterdir())


@pytest.mark.parametrize("text", ["Not the executed demo", "x" * 16385])
def test_preview_refuses_wrong_or_oversized_transcript(tmp_path: Path, text: str) -> None:
    transcript = tmp_path / "transcript.txt"
    transcript.write_text(text)
    with pytest.raises(ValueError):
        create_preview(transcript, tmp_path / "preview.gif", tmp_path / "preview.png")
    assert not (tmp_path / "preview.gif").exists()
    assert not (tmp_path / "preview.png").exists()


def test_committed_preview_and_links_match_the_transcript(tmp_path: Path) -> None:
    gif, still = tmp_path / "preview.gif", tmp_path / "preview.png"
    create_preview(ASSETS / "screening-priority.txt", gif, still)
    assert gif.read_bytes() == (ASSETS / "screening-priority.gif").read_bytes()
    assert (ASSETS / "screening-priority.gif").stat().st_size < 800_000
    with Image.open(still) as actual, Image.open(ASSETS / "screening-priority.png") as saved:
        assert actual.size == saved.size
        assert actual.convert("RGB").tobytes() == saved.convert("RGB").tobytes()
    for document in (ROOT / "README.md", GUIDE):
        text = document.read_text()
        for name in ("screening-priority.png", "screening-priority.gif", "screening-priority.txt"):
            assert name in text, (document.name, name)


def test_guide_simulation_table_matches_a_fresh_simulation() -> None:
    papers = synthetic_papers()
    guide = GUIDE.read_text()
    prioritized, default = [], []
    for seed in SIMULATION_SEEDS:
        ordered = collection_order(papers, seed)
        fast = simulate_screening(ordered, strategy="most_likely_relevant_first")
        slow = simulate_screening(ordered, strategy="collection_order")
        prioritized.append(fast)
        default.append(slow)
        row = (
            f"| {seed} | {fast.papers_to_recall_target} | {slow.papers_to_recall_target} | "
            f"{fast.papers_to_find_all_relevant} | {slow.papers_to_find_all_relevant} | "
            f"{fast.work_saved_over_sampling:.2f} | {slow.work_saved_over_sampling:.2f} | "
            f"{fast.longest_irrelevant_run_before_last_relevant} | "
            f"{slow.longest_irrelevant_run_before_last_relevant} |"
        )
        assert row in guide, row
    median_fast = statistics.median(r.papers_to_recall_target for r in prioritized)
    median_slow = statistics.median(r.papers_to_recall_target for r in default)
    assert f"| Median | {median_fast:g} | {median_slow:g} |" in guide
    curves = {r.strategy: r.relevant_found for r in (prioritized[0], default[0])}
    for screened in (10, 20, 30, 50):
        fast_found = curves["most_likely_relevant_first"][screened - 1]
        slow_found = curves["collection_order"][screened - 1]
        assert f"| {screened} | {fast_found} | {slow_found} |" in guide


def test_standalone_python_guide_example_runs(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    blocks = re.findall(r"```python\n(.*?)```", GUIDE.read_text(), re.DOTALL)
    monkeypatch.setattr(sys, "argv", ["screening_priority_example.py"])
    exec(compile(blocks[0], str(GUIDE), "exec"), {})  # noqa: S102
    assert capsys.readouterr().out.splitlines() == [
        "needs_include_and_exclude",
        "ready ['candidate-grounded', 'candidate-battery']",
        "candidate-grounded 0.593: answers, cite, evidence, passages, questions",
        "collection_order 4 most_likely_relevant_first 3",
    ]
    assert "Expected output" in GUIDE.read_text()
