"""Executed synthetic workflow and pixel-based reproducibility, without browser tooling."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image
from scripts.create_run_comparison_workbench_gif import create_preview
from scripts.demo_run_comparison_workbench import ComparisonPage, run_demo

from agent.comparison_models import RunComparison
from agent.evidence import EvidenceBundle
from agent.run_comparison import compare_bundles
from tests.test_run_comparison import ComparisonAPI, _save
from tests.test_run_comparison import comparison_api as comparison_api

ASSETS = Path(__file__).resolve().parents[1] / "docs/assets"


def test_demo_measures_native_api_python_and_restart_parity(tmp_path: Path) -> None:
    output = tmp_path / "demo"
    transcript = run_demo(output)
    checks = json.loads((output / "checks.json").read_text())
    assert checks["events_before"] == checks["events_after"] == 16
    assert checks["remaining_corpus_chunks"] == checks["external_calls"] == 0
    assert not any(checks["forbidden_call_counts"].values())
    for name in (
        "native_python_api_parity",
        "exact_history_return",
        "restart_comparison_identical",
        "restart_source_identical",
        "database_bytes_unchanged_during_reads",
        "temporary_database_removed",
    ):
        assert checks[name] is True
    assert checks["missing_candidate_status"] == 404 and checks["same_run_changes"] is False
    assert checks["counts"]["shared_count"] == checks["counts"]["added_count"] == 1
    assert checks["counts"]["removed_count"] == 1
    comparison = RunComparison.model_validate_json((output / "comparison.json").read_bytes())
    bundles = [
        EvidenceBundle.model_validate_json((output / f"{side}.json").read_bytes())
        for side in ("baseline", "candidate")
    ]
    assert compare_bundles(*bundles) == comparison
    assert all(bundle.generation.provider == "fake" for bundle in bundles)
    page = ComparisonPage((output / "comparison.html").read_text())
    assert json.loads(page.values["evidence-statistics"][0]) == checks["counts"]
    assert transcript == (output / "transcript.txt").read_text()
    assert transcript == (ASSETS / "saved-run-comparison-workbench.txt").read_text()
    assert run_demo(tmp_path / "repeat") == transcript
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_demo(output)
    assert (output / "transcript.txt").read_text() == transcript


def test_demo_cli_ignores_ambient_credentials_and_settings(tmp_path: Path) -> None:
    sentinel = tmp_path / "private.sqlite3"
    sentinel.write_bytes(b"unchanged sentinel")
    (tmp_path / ".env").write_text("SCHOLAR_RAG_MAX_HOPS=invalid\n")
    output = tmp_path / "demo"
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "scripts.demo_run_comparison_workbench",
            "--output-dir",
            str(output),
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "SCHOLAR_RAG_DATABASE_PATH": str(sentinel),
            "SCHOLAR_RAG_DEFAULT_MODEL": "anthropic",
            "SCHOLAR_RAG_MAX_HOPS": "invalid",
            "OPENAI_API_KEY": "synthetic-unused",
            "ANTHROPIC_API_KEY": "synthetic-unused",
            "GEMINI_API_KEY": "synthetic-unused",
            "MOONSHOT_API_KEY": "synthetic-unused",
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert sentinel.read_bytes() == b"unchanged sentinel"
    assert not (output / "synthetic.sqlite3").exists()
    assert all(b"synthetic-unused" not in path.read_bytes() for path in output.iterdir())


def test_decoded_gif_and_opt_in_still_match_reproduced_pixels(tmp_path: Path) -> None:
    transcript = ASSETS / "saved-run-comparison-workbench.txt"
    gif, still = tmp_path / "demo.gif", tmp_path / "demo.png"
    create_preview(transcript, gif, still)
    with (
        Image.open(gif) as actual,
        Image.open(ASSETS / "saved-run-comparison-workbench.gif") as expected,
    ):
        assert actual.format == expected.format == "GIF"
        assert actual.n_frames == expected.n_frames == 4
        assert actual.size == expected.size == (1120, 540)
        assert actual.info["loop"] == expected.info["loop"] == 0
        frames = set()
        for number in range(4):
            actual.seek(number)
            expected.seek(number)
            assert actual.info["duration"] == expected.info["duration"] == 3500
            pixels = actual.convert("RGB").tobytes()
            assert pixels == expected.convert("RGB").tobytes()
            frames.add(pixels)
        assert len(frames) == 4
        actual.seek(0)
        first_pixels = actual.convert("RGB").tobytes()
    for path in (still, ASSETS / "saved-run-comparison-workbench.png"):
        with Image.open(path) as image:
            assert image.format == "PNG" and image.size == (1120, 540)
            assert image.convert("RGB").tobytes() == first_pixels
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        create_preview(transcript, gif, still)


def test_demo_and_renderer_refuse_symlinks_invalid_transcripts_and_aliased_outputs(
    tmp_path: Path,
) -> None:
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_demo(alias)
    transcript = tmp_path / "transcript.txt"
    for content in ("wrong", "x" * 16_385):
        transcript.write_text(content)
        with pytest.raises(ValueError, match="four bounded measured"):
            create_preview(transcript, tmp_path / "new.gif", tmp_path / "new.png")
    assert not (tmp_path / "new.gif").exists()
    with pytest.raises(ValueError, match="different paths"):
        create_preview(transcript, tmp_path / "same", tmp_path / "same")


def test_documented_python_example_reads_exact_ids_without_startup(
    comparison_api: ComparisonAPI,
    tmp_path: Path,
) -> None:
    api = comparison_api
    _save(api, "baseline/\u7814")
    _save(api, "candidate\0")
    guide = ASSETS.parent / "guides/SAVED_RUN_COMPARISON_WORKBENCH_GUIDE.md"
    snippet = guide.read_text().split("```python\n", 1)[1].split("```", 1)[0]
    snippet = (
        snippet.replace('"/absolute/path/existing-runs.sqlite3"', repr(str(api.path)))
        .replace('"copy-baseline-id"', repr("baseline/\u7814"))
        .replace('"copy-candidate-id"', repr("candidate\0"))
    )
    before = api.path.read_bytes()
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-c", snippet],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    comparison = RunComparison.model_validate_json(completed.stdout)
    assert comparison.baseline.run_id == "baseline/\u7814"
    assert comparison.candidate.run_id == "candidate\0"
    assert comparison.any_changes is False
    assert api.path.read_bytes() == before
    assert not (tmp_path / ".scholar-rag-agent.sqlite3").exists()
