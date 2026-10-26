"""Offline demo, documentation and opt-in measured-illustration contracts."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image
from scripts.create_human_screening_workbench_gif import create_preview
from scripts.demo_human_screening_workbench import REASONS, SELECTED_ID, run_demo

from storage.paper_screening import ScreeningHistory
from tests.test_corpus_explorer import ExplorerAPI, ExplorerDOM, values
from tests.test_corpus_explorer import api as api

ASSETS = Path(__file__).resolve().parents[1] / "docs/assets"


def test_demo_proves_persistence_exact_reasons_and_rejected_writes(tmp_path: Path) -> None:
    output = tmp_path / "demo"
    transcript = run_demo(output)
    checks = json.loads((output / "checks.json").read_text())
    assert checks["successful_posts"] == 3
    assert checks["submit_service_attempts"] == 4
    assert checks["redirect_statuses"] == [303, 303, 303]
    assert checks["rejected_statuses"] == {"cross_origin": 403, "invalid": 422, "stale": 409}
    assert checks["events"] == 0 and not any(checks["forbidden_call_counts"].values())
    for name in (
        "rejected_posts_database_unchanged",
        "get_database_unchanged",
        "python_api_history_equal",
        "reasons_exact",
        "collection_unchanged",
        "restart_html_identical",
        "restart_database_unchanged",
        "exact_opaque_source",
        "temporary_database_removed",
    ):
        assert checks[name] is True
    history = ScreeningHistory.model_validate_json((output / "history.json").read_bytes())
    assert history.document_id == SELECTED_ID
    assert [review.reason for review in history.items] == list(reversed(REASONS))
    assert [review.revision for review in history.items] == [3, 2, 1]
    root = ExplorerDOM((output / "restarted.html").read_text()).root
    assert values(root, "history-revision") == ["3", "2", "1"]
    assert values(root, "latest-review-decision") == ["unsure"]
    assert transcript == (output / "transcript.txt").read_text()
    assert transcript == (ASSETS / "human-screening-workbench.txt").read_text()
    assert run_demo(tmp_path / "repeat") == transcript
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_demo(output)


def test_cli_ignores_ambient_credentials_dotenv_and_database(tmp_path: Path) -> None:
    sentinel = tmp_path / "private.sqlite3"
    sentinel.write_bytes(b"unchanged sentinel")
    (tmp_path / ".env").write_text("SCHOLAR_RAG_MAX_HOPS=invalid\n")
    output = tmp_path / "demo"
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "scripts.demo_human_screening_workbench",
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
    assert all(b"synthetic-unused" not in path.read_bytes() for path in output.iterdir())


def test_reproduced_gif_and_opt_in_still_have_identical_pixels(tmp_path: Path) -> None:
    gif, still = tmp_path / "demo.gif", tmp_path / "demo.png"
    create_preview(ASSETS / "human-screening-workbench.txt", gif, still)
    with Image.open(gif) as actual, Image.open(ASSETS / "human-screening-workbench.gif") as saved:
        assert actual.size == saved.size == (1120, 540)
        assert actual.n_frames == saved.n_frames == 4
        frames = set()
        for index in range(4):
            actual.seek(index)
            saved.seek(index)
            assert actual.info["duration"] == saved.info["duration"] == 3500
            pixels = actual.convert("RGB").tobytes()
            assert pixels == saved.convert("RGB").tobytes()
            frames.add(pixels)
        assert len(frames) == 4
        actual.seek(0)
        first = actual.convert("RGB").tobytes()
    for path in (still, ASSETS / "human-screening-workbench.png"):
        with Image.open(path) as image:
            assert image.format == "PNG" and image.convert("RGB").tobytes() == first
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        create_preview(ASSETS / "human-screening-workbench.txt", gif, still)


def test_renderer_and_demo_refuse_overwrites_and_invalid_transcripts(tmp_path: Path) -> None:
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


def test_guide_and_indexes_label_preview_as_measured_not_browser_footage() -> None:
    root = ASSETS.parents[1]
    guide = root / "docs/guides/HUMAN_SCREENING_WORKBENCH_GUIDE.md"
    text = guide.read_text()
    for phrase in (
        "## Native browser workflow",
        "## API contract",
        "## Python",
        "## Portfolio",
        "## Reproduce",
        "https://github.com/asreview/asreview",
        "workflow inspiration only",
        "not browser footage",
        "65536",
        "same-origin",
        "DNS rebinding",
        "no authentication",
        "reason",
        "303",
    ):
        assert phrase in text
    for path in (root / "README.md", root / "docs/README.md", root / "docs/DEMO.md"):
        assert guide.name in path.read_text()
    assert "](../assets/human-screening-workbench.gif)" in text
    assert "human-screening-workbench.png" in text
    assert "human-screening-workbench.txt" in text


def test_documented_python_example_records_one_exact_synthetic_opinion(
    api: ExplorerAPI, tmp_path: Path
) -> None:
    api.seed(SELECTED_ID)
    collection = api.container.paper_collections.create(
        name="Python documentation fixture", document_ids=[SELECTED_ID]
    )
    guide = ASSETS.parent / "guides/HUMAN_SCREENING_WORKBENCH_GUIDE.md"
    snippet = guide.read_text().split("```python\n", 1)[1].split("```", 1)[0]
    snippet = (
        snippet.replace('"/absolute/path/existing-corpus.sqlite3"', repr(str(api.path)))
        .replace('"col_REPLACE_WITH_32_LOWERCASE_HEX_DIGITS"', repr(collection.collection_id))
        .replace('"copy-exact-member-id"', repr(SELECTED_ID))
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", snippet],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    history = ScreeningHistory.model_validate_json(result.stdout)
    assert history.document_id == SELECTED_ID
    assert history.latest_review is not None
    assert history.latest_review.revision == 1
    assert history.latest_review.decision == "unsure"
    assert history.items == [history.latest_review]
    assert api.container.event_log.list_events() == []
