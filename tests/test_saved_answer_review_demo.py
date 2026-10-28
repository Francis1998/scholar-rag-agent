"""Reproduce the guarded saved-answer workbench and its actual browser-frame artifacts."""

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image
from scripts.demo_frozen_annotation import AuthoringPage

from agent.review_models import ReviewHistoryPage, SavedAnswerReview
from tests.test_corpus_explorer import ExplorerAPI
from tests.test_corpus_explorer import api as api
from tests.test_review_notebook import fixture_run

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs/assets"
GUIDE = ROOT / "docs/guides/SAVED_ANSWER_REVIEW_WORKBENCH_GUIDE.md"


def test_guarded_demo_retains_exact_opinion_and_restart_evidence(tmp_path: Path) -> None:
    demo = importlib.import_module("scripts.demo_saved_answer_review")
    output = tmp_path / "demo"
    transcript = demo.run_demo(output)
    checks = json.loads((output / "checks.json").read_text())
    assert checks["save_status"] == checks["retry_status"] == 303
    assert checks["review_count"] == 1
    assert checks["rejected_statuses"] == {
        "cross_origin": 403,
        "unknown_source": 422,
        "changed_uuid_payload": 409,
    }
    assert checks["events_before"] == checks["events_after"] == 8
    assert not any(checks["forbidden_call_counts"].values())
    for key in (
        "get_preview_database_unchanged",
        "rejections_database_unchanged",
        "retry_record_unchanged",
        "python_api_equal",
        "corpus_change_delete_restart_identical",
    ):
        assert checks[key] is True
    reviews = ReviewHistoryPage.model_validate_json((output / "reviews.json").read_bytes())
    assert len(reviews.reviews) == 1
    assert reviews.reviews[0].comment == demo.COMMENT
    assert reviews.reviews[0].cited_chunk_ids == (demo.CHUNK_ID,)
    page = AuthoringPage((output / "saved-review.html").read_bytes().decode())
    assert page.values["review-comment"] == [demo.COMMENT]
    assert page.passages == [demo.TEXT]
    assert (output / "saved-review.html").read_bytes() == (
        output / "deleted-restart.html"
    ).read_bytes()
    assert transcript == (ASSETS / "saved-answer-review-workbench.txt").read_text()
    assert demo.run_demo(tmp_path / "repeat") == transcript
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        demo.run_demo(output)
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        demo.run_demo(alias)


def test_demo_ignores_ambient_settings_and_credentials(tmp_path: Path) -> None:
    sentinel = tmp_path / "private.sqlite3"
    sentinel.write_bytes(b"untouched")
    (tmp_path / ".env").write_text("SCHOLAR_RAG_MAX_HOPS=invalid\n")
    output = tmp_path / "demo"
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "scripts.demo_saved_answer_review", "--output-dir", str(output)],
        cwd=tmp_path,
        env={
            **os.environ,
            "SCHOLAR_RAG_DATABASE_PATH": str(sentinel),
            "SCHOLAR_RAG_MAX_HOPS": "invalid",
            "OPENAI_API_KEY": "synthetic-unused",
            "ANTHROPIC_API_KEY": "synthetic-unused",
            "GEMINI_API_KEY": "synthetic-unused",
            "MOONSHOT_API_KEY": "synthetic-unused",
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert sentinel.read_bytes() == b"untouched"
    assert all(b"synthetic-unused" not in path.read_bytes() for path in output.iterdir())


def test_committed_animation_has_four_distinct_actual_browser_frames() -> None:
    with Image.open(ASSETS / "saved-answer-review-workbench.gif") as image:
        assert image.format == "GIF" and image.size == (1280, 960) and image.n_frames == 4
        frames = set()
        for index in range(4):
            image.seek(index)
            assert image.info["duration"] == 3000
            frames.add(image.convert("RGB").tobytes())
        assert len(frames) == 4
    with Image.open(ASSETS / "saved-answer-review-workbench.png") as image:
        assert image.format == "PNG" and image.size == (1280, 960)


def test_browser_frame_assembler_reuses_renderer_and_rejects_overwrites(tmp_path: Path) -> None:
    demo = importlib.import_module("scripts.demo_saved_answer_review")
    renderer = importlib.import_module("scripts.create_saved_answer_review_gif")
    frames = tmp_path / "frames"
    frames.mkdir()
    for index, name in enumerate(demo.FRAME_PAGES):
        Image.new("RGB", (1280, 960), (index * 60, 30, 90)).save(frames / f"{name}.png")
    gif, still = tmp_path / "new.gif", tmp_path / "new.png"
    renderer.create_preview(frames, gif, still)
    assert still.read_bytes() == (frames / f"{demo.FRAME_PAGES[0]}.png").read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        renderer.create_preview(frames, gif, still)
    with pytest.raises(ValueError, match="different paths"):
        renderer.create_preview(frames, tmp_path / "same", tmp_path / "same")
    Image.new("RGB", (10, 10)).save(frames / "preview.png")
    with pytest.raises(ValueError, match="1280x960"):
        renderer.create_preview(frames, tmp_path / "bad.gif", tmp_path / "bad.png")


def test_guide_discovery_and_comparative_limits_are_explicit() -> None:
    text = GUIDE.read_text()
    for phrase in (
        "## Native browser workflow",
        "## API contract",
        "## Python",
        "## Portfolio",
        "## Reproduce",
        "same-origin",
        "DNS rebinding",
        "no authentication",
        "https://github.com/asreview/asreview",
        "https://asreview.readthedocs.io/en/latest/lab/screening.html",
        "https://www.zotero.org/support/notes",
        "https://docs.langchain.com/langsmith/annotation-queues",
        "2026-10-27",
        "GPT-6 Astra",
        "Claude Fable 5.1",
        "Opus 5.5",
        "Sonnet 5.5",
        "Haiku 5.5",
        "Gemini 4 Argon",
        "Gemini 3.8 Flash",
        "Kimi K3.1 Pro",
        "four actual local browser",
        "not a continuous",
        "524,288",
        "16,384",
        "303",
    ):
        assert phrase in text
    for path in (ROOT / "README.md", ROOT / "docs/README.md", ROOT / "docs/DEMO.md"):
        assert GUIDE.name in path.read_text()
    for suffix in ("png", "gif", "txt"):
        assert f"saved-answer-review-workbench.{suffix}" in text


def test_documented_python_confirmation_and_precise_reader_work(
    api: ExplorerAPI, tmp_path: Path
) -> None:
    bundle = fixture_run(api)
    code = GUIDE.read_text().split("```python\n", 1)[1].split("```", 1)[0]
    for old, new in (
        ('"/absolute/path/existing-evidence.sqlite3"', repr(str(api.path))),
        ('"COPY_EXACT_RUN_ID"', repr(bundle.run_id)),
        ('"COPY_EXACT_CHUNK_ID"', repr(bundle.snapshot.sources[0].chunk.chunk_id)),
    ):
        code = code.replace(old, new)
    before = api.path.read_bytes()
    rejected = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        cwd=tmp_path,
        input="CANCEL\n",
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert rejected.returncode != 0 and "Not saved." in rejected.stderr
    assert api.path.read_bytes() == before
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        cwd=tmp_path,
        input="SAVE\n",
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    saved = SavedAnswerReview.model_validate_json(
        result.stdout.split("Type SAVE to persist this exact preview: ", 1)[1]
    )
    assert api.container.notebook_reviews.get_review(bundle.run_id, saved.review_id) == saved
