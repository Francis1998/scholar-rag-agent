"""Verify the synthetic CLI, its real response artifacts, and transcript-driven animation."""

import json
import os
import subprocess
import sys
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx
import pytest
from PIL import Image
from scripts import demo_answer_reviews
from scripts.create_answer_reviews_gif import create_gif
from scripts.demo_answer_reviews import OUTPUT_NAMES, PANEL_TITLES, run_demo

from agent.answer_reviews import AnswerReviewService
from tests.test_answer_reviews_api import no_live_work


def test_actual_offline_demo_artifacts_cleanup_and_rendering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", no_live_work)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", no_live_work)
    monkeypatch.setattr(
        demo_answer_reviews, "TemporaryDirectory", partial(TemporaryDirectory, dir=tmp_path)
    )
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "MOONSHOT_API_KEY"):
        monkeypatch.setenv(name, "synthetic-unused-credential")
    monkeypatch.setenv("SCHOLAR_RAG_DEFAULT_MODEL", "anthropic")
    monkeypatch.setenv("SCHOLAR_RAG_MAX_HOPS", "invalid")
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("SCHOLAR_RAG_MAX_SOURCE_DOCS=0\n", encoding="utf-8")
    output_dir = tmp_path / "demo"
    transcript = run_demo(output_dir)
    assert {path.name for path in output_dir.iterdir()} == set(OUTPUT_NAMES)
    assert (output_dir / "transcript.txt").read_text(encoding="utf-8") == transcript
    assert not list(tmp_path.glob("scholar-answer-reviews-*"))
    assert not list(tmp_path.rglob("*.sqlite3"))
    bundle = json.loads((output_dir / "bundle.json").read_text(encoding="utf-8"))
    assert bundle["generation"]["provider"] == "fake"
    assert bundle["snapshot"]["sources"][0]["chunk"]["text"] == demo_answer_reviews.DEMO_TEXT
    assert demo_answer_reviews.DEMO_TEXT in (output_dir / "bundle.md").read_text(encoding="utf-8")
    first = json.loads((output_dir / "page-1.json").read_text(encoding="utf-8"))
    second = json.loads((output_dir / "page-2.json").read_text(encoding="utf-8"))
    assert first["reviews"][0]["decision"] == "accepted"
    assert first["next_cursor"] == first["reviews"][0]["sequence"]
    assert second["reviews"][0]["decision"] == "needs_revision"
    assert second["next_cursor"] is None
    assert second["reviews"] == [json.loads((output_dir / "review-created.json").read_bytes())]
    assert (output_dir / "review-created.json").read_bytes() == (
        output_dir / "review-retry.json"
    ).read_bytes()
    assert "Post-save runtime/corpus-read/event-write calls: 0" in transcript
    assert "JSON export unchanged: True; Markdown unchanged: True" in transcript
    assert "Agent event rows unchanged: True" in transcript
    assert "Temporary demo database removed: True" in transcript
    assert "synthetic-unused" not in (output_dir / "bundle.json").read_text(encoding="utf-8")

    gif = output_dir / "answer-reviews.gif"
    create_gif(output_dir / "transcript.txt", gif)
    with Image.open(gif) as image:
        assert image.n_frames == 4
        assert image.size == (1120, 540)
        assert image.info["loop"] == 0
        for frame in range(4):
            image.seek(frame)
            assert image.info["duration"] == 3500
    again = output_dir / "again.gif"
    create_gif(output_dir / "transcript.txt", again)
    assert gif.read_bytes() == again.read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        create_gif(output_dir / "transcript.txt", gif)


@pytest.mark.parametrize("name", OUTPUT_NAMES)
def test_demo_preserves_all_named_output_collisions(tmp_path: Path, name: str) -> None:
    path = tmp_path / name
    path.write_text("Keep this existing artifact.", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_demo(tmp_path)
    assert path.read_text(encoding="utf-8") == "Keep this existing artifact."
    assert list(tmp_path.iterdir()) == [path]


def test_demo_cleans_temporary_database_even_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        demo_answer_reviews, "TemporaryDirectory", partial(TemporaryDirectory, dir=tmp_path)
    )
    monkeypatch.setattr(AnswerReviewService, "create", no_live_work)
    with pytest.raises(AssertionError, match="Reviews must not generate"):
        run_demo(tmp_path / "failed-demo")
    assert not list(tmp_path.glob("scholar-answer-reviews-*"))
    assert list((tmp_path / "failed-demo").iterdir()) == []


@pytest.mark.parametrize("text", ["unrelated output", "\n\n".join(["unrelated"] * 4)])
def test_renderer_rejects_unrelated_transcript(tmp_path: Path, text: str) -> None:
    source = tmp_path / "unrelated.txt"
    source.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="demo_answer_reviews"):
        create_gif(source, tmp_path / "uncreated.gif")
    assert not (tmp_path / "uncreated.gif").exists()


@pytest.mark.parametrize("text", ["W" * 90, "\n".join(["Too many lines"] * 20)])
def test_renderer_rejects_overflow(tmp_path: Path, text: str) -> None:
    source = tmp_path / "overflow.txt"
    source.write_text("\n\n".join(f"{title}\n{text}" for title in PANEL_TITLES), encoding="utf-8")
    with pytest.raises(ValueError, match=r"width|fit"):
        create_gif(source, tmp_path / "uncreated.gif")
    assert not (tmp_path / "uncreated.gif").exists()


def test_offline_cli_and_renderer_do_not_use_ambient_database(tmp_path: Path) -> None:
    sentinel = tmp_path / "unrelated.sqlite3"
    sentinel.write_bytes(b"Synthetic sentinel, not even a SQLite database.")
    env = {
        **os.environ,
        "SCHOLAR_RAG_DATABASE_PATH": str(sentinel),
        "SCHOLAR_RAG_DEFAULT_MODEL": "anthropic",
        "SCHOLAR_RAG_MAX_HOPS": "invalid",
        "OPENAI_API_KEY": "synthetic-unused-credential",
        "ANTHROPIC_API_KEY": "synthetic-unused-credential",
        "GEMINI_API_KEY": "synthetic-unused-credential",
        "MOONSHOT_API_KEY": "synthetic-unused-credential",
    }
    output = tmp_path / "cli"
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "scripts.demo_answer_reviews", "--output-dir", str(output)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == (output / "transcript.txt").read_text(encoding="utf-8")
    gif = output / "answer-reviews.gif"
    rendered = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "scripts.create_answer_reviews_gif",
            "--transcript",
            str(output / "transcript.txt"),
            "--output",
            str(gif),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert rendered.returncode == 0, rendered.stderr
    assert gif.exists()
    assert sentinel.read_bytes() == b"Synthetic sentinel, not even a SQLite database."
    assert not (tmp_path / ".scholar-rag-agent.sqlite3").exists()
