"""Require reproducible actual browser/API output and decodable screenshot-based assets."""

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import pytest
from PIL import Image
from scripts.create_corpus_explorer_gif import FRAME_SIZE
from scripts.create_saved_run_browser_gif import create_gif
from scripts.demo_saved_run_browser import FRAME_PAGES, run_demo, serve_demo

from agent.evidence import EvidenceBundle
from agent.evidence_html import render_html
from storage.evidence_export import SQLiteEvidenceReader
from tests.test_corpus_explorer import ExplorerAPI, ExplorerDOM, href, identities, values
from tests.test_corpus_explorer import api as api
from tests.test_run_explorer import complete


def test_demo_records_actual_native_outcomes_and_restart_guards(tmp_path: Path) -> None:
    output = tmp_path / "demo"
    transcript = run_demo(output)
    assert (output / "transcript.txt").read_text() == transcript
    checks = json.loads((output / "checks.json").read_text())
    urls = json.loads((output / "pages.json").read_text())
    assert set(urls) == set(FRAME_PAGES)
    assert checks["legacy_status"] == checks["incomplete_status"] == 409
    assert checks["query_preview_characters"] == 300 and checks["query_truncated"] is True
    assert checks["events_before"] == checks["events_after"] == 19
    assert checks["remaining_corpus_chunks"] == checks["socket_connection_calls"] == 0
    assert not any(checks["forbidden_call_counts"].values())
    assert checks["api_python_browser_parity"] and checks["history_return_preserved"]
    assert checks["corpus_change_delete_restart_identical"] and checks["all_downloads_identical"]
    assert checks["database_bytes_unchanged_during_reads"]
    database = output / "history.sqlite3"
    assert hashlib.sha256(database.read_bytes()).hexdigest() == checks["database_sha256"]
    bundle = EvidenceBundle.model_validate_json((output / "evidence.json").read_bytes())
    assert bundle == SQLiteEvidenceReader(database).export(checks["frozen_run_id"])
    assert bundle.generation.provider == "fake" and len(bundle.snapshot.sources) == 2
    assert (output / "evidence.html").read_text() == render_html(bundle)
    assert (output / "frozen.html").read_bytes() == (output / "restarted.html").read_bytes()
    frozen = ExplorerDOM((output / "frozen.html").read_text()).root
    assert href(frozen, "back-to-runs") == urls["history-next"]
    assert values(frozen, "passage") == [source.chunk.text for source in bundle.snapshot.sources]
    for number, name in enumerate(("history-first", "history-next")):
        root = ExplorerDOM((output / f"{name}.html").read_text()).root
        assert identities(root, "run-id") == checks["history_pages"][number]
    for name in ("legacy", "incomplete"):
        root = ExplorerDOM((output / f"{name}.html").read_text()).root
        assert not any(
            node.attrs.get("id", "").startswith("download-") for node in root.by_tag("a")
        )
    before = database.read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_demo(output)
    assert database.read_bytes() == before
    assert run_demo(tmp_path / "repeat") == transcript


def test_demo_cli_ignores_ambient_credentials_and_database(tmp_path: Path) -> None:
    sentinel = tmp_path / "private.sqlite3"
    with closing(sqlite3.connect(sentinel)) as connection, connection:
        connection.execute("CREATE TABLE marker (value TEXT)")
        connection.execute("INSERT INTO marker VALUES ('keep')")
    before = sentinel.read_bytes()
    (tmp_path / ".env").write_text("SCHOLAR_RAG_MAX_SOURCE_DOCS=invalid\n")
    output = tmp_path / "demo"
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "scripts.demo_saved_run_browser", "--output-dir", str(output)],
        cwd=tmp_path,
        env={
            **os.environ,
            "SCHOLAR_RAG_DATABASE_PATH": str(sentinel),
            "SCHOLAR_RAG_DEFAULT_MODEL": "anthropic",
            "SCHOLAR_RAG_MAX_HOPS": "invalid",
            "OPENAI_API_KEY": "synthetic-unused-key",
            "ANTHROPIC_API_KEY": "synthetic-unused-key",
            "GEMINI_API_KEY": "synthetic-unused-key",
            "MOONSHOT_API_KEY": "synthetic-unused-key",
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert sentinel.read_bytes() == before
    for path in output.iterdir():
        assert b"synthetic-unused-key" not in path.read_bytes()


def test_demo_refuses_symlinks_and_unmarked_serve_directories(tmp_path: Path) -> None:
    target = tmp_path / "output"
    target.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_demo(target)
    (tmp_path / "checks.json").write_text('{"fixture":"not-our-fixture"}')
    with pytest.raises(ValueError, match="existing saved-run browser demo"):
        serve_demo(tmp_path, 8768)
    assert not (tmp_path / "history.sqlite3").exists()


def test_screenshot_renderer_preserves_four_distinct_frames_and_refuses_replacements(
    tmp_path: Path,
) -> None:
    for name, color in zip(FRAME_PAGES, ("red", "green", "blue", "yellow"), strict=True):
        Image.new("RGB", FRAME_SIZE, color).save(tmp_path / f"{name}.png")
    first, second = tmp_path / "first.gif", tmp_path / "second.gif"
    create_gif(tmp_path, first)
    create_gif(tmp_path, second)
    assert first.read_bytes() == second.read_bytes()
    with Image.open(first) as image:
        assert image.n_frames == 4 and image.size == FRAME_SIZE
        pixels = set()
        for frame in range(image.n_frames):
            image.seek(frame)
            image.load()
            assert image.info["duration"] == 3000
            pixels.add(image.convert("RGB").getpixel((0, 0)))
        assert len(pixels) == 4
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        create_gif(tmp_path, first)
    Image.new("RGB", (10, 10), "white").save(tmp_path / f"{FRAME_PAGES[0]}.png")
    with pytest.raises(ValueError, match="1280x960"):
        create_gif(tmp_path, tmp_path / "wrong.gif")


def test_checked_in_browser_gif_decodes_all_frames_and_has_a_still_preview() -> None:
    assets = Path(__file__).resolve().parents[1] / "docs/assets"
    with Image.open(assets / "saved-run-browser.gif") as image:
        assert image.format == "GIF" and image.n_frames == 4 and image.size == FRAME_SIZE
        assert image.info["loop"] == 0
        frames = set()
        for number in range(image.n_frames):
            image.seek(number)
            image.load()
            assert image.info["duration"] == 3000
            frames.add(hashlib.sha256(image.convert("RGB").tobytes()).hexdigest())
        assert len(frames) == 4
    with Image.open(assets / "saved-run-browser.png") as image:
        image.load()
        assert image.format == "PNG" and image.size == FRAME_SIZE


def test_documented_python_reader_runs_without_startup_or_database_writes(
    api: ExplorerAPI, tmp_path: Path
) -> None:
    bundle = complete(api)
    guide = Path(__file__).resolve().parents[1] / "docs/guides/SAVED_RUN_BROWSER_GUIDE.md"
    snippet = guide.read_text().split("```python\n", 1)[1].split("```", 1)[0]
    destination = tmp_path / "documented-reader.html"
    snippet = (
        snippet.replace('"/absolute/path/existing-runs.sqlite3"', repr(str(api.path)))
        .replace('"/absolute/path/new-frozen-evidence.html"', repr(str(destination)))
        .replace('"copy-an-exact-run-id"', repr(bundle.run_id))
    )
    before = api.path.read_bytes()
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", snippet],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert destination.read_text() == render_html(bundle)
    assert api.path.read_bytes() == before
    assert not (tmp_path / ".scholar-rag-agent.sqlite3").exists()
