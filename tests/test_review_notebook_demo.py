"""Actual-output provenance, offline isolation and screenshot assembly for the notebook."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from scripts.create_corpus_explorer_gif import FRAME_SIZE
from scripts.create_review_notebook_gif import create_gif
from scripts.demo_evidence_export import offline_settings
from scripts.demo_review_notebook import (
    FRAME_PAGES,
    QUOTE,
    notebook_guards,
    run_demo,
    serve_demo,
)

from api.application import create_app
from storage.evidence_export import SQLiteEvidenceReader
from tests.test_corpus_explorer import ExplorerDOM, values


def test_demo_persists_actual_frozen_opinions_and_measured_reads(tmp_path: Path) -> None:
    output = tmp_path / "demo"
    transcript = run_demo(output)
    assert (output / "transcript.txt").read_text() == transcript
    checks = json.loads((output / "checks.json").read_text())
    urls = json.loads((output / "pages.json").read_text())
    assert set(urls) == set(FRAME_PAGES)
    assert checks["provider"] == "fake"
    assert checks["review_decisions"] == ["rejected", "needs_revision", "accepted"]
    assert checks["annotations"] == 2
    assert checks["quote"] == QUOTE and checks["start"] > checks["first_occurrence"]
    assert checks["events_before"] == checks["events_after"] == 8
    assert checks["remaining_corpus_chunks"] == checks["socket_connection_calls"] == 0
    assert not any(checks["forbidden_call_counts"].values())
    assert checks["history_return_preserved"] and checks["independent_pagination"]
    assert checks["corpus_change_delete_restart_identical"]
    assert checks["database_bytes_unchanged_during_reads"]
    path = output / "notebook.sqlite3"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == checks["database_sha256"]
    bundle = SQLiteEvidenceReader(path).export(checks["run_id"])
    root = ExplorerDOM((output / "exact-quote.html").read_text()).root
    assert values(root, "passage-text") == [bundle.snapshot.sources[0].chunk.text]
    assert root.by_id("annotation-quote").text() == checks["quote"]
    app = create_app(offline_settings(path))
    before = path.read_bytes()
    with TestClient(app) as client, notebook_guards(app.state.container) as guards:
        for name, url in urls.items():
            response = client.get(url)
            assert response.status_code == 200
            assert response.content == (output / f"{name}.html").read_bytes()
        assert not any(guard.call_count for guard in guards.values())
    assert path.read_bytes() == before
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_demo(output)
    assert path.read_bytes() == before
    assert run_demo(tmp_path / "repeat") == transcript


def test_demo_cli_and_documented_python_ignore_ambient_settings(tmp_path: Path) -> None:
    sentinel = tmp_path / "private.sqlite3"
    sentinel.write_bytes(b"Do not read or write the ambient database.")
    (tmp_path / ".env").write_text("SCHOLAR_RAG_MAX_SOURCE_DOCS=invalid\n")
    output = tmp_path / "demo"
    environment = {
        **os.environ,
        "SCHOLAR_RAG_DATABASE_PATH": str(sentinel),
        "SCHOLAR_RAG_DEFAULT_MODEL": "anthropic",
        "SCHOLAR_RAG_MAX_HOPS": "invalid",
        **dict.fromkeys(
            ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "MOONSHOT_API_KEY"),
            "synthetic-unused-key",
        ),
    }
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "scripts.demo_review_notebook", "--output-dir", str(output)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    guide = Path(__file__).resolve().parents[1] / "docs/guides/HUMAN_REVIEW_NOTEBOOK_GUIDE.md"
    source = guide.read_text().split("```python\n", 1)[1].split("```", 1)[0]
    source = source.replace('"/absolute/path/to/generated-notebook-demo"', repr(str(output)))
    before = (output / "notebook.sqlite3").read_bytes()
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", source],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert QUOTE in result.stdout
    assert (output / "notebook.sqlite3").read_bytes() == before
    assert sentinel.read_bytes() == b"Do not read or write the ambient database."
    assert not (tmp_path / ".scholar-rag-agent.sqlite3").exists()
    for path in output.iterdir():
        assert b"synthetic-unused-key" not in path.read_bytes()


def test_serve_guards_fixture_identity_hash_and_loopback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "demo"
    run_demo(output)
    run = Mock()
    monkeypatch.setattr("scripts.demo_review_notebook.uvicorn.run", run)
    serve_demo(output, 8770)
    assert run.call_args.kwargs == {"host": "127.0.0.1", "port": 8770, "access_log": False}
    with (output / "notebook.sqlite3").open("ab") as file:
        file.write(b"changed")
    with pytest.raises(ValueError, match="changed after"):
        serve_demo(output, 8770)
    (output / "checks.json").write_text('{"fixture":"not-our-fixture"}')
    with pytest.raises(ValueError, match="existing human-review notebook"):
        serve_demo(output, 8770)
    assert run.call_count == 1
    dangling = tmp_path / "dangling"
    dangling.symlink_to(tmp_path / "absent")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_demo(dangling)


def test_renderer_preserves_the_named_four_screenshots(tmp_path: Path) -> None:
    for name, color in zip(FRAME_PAGES, ("red", "green", "blue", "yellow"), strict=True):
        Image.new("RGB", FRAME_SIZE, color).save(tmp_path / f"{name}.png")
    first, second = tmp_path / "first.gif", tmp_path / "second.gif"
    create_gif(tmp_path, first)
    create_gif(tmp_path, second)
    assert first.read_bytes() == second.read_bytes()
    with Image.open(first) as image:
        assert image.n_frames == 4 and image.size == FRAME_SIZE
        frames = set()
        for index in range(4):
            image.seek(index)
            assert image.info["duration"] == 3000
            frames.add(image.convert("RGB").tobytes())
        assert len(frames) == 4
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        create_gif(tmp_path, first)


def test_checked_in_gif_decodes_and_links_have_a_still_preview() -> None:
    assets = Path(__file__).resolve().parents[1] / "docs/assets"
    with Image.open(assets / "human-review-notebook.gif") as image:
        assert image.format == "GIF" and image.n_frames == 4 and image.size == FRAME_SIZE
        frames = set()
        for index in range(4):
            image.seek(index)
            image.load()
            assert image.info["duration"] == 3000
            frames.add(hashlib.sha256(image.convert("RGB").tobytes()).hexdigest())
        assert len(frames) == 4
    with Image.open(assets / "human-review-notebook.png") as image:
        image.load()
        assert image.format == "PNG" and image.size == FRAME_SIZE
