"""Reproducibility and isolation checks for the actual-HTML/browser-frame demonstration."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from scripts.create_corpus_explorer_gif import create_gif
from scripts.demo_corpus_explorer import (
    FRAME_PAGES,
    SELECTED_ID,
    ExplorerPage,
    run_demo,
    serve_demo,
)
from scripts.demo_evidence_export import offline_settings

from api.application import create_app
from tests.test_corpus_explorer import no_work


def test_demo_uses_actual_html_and_persists_without_ambient_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", no_work)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", no_work)
    monkeypatch.setenv("SCHOLAR_RAG_DEFAULT_MODEL", "anthropic")
    monkeypatch.setenv("SCHOLAR_RAG_MAX_HOPS", "invalid")
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "MOONSHOT_API_KEY"):
        monkeypatch.setenv(key, "synthetic-unused-credential")
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "SCHOLAR_RAG_DATABASE_PATH=unrelated.sqlite3\nSCHOLAR_RAG_MAX_HOPS=invalid\n",
        encoding="utf-8",
    )
    output = tmp_path / "demo"
    transcript = run_demo(output)
    assert (output / "transcript.txt").read_text(encoding="utf-8") == transcript
    checks = json.loads((output / "checks.json").read_bytes())
    assert checks["catalog_pages"] == [[SELECTED_ID, "02-limitations"], ["03-protocol"]]
    assert checks["chunk_pages"] == [
        ["chunk-1", "chunk-10"],
        ["chunk-2", "chunk-20"],
        ["chunk-3"],
    ]
    assert checks["chunk_indices_in_id_order"] == [0, 9, 1, 19, 2]
    assert checks["restart_html_identical"] and checks["database_bytes_unchanged"]
    assert checks["events_before"] == checks["events_after"] == 0
    assert set(checks["forbidden_call_counts"].values()) == {0}
    assert "synthetic-unused" not in transcript
    assert not (tmp_path / "unrelated.sqlite3").exists()
    app = create_app(offline_settings(output / "corpus.sqlite3"))
    with TestClient(app) as client:
        urls = json.loads((output / "pages.json").read_bytes())
        for name, url in urls.items():
            response = client.get(url)
            assert response.status_code == 200
            assert response.content == (output / f"{name}.html").read_bytes()
    parsed = ExplorerPage((output / "passages-last.html").read_text())
    assert parsed.documents == [SELECTED_ID]
    assert parsed.indices == ["2"]
    assert len(parsed.passages[0]) == 4000
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_demo(output)
    assert before == {path.name: path.read_bytes() for path in output.iterdir()}
    second = tmp_path / "demo-two"
    assert run_demo(second) == transcript
    assert (second / "pages.json").read_bytes() == (output / "pages.json").read_bytes()
    assert (second / "checks.json").read_bytes() == (output / "checks.json").read_bytes()


def test_demo_and_gif_reject_dangling_destinations(tmp_path: Path) -> None:
    output = tmp_path / "demo"
    output.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        run_demo(output)
    gif = tmp_path / "demo.gif"
    gif.symlink_to(tmp_path / "missing.gif")
    with pytest.raises(FileExistsError):
        create_gif(output, gif)


def test_gif_preserves_four_distinct_screenshot_frames(tmp_path: Path) -> None:
    for index, name in enumerate(FRAME_PAGES):
        Image.new("RGB", (1280, 960), (index * 60, 30, 50)).save(tmp_path / f"{name}.png")
    output = tmp_path / "demo.gif"
    create_gif(tmp_path, output)
    with Image.open(output) as image:
        assert image.n_frames == 4
        assert image.size == (1280, 960) and image.info["loop"] == 0
        for index in range(4):
            image.seek(index)
            assert image.info["duration"] == 3000
            assert image.convert("RGB").getpixel((0, 0)) == (index * 60, 30, 50)
    second = tmp_path / "second.gif"
    create_gif(tmp_path, second)
    assert output.read_bytes() == second.read_bytes()
    with pytest.raises(FileExistsError):
        create_gif(tmp_path, output)


@pytest.mark.parametrize("size", [(1200, 960), (1280, 900), (1280, 960)])
def test_gif_rejects_resizing_and_duplicate_frames(tmp_path: Path, size: tuple[int, int]) -> None:
    for name in FRAME_PAGES:
        Image.new("RGB", size, "white").save(tmp_path / f"{name}.png")
    output = tmp_path / "demo.gif"
    with pytest.raises(ValueError):
        create_gif(tmp_path, output)
    assert not output.exists()


def test_serve_requires_a_real_demo_and_disables_forbidden_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import demo_corpus_explorer

    directory = tmp_path / "demo"
    run_demo(directory)
    run = Mock()
    monkeypatch.setattr(demo_corpus_explorer.uvicorn, "run", run)
    serve_demo(directory, 8767)
    assert run.call_args.kwargs == {"host": "127.0.0.1", "port": 8767, "access_log": False}
    (directory / "checks.json").write_text('{"fixture":"not-this-demo"}')
    with pytest.raises(ValueError, match="existing corpus explorer demo"):
        serve_demo(directory, 8767)
    assert run.call_count == 1


@pytest.mark.parametrize("example", ["cli", "python-guide"])
def test_executable_examples_ignore_ambient_sources(tmp_path: Path, example: str) -> None:
    sentinel = tmp_path / "untouched.sqlite3"
    sentinel.write_bytes(b"Never open this ambient path.")
    env = {
        **os.environ,
        "SCHOLAR_RAG_DATABASE_PATH": str(sentinel),
        "SCHOLAR_RAG_MAX_HOPS": "invalid",
        "OPENAI_API_KEY": "synthetic-unused-credential",
    }
    prefix = """
import httpx
from scripts.demo_corpus_explorer import _no_work
from llm.router import RoutingLLMAdapter
httpx.HTTPTransport.handle_request = _no_work
httpx.AsyncHTTPTransport.handle_async_request = _no_work
RoutingLLMAdapter.generate = _no_work
"""
    if example == "cli":
        source = """
import runpy, sys
sys.argv = ["demo_corpus_explorer", "--output-dir", "output"]
runpy.run_module("scripts.demo_corpus_explorer", run_name="__main__")
"""
    else:
        guide = (
            Path(__file__).resolve().parents[1] / "docs/guides/CORPUS_EXPLORER_GUIDE.md"
        ).read_text()
        match = re.search(r"<!-- python-example:start -->\n```python\n(.*?)\n```", guide, re.DOTALL)
        assert match is not None
        source = match.group(1)
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", prefix + source],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert sentinel.read_bytes() == b"Never open this ambient path."
    assert "synthetic-unused" not in result.stdout
    assert not (tmp_path / ".scholar-rag-agent.sqlite3").exists()
