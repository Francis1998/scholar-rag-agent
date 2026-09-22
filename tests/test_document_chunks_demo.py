"""Reproduce the evidence-reader showcase without ambient storage or providers."""

import json
import os
import re
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import httpx
import pytest
from PIL import Image
from scripts.create_document_chunks_gif import create_gif
from scripts.demo_document_chunks import PANEL_TITLES, run_demo

from storage.document_chunks import DocumentChunksPage
from tests.test_document_chunks import no_work


def test_real_demo_is_offline_and_its_gif_is_reproducible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    assert len(transcript.strip().split("\n\n")) == 4
    pages = [
        DocumentChunksPage.model_validate_json((output / f"page-{index}.json").read_bytes())
        for index in range(1, 4)
    ]
    chunks = [chunk for page in pages for chunk in page.chunks]
    assert [len(page.chunks) for page in pages] == [2, 2, 2]
    assert len({chunk.chunk_id for chunk in chunks}) == 6
    assert {chunk.document_id for chunk in chunks} == {pages[0].document_id}
    assert pages[0].next_cursor and pages[1].next_cursor and pages[2].next_cursor is None
    assert sorted(chunk.chunk_index for chunk in chunks) == list(range(6))
    assert "EXCLUDED_EVIDENCE" not in "".join(chunk.text for chunk in chunks)
    imported = DocumentChunksPage.model_validate_json((output / "long-chunk.json").read_bytes())
    assert len(imported.chunks[0].text) == 4000 and imported.chunks[0].text_truncated
    assert imported.chunks[0].chunk_index == 7
    request = json.loads((output / "query-request.json").read_bytes())
    assert request["document_ids"] == [pages[0].document_id]
    assert "Events before/after browsing: 0/0" in transcript
    assert "Query is a saved request only" in transcript
    assert "synthetic-unused" not in transcript
    assert not list(output.glob("*.sqlite3"))
    assert not (tmp_path / "unrelated.sqlite3").exists()

    before = {path.name: path.read_bytes() for path in output.iterdir()}
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_demo(output)
    assert before == {path.name: path.read_bytes() for path in output.iterdir()}
    second = tmp_path / "second-demo"
    assert run_demo(second) == transcript
    assert before == {path.name: path.read_bytes() for path in second.iterdir()}

    gif = tmp_path / "evidence.gif"
    create_gif(output / "transcript.txt", gif)
    with Image.open(gif) as image:
        assert image.n_frames == 4 and image.size == (1120, 540)
        assert image.info["loop"] == 0
        frames = []
        for index in range(image.n_frames):
            image.seek(index)
            assert image.info["duration"] == 3500
            frames.append(image.convert("RGB").tobytes())
        assert len(set(frames)) == 4
    another = tmp_path / "another.gif"
    create_gif(output / "transcript.txt", another)
    assert gif.read_bytes() == another.read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        create_gif(output / "transcript.txt", gif)


@pytest.mark.parametrize("name", ["catalog.json", "page-2.json", "query-request.json"])
def test_demo_refuses_existing_named_output_before_doing_work(tmp_path: Path, name: str) -> None:
    original = tmp_path / name
    original.write_text("Keep this artifact", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_demo(tmp_path)
    assert original.read_text(encoding="utf-8") == "Keep this artifact"
    assert [path.name for path in tmp_path.iterdir()] == [name]


def test_demo_and_gif_refuse_dangling_output_symlinks(tmp_path: Path) -> None:
    (tmp_path / "catalog.json").symlink_to(tmp_path / "absent.json")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_demo(tmp_path)
    gif = tmp_path / "demo.gif"
    gif.symlink_to(tmp_path / "absent.gif")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        create_gif(tmp_path / "absent.txt", gif)


@pytest.mark.parametrize(
    "text",
    [
        "unrelated text",
        "\n\n".join(["unrelated panel"] * 4),
        "\n\n".join(f"{title}\n{'Too much text ' * 1000}" for title in PANEL_TITLES),
        "\n\n".join(f"{title}\n{'W' * 90}" for title in PANEL_TITLES),
    ],
)
def test_gif_rejects_unrelated_or_clipped_transcripts(tmp_path: Path, text: str) -> None:
    source = tmp_path / "transcript.txt"
    source.write_text(text, encoding="utf-8")
    output = tmp_path / "not-created.gif"
    with pytest.raises(ValueError):
        create_gif(source, output)
    assert not output.exists()


@pytest.mark.parametrize("example", ["cli", "guide"])
def test_subprocess_examples_ignore_invalid_ambient_sources(tmp_path: Path, example: str) -> None:
    sentinel = tmp_path / "unrelated.sqlite3"
    with closing(sqlite3.connect(sentinel)) as connection:
        connection.execute("CREATE TABLE marker (value TEXT)")
        connection.execute("INSERT INTO marker VALUES ('original')")
        connection.commit()
    before = sentinel.read_bytes()
    env = {
        **os.environ,
        "SCHOLAR_RAG_DATABASE_PATH": str(sentinel),
        "SCHOLAR_RAG_MAX_HOPS": "invalid",
        "SCHOLAR_RAG_DEFAULT_MODEL": "anthropic",
        "OPENAI_API_KEY": "synthetic-unused-credential",
        "ANTHROPIC_API_KEY": "synthetic-unused-credential",
        "GEMINI_API_KEY": "synthetic-unused-credential",
        "MOONSHOT_API_KEY": "synthetic-unused-credential",
    }
    (tmp_path / ".env").write_text(
        "SCHOLAR_RAG_MAX_SOURCE_DOCS=invalid\nSCHOLAR_RAG_OPENAI_MODEL=\n",
        encoding="utf-8",
    )
    prefix = """
import httpx
from unittest.mock import patch
from scripts.demo_document_chunks import _no_work
from llm.router import RoutingLLMAdapter

httpx.HTTPTransport.handle_request = _no_work
httpx.AsyncHTTPTransport.handle_async_request = _no_work
RoutingLLMAdapter.generate = _no_work
"""
    if example == "cli":
        source = """
import runpy
import sys
sys.argv = ["demo_document_chunks", "--output-dir", "output"]
runpy.run_module("scripts.demo_document_chunks", run_name="__main__")
"""
    else:
        guide = (
            Path(__file__).resolve().parents[1] / "docs/guides/DOCUMENT_CHUNKS_GUIDE.md"
        ).read_text(encoding="utf-8")
        match = re.search(
            r"<!-- offline-reader-example:start -->\n```python\n(.*?)\n```",
            guide,
            flags=re.DOTALL,
        )
        assert match is not None
        source = match.group(1)
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", prefix + source],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert sentinel.read_bytes() == before
    assert not (tmp_path / ".scholar-rag-agent.sqlite3").exists()
    assert "synthetic-unused" not in result.stdout
    assert "document_ids" in result.stdout
