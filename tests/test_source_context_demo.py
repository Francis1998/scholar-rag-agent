"""Reproduce the source-context showcase from real, isolated API/Python/HTML reads."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image


def test_demo_outputs_are_measured_offline_reproducible_and_not_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.create_source_context_gif import create_gif
    from scripts.demo_source_context import run_demo

    from storage.source_context import SourceContext

    sentinel = tmp_path / "unrelated.sqlite3"
    sentinel.write_bytes(b"unrelated data must not be opened")
    monkeypatch.setenv("SCHOLAR_RAG_DATABASE_PATH", str(sentinel))
    monkeypatch.setenv("SCHOLAR_RAG_MAX_HOPS", "invalid")
    monkeypatch.setenv("SCHOLAR_RAG_DEFAULT_MODEL", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unused-synthetic-credential")
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("SCHOLAR_RAG_MAX_HOPS=invalid\n", encoding="utf-8")
    output = tmp_path / "first"
    transcript = run_demo(output)
    assert (output / "transcript.txt").read_text(encoding="utf-8") == transcript
    assert len(transcript.strip().split("\n\n")) == 4
    result = SourceContext.model_validate_json((output / "context.json").read_bytes())
    assert [chunk.chunk_index for chunk in result.chunks] == [8, 9, 10, 11, 12]
    assert result.chunks[2].is_anchor and result.chunks[2].chunk_id == "chunk-10"
    assert result.chunks[-1].text_truncated and len(result.chunks[-1].text) == 4000
    recentered = SourceContext.model_validate_json((output / "recentered.json").read_bytes())
    assert recentered.anchor_chunk_index == 11
    checks = json.loads((output / "checks.json").read_bytes())
    assert checks["restart_identical"] and checks["python_api_identical"]
    assert checks["database_bytes_unchanged"] and checks["excluded_chunks_returned"] == 0
    assert checks["events_before"] == checks["events_after"] == 0
    assert not any(checks["forbidden_call_counts"].values())
    assert checks["missing_anchor_status"] == 404
    assert "CURRENT CORPUS" in (output / "context.html").read_text(encoding="utf-8")
    assert "selected passage" in transcript.lower()
    assert "unused-synthetic-credential" not in transcript
    assert sentinel.read_bytes() == b"unrelated data must not be opened"
    assert not list(output.glob("*.sqlite3"))
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_demo(output)
    assert before == {path.name: path.read_bytes() for path in output.iterdir()}
    second = tmp_path / "second"
    assert run_demo(second) == transcript
    assert before == {path.name: path.read_bytes() for path in second.iterdir()}
    gif = tmp_path / "source-context.gif"
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
    reproduced = tmp_path / "reproduced.gif"
    create_gif(output / "transcript.txt", reproduced)
    assert reproduced.read_bytes() == gif.read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        create_gif(output / "transcript.txt", gif)


def test_demo_and_gif_refuse_symlinks_and_unrelated_transcripts(tmp_path: Path) -> None:
    from scripts.create_source_context_gif import create_gif
    from scripts.demo_source_context import run_demo

    output = tmp_path / "context.json"
    output.symlink_to(tmp_path / "absent.json")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_demo(tmp_path)
    assert not (tmp_path / "absent.json").exists()
    gif = tmp_path / "output.gif"
    gif.symlink_to(tmp_path / "absent.gif")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        create_gif(tmp_path / "absent.txt", gif)
    text = tmp_path / "transcript.txt"
    text.write_text("Unrelated output\n", encoding="utf-8")
    with pytest.raises(ValueError):
        create_gif(text, tmp_path / "new.gif")
    assert not (tmp_path / "new.gif").exists()


@pytest.mark.parametrize("example", ["cli", "guide"])
def test_documented_examples_ignore_ambient_credentials_and_database(
    tmp_path: Path, example: str
) -> None:
    sentinel = tmp_path / "unrelated.sqlite3"
    sentinel.write_bytes(b"original")
    prefix = """
import httpx
import socket
from scripts.demo_source_context import _no_work
httpx.HTTPTransport.handle_request = _no_work
httpx.AsyncHTTPTransport.handle_async_request = _no_work
socket.create_connection = _no_work
socket.socket.connect = _no_work
"""
    if example == "cli":
        source = """
import runpy
import sys
sys.argv = ["demo_source_context", "--output-dir", "output"]
runpy.run_module("scripts.demo_source_context", run_name="__main__")
"""
    else:
        guide = (
            Path(__file__).resolve().parents[1] / "docs/guides/SOURCE_CONTEXT_GUIDE.md"
        ).read_text(encoding="utf-8")
        match = re.search(
            r"<!-- offline-context-example:start -->\n```python\n(.*?)\n```",
            guide,
            flags=re.DOTALL,
        )
        assert match is not None
        source = match.group(1)
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", prefix + source],
        cwd=tmp_path,
        env={
            **os.environ,
            "SCHOLAR_RAG_DATABASE_PATH": str(sentinel),
            "SCHOLAR_RAG_DEFAULT_MODEL": "anthropic",
            "SCHOLAR_RAG_MAX_HOPS": "invalid",
            "ANTHROPIC_API_KEY": "unused-synthetic-credential",
        },
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "8, 9, 10, 11, 12" in result.stdout
    assert sentinel.read_bytes() == b"original"
    assert "unused-synthetic-credential" not in result.stdout
