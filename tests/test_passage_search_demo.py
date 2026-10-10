"""Reproduce the browser search showcase, native output and complete guide offline."""

import json
import os
import re
import socket
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image


def test_demo_is_measured_isolated_reproducible_and_refuses_overwriting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.create_passage_search_gif import create_gif
    from scripts.demo_passage_search import PHRASE, SearchPage, _no_work, run_demo

    from storage.literal_search import LiteralSearchPage

    sentinel = tmp_path / "unrelated.sqlite3"
    sentinel.write_bytes(b"unrelated database must not be opened")
    monkeypatch.setenv("SCHOLAR_RAG_DATABASE_PATH", str(sentinel))
    monkeypatch.setenv("SCHOLAR_RAG_MAX_HOPS", "invalid")
    monkeypatch.setenv("SCHOLAR_RAG_DEFAULT_MODEL", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unused-synthetic-credential")
    monkeypatch.setattr(socket, "create_connection", _no_work)
    monkeypatch.setattr(socket.socket, "connect", _no_work)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("SCHOLAR_RAG_MAX_HOPS=invalid\n", encoding="utf-8")
    output = tmp_path / "first"
    transcript = run_demo(output)
    assert (output / "transcript.txt").read_text(encoding="utf-8") == transcript
    committed = Path(__file__).resolve().parents[1] / "docs/assets/browser-passage-search.txt"
    assert committed.read_text(encoding="utf-8") == transcript
    checks = json.loads((output / "checks.json").read_bytes())
    assert checks["whole_corpus_matches"] == 4 and checks["selected_paper_matches"] == 3
    assert checks["collection_revision"] == 1
    assert (checks["match_start"], checks["match_end"], checks["query_codepoints"]) == (
        1140,
        1147,
        7,
    )
    assert (checks["excerpt_start"], checks["excerpt_end"]) == (1020, 1231)
    assert checks["highlight_identical"] and checks["exact_search_return"]
    assert checks["context_anchor"] == "chunk-2" and checks["source_indices"] == [0, 1, 2]
    assert checks["empty_matches"] == 0 and checks["changed_query_cursor_status"] == 422
    assert checks["python_api_identical"] and checks["restart_identical"]
    assert checks["database_bytes_unchanged"]
    assert checks["events_before"] == checks["events_after"] == 0
    assert not any(checks["forbidden_call_counts"].values())
    first = SearchPage((output / "collection-first.html").read_text(encoding="utf-8"))
    page = LiteralSearchPage.model_validate_json((output / "collection-first.json").read_bytes())
    assert first.chunks == [page.matches[0].chunk_id] == ["chunk-10"]
    assert first.highlights == [PHRASE]
    assert first.passages == [page.matches[0].excerpt]
    assert page.matches[0].excerpt.count(PHRASE) == 2
    assert not list(output.glob("*.sqlite3"))
    assert sentinel.read_bytes() == b"unrelated database must not be opened"
    assert "unused-synthetic-credential" not in transcript
    files_before = {path.name: path.read_bytes() for path in output.iterdir()}
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_demo(output)
    assert files_before == {path.name: path.read_bytes() for path in output.iterdir()}
    second = tmp_path / "second"
    assert run_demo(second) == transcript
    assert json.loads((second / "checks.json").read_bytes()) == checks
    gif = tmp_path / "browser-search.gif"
    create_gif(output / "transcript.txt", gif)
    assert gif.stat().st_size < 262144
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
    create_gif(second / "transcript.txt", reproduced)
    assert reproduced.read_bytes() == gif.read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        create_gif(output / "transcript.txt", gif)


def test_demo_and_renderer_refuse_symlinks_and_unrelated_transcripts(tmp_path: Path) -> None:
    from scripts.create_passage_search_gif import create_gif
    from scripts.demo_passage_search import run_demo

    output = tmp_path / "output"
    output.symlink_to(tmp_path / "absent")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_demo(output)
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
def test_documented_examples_ignore_environment_dotenv_and_network(
    tmp_path: Path, example: str
) -> None:
    sentinel = tmp_path / "unrelated.sqlite3"
    sentinel.write_bytes(b"original")
    (tmp_path / ".env").write_text("SCHOLAR_RAG_MAX_HOPS=invalid\n", encoding="utf-8")
    prefix = """
import httpx
import socket
from scripts.demo_passage_search import _no_work
httpx.HTTPTransport.handle_request = _no_work
httpx.AsyncHTTPTransport.handle_async_request = _no_work
socket.create_connection = _no_work
socket.socket.connect = _no_work
"""
    if example == "cli":
        source = """
import runpy
import sys
sys.argv = ["demo_passage_search", "--output-dir", "output"]
runpy.run_module("scripts.demo_passage_search", run_name="__main__")
"""
    else:
        guide = (
            Path(__file__).resolve().parents[1] / "docs/guides/BROWSER_PASSAGE_SEARCH_GUIDE.md"
        ).read_text(encoding="utf-8")
        match = re.search(
            r"<!-- offline-browser-search-example:start -->\n```python\n(.*?)\n```",
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
    assert "chunk-10" in result.stdout and "1140" in result.stdout and "1147" in result.stdout
    assert sentinel.read_bytes() == b"original"
    assert "unused-synthetic-credential" not in result.stdout
