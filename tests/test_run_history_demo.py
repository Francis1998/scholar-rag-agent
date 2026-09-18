"""Exercise the real synthetic demo and its transcript-driven illustration."""

import importlib
import json
import os
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import httpx
import pytest
from PIL import Image

from storage.run_history import RunHistoryPage
from tests.test_run_history import _no_work


def test_demo_recovers_real_runs_offline_and_generates_reproducible_gif(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo = importlib.import_module("scripts.demo_run_history")
    renderer = importlib.import_module("scripts.create_run_history_gif")
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", _no_work)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _no_work)
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "MOONSHOT_API_KEY"):
        monkeypatch.setenv(name, "synthetic-unused-credential")
    monkeypatch.setenv("SCHOLAR_RAG_DEFAULT_MODEL", "anthropic")
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "ANTHROPIC_API_KEY=synthetic-unused-dotenv\nSCHOLAR_RAG_DEFAULT_MODEL=anthropic\n",
        encoding="utf-8",
    )
    output = tmp_path / "demo"
    transcript = demo.run_demo(output)
    assert (output / "transcript.txt").read_text(encoding="utf-8") == transcript
    first = RunHistoryPage.model_validate_json((output / "page-1.json").read_bytes())
    second = RunHistoryPage.model_validate_json((output / "page-2.json").read_bytes())
    filtered = RunHistoryPage.model_validate_json((output / "done.json").read_bytes())
    assert [run.recorded_state for run in first.runs] == ["REASONING", "ERROR"]
    assert [run.recorded_state for run in second.runs] == ["DONE"]
    assert first.next_cursor == first.runs[-1].first_event_id
    assert second.next_cursor is filtered.next_cursor is None
    assert filtered.runs == second.runs
    bundle = json.loads((output / "bundle.json").read_text(encoding="utf-8"))
    assert bundle["run_id"] == second.runs[0].run_id
    assert bundle["generation"]["provider"] == "fake"
    assert bundle["query"] == second.runs[0].query_summary
    assert "synthetic-unused" not in json.dumps(bundle)
    assert "JSON export after restart: byte-identical" in transcript
    assert "fixture, not an active job" in transcript
    assert "No duplicates: 3 discovered runs" in transcript
    assert (
        (output / "bundle.md")
        .read_text(encoding="utf-8")
        .startswith("# Research evidence bundle\n")
    )
    assert len(json.loads((output / "events.json").read_text(encoding="utf-8"))) == 8

    before = {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        demo.run_demo(output)
    assert before == {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}
    gif = tmp_path / "run-history.gif"
    renderer.create_gif(output / "transcript.txt", gif)
    with Image.open(gif) as image:
        assert image.n_frames == 4
        assert image.size == (1120, 600)
        assert image.info["loop"] == 0
        assert "synthetic" in image.info["comment"].decode("utf-8").lower()
        for index in range(image.n_frames):
            image.seek(index)
            assert image.info["duration"] == 4000
    another = tmp_path / "again.gif"
    renderer.create_gif(output / "transcript.txt", another)
    assert gif.read_bytes() == another.read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        renderer.create_gif(output / "transcript.txt", gif)


def test_demo_refuses_any_named_output_collision(tmp_path: Path) -> None:
    demo = importlib.import_module("scripts.demo_run_history")
    saved = tmp_path / "done.json"
    saved.write_text("Keep this artifact", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        demo.run_demo(tmp_path)
    assert saved.read_text(encoding="utf-8") == "Keep this artifact"
    assert not (tmp_path / "history.sqlite3").exists()


@pytest.mark.parametrize("transcript", ["unrelated text", "\n\n".join(["unrelated"] * 4)])
def test_gif_rejects_unrelated_transcripts(tmp_path: Path, transcript: str) -> None:
    renderer = importlib.import_module("scripts.create_run_history_gif")
    source = tmp_path / "unrelated.txt"
    source.write_text(transcript, encoding="utf-8")
    output = tmp_path / "not-created.gif"
    with pytest.raises(ValueError, match="run-history demo"):
        renderer.create_gif(source, output)
    assert not output.exists()


def test_gif_rejects_overflow_instead_of_clipping_actual_output(tmp_path: Path) -> None:
    demo = importlib.import_module("scripts.demo_run_history")
    renderer = importlib.import_module("scripts.create_run_history_gif")
    output = tmp_path / "demo"
    demo.run_demo(output)
    source = output / "transcript.txt"
    text = source.read_text(encoding="utf-8")
    first, rest = text.split("\n\n", 1)
    source.write_text(first + "\n" + ("Too much text " * 1000) + "\n\n" + rest, encoding="utf-8")
    gif = tmp_path / "not-created.gif"
    with pytest.raises(ValueError, match="fit"):
        renderer.create_gif(source, gif)
    assert not gif.exists()


@pytest.mark.parametrize("module", ["scripts.demo_run_history", "scripts.demo_evidence_export"])
@pytest.mark.parametrize("ambient", ["valid", "invalid-environment", "invalid-dotenv"])
def test_offline_cli_never_initializes_or_uses_ambient_settings(
    tmp_path: Path, module: str, ambient: str
) -> None:
    sentinel = tmp_path / "unrelated.sqlite3"
    with closing(sqlite3.connect(sentinel)) as connection:
        connection.execute("CREATE TABLE unrelated_marker (value TEXT NOT NULL)")
        connection.execute("INSERT INTO unrelated_marker VALUES ('synthetic sentinel')")
        connection.commit()
    before = sentinel.read_bytes()
    env = {
        **os.environ,
        "SCHOLAR_RAG_DATABASE_PATH": str(sentinel),
        "SCHOLAR_RAG_AGENT_ID": "ambient-agent-must-not-be-used",
        "SCHOLAR_RAG_DEFAULT_MODEL": "anthropic",
        "OPENAI_API_KEY": "synthetic-unused-credential",
        "ANTHROPIC_API_KEY": "synthetic-unused-credential",
        "GEMINI_API_KEY": "synthetic-unused-credential",
        "MOONSHOT_API_KEY": "synthetic-unused-credential",
    }
    invalid = {
        "SCHOLAR_RAG_OPENAI_MODEL": "",
        "SCHOLAR_RAG_ANTHROPIC_MODEL": "",
        "SCHOLAR_RAG_GEMINI_MODEL": "",
        "SCHOLAR_RAG_KIMI_MODEL": "",
        "SCHOLAR_RAG_MAX_SOURCE_DOCS": "0",
        "SCHOLAR_RAG_MAX_HOPS": "99",
        "SCHOLAR_RAG_RETRIEVAL_TIMEOUT_SECONDS": "invalid",
        "SCHOLAR_RAG_REASONING_TIMEOUT_SECONDS": "invalid",
    }
    if ambient == "invalid-environment":
        env.update(invalid)
    elif ambient == "invalid-dotenv":
        for key in invalid:
            env.pop(key, None)
        (tmp_path / ".env").write_text(
            "\n".join(f"{key}={value}" for key, value in invalid.items()), encoding="utf-8"
        )
    output = tmp_path / "offline-output"
    command = """
import runpy
import sys
import httpx

def no_network(*args, **kwargs):
    raise AssertionError("Offline demo attempted an external request")

httpx.HTTPTransport.handle_request = no_network
httpx.AsyncHTTPTransport.handle_async_request = no_network
module, destination = sys.argv[1:]
sys.argv = [module, "--output-dir", destination]
runpy.run_module(module, run_name="__main__")
"""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", command, module, str(output)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert sentinel.read_bytes() == before, "Demo changed an unrelated ambient database"
    assert not (tmp_path / ".scholar-rag-agent.sqlite3").exists()
    bundle = json.loads((output / "bundle.json").read_text(encoding="utf-8"))
    assert bundle["agent_id"] == "local-agent"
    assert bundle["generation"]["provider"] == "fake"
    assert bundle["configuration"]["max_source_docs"] == 50
    assert bundle["configuration"]["max_hops"] == 5
    assert "synthetic-unused" not in json.dumps(bundle)


def test_gif_rejects_horizontal_pixel_overflow(tmp_path: Path) -> None:
    renderer = importlib.import_module("scripts.create_run_history_gif")
    source = tmp_path / "wide-glyphs.txt"
    source.write_text(
        "\n\n".join(f"{title}\n{'W' * 90}" for title in renderer._TITLES),
        encoding="utf-8",
    )
    output = tmp_path / "not-created.gif"
    with pytest.raises(ValueError, match="fit"):
        renderer.create_gif(source, output)
    assert not output.exists()
