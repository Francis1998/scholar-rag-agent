"""Output-contract tests for the real offline comparison demo and its illustration."""

import importlib
import os
import sqlite3
import subprocess
import sys
import tempfile
from contextlib import closing
from pathlib import Path

import httpx
import pytest
from PIL import Image

from agent.comparison_models import RunComparison
from agent.evidence import EvidenceBundle
from agent.run_comparison import compare_bundles
from tests.test_run_comparison import _no_work


def test_demo_compares_real_saved_runs_and_renders_reproducible_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo = importlib.import_module("scripts.demo_run_comparison")
    renderer = importlib.import_module("scripts.create_run_comparison_gif")
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _no_work)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", _no_work)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SCHOLAR_RAG_DEFAULT_MODEL", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "synthetic-unused-credential")
    monkeypatch.setenv("SCHOLAR_RAG_MAX_HOPS", "invalid")
    (tmp_path / ".env").write_text(
        "SCHOLAR_RAG_MAX_SOURCE_DOCS=invalid\nOPENAI_API_KEY=synthetic-unused-dotenv\n",
        encoding="utf-8",
    )
    output = tmp_path / "output"
    transcript = demo.run_demo(output)
    assert (output / "transcript.txt").read_text(encoding="utf-8") == transcript
    baseline = EvidenceBundle.model_validate_json((output / "baseline.json").read_bytes())
    candidate = EvidenceBundle.model_validate_json((output / "candidate.json").read_bytes())
    result = RunComparison.model_validate_json((output / "comparison.json").read_bytes())
    assert baseline.run_id != candidate.run_id
    assert baseline.generation.provider == candidate.generation.provider == "fake"
    assert result == compare_bundles(baseline, candidate)
    assert result.changes.query_changed and result.changes.document_scope_changed
    assert result.changes.answer_text_changed
    assert result.evidence.statistics.baseline_count == 2
    assert result.evidence.statistics.candidate_count == 2
    assert result.evidence.statistics.shared_count == 1
    assert result.evidence.statistics.added_count == result.evidence.statistics.removed_count == 1
    assert result.evidence.statistics.identity_jaccard == 1 / 3
    assert "synthetic-unused" not in result.model_dump_json()
    assert "Comparison after corpus deletion/restart: byte-identical" in transcript
    assert "Comparison event writes: 0" in transcript
    assert "Temporary demo database removed: True" in transcript
    assert not list(tmp_path.rglob("*.sqlite3"))
    assert {item.name for item in output.iterdir()} == {
        "baseline.json",
        "candidate.json",
        "comparison.json",
        "transcript.txt",
    }
    before = {item.name: item.read_bytes() for item in output.iterdir()}
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        demo.run_demo(output)
    assert before == {item.name: item.read_bytes() for item in output.iterdir()}
    first, second = tmp_path / "first.gif", tmp_path / "second.gif"
    renderer.create_gif(output / "transcript.txt", first)
    renderer.create_gif(output / "transcript.txt", second)
    assert first.read_bytes() == second.read_bytes()
    with Image.open(first) as image:
        assert image.n_frames == 4
        assert image.size == (1120, 540)
        assert image.info["loop"] == 0
        for index in range(image.n_frames):
            image.seek(index)
            assert image.info["duration"] == 3500
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        renderer.create_gif(output / "transcript.txt", first)


@pytest.mark.parametrize(
    "name", ["baseline.json", "candidate.json", "comparison.json", "transcript.txt"]
)
def test_demo_refuses_any_output_collision(tmp_path: Path, name: str) -> None:
    demo = importlib.import_module("scripts.demo_run_comparison")
    existing = tmp_path / name
    existing.write_text("Preserve this artifact", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        demo.run_demo(tmp_path)
    assert existing.read_text(encoding="utf-8") == "Preserve this artifact"
    assert {item.name for item in tmp_path.iterdir()} == {name}


@pytest.mark.parametrize("damage", ["unrelated", "horizontal", "vertical"])
def test_gif_rejects_invalid_or_overflowing_transcripts(tmp_path: Path, damage: str) -> None:
    demo = importlib.import_module("scripts.demo_run_comparison")
    renderer = importlib.import_module("scripts.create_run_comparison_gif")
    if damage == "unrelated":
        transcript = "\n\n".join(["Not the comparison demo"] * 4)
    else:
        body = "W" * 90 if damage == "horizontal" else "too much text " * 1000
        transcript = "\n\n".join(f"{title}\n{body}" for title in demo.PANEL_TITLES)
    source = tmp_path / "transcript.txt"
    source.write_text(transcript, encoding="utf-8")
    destination = tmp_path / "not-created.gif"
    with pytest.raises(ValueError):
        renderer.create_gif(source, destination)
    assert not destination.exists()


@pytest.mark.parametrize("ambient", ["valid", "invalid-environment", "invalid-dotenv"])
def test_comparison_cli_ignores_ambient_settings_and_databases(
    tmp_path: Path, ambient: str
) -> None:
    sentinel = tmp_path / "unrelated.sqlite3"
    with closing(sqlite3.connect(sentinel)) as connection:
        connection.execute("CREATE TABLE marker (value TEXT)")
        connection.execute("INSERT INTO marker VALUES ('synthetic sentinel')")
        connection.commit()
    before = sentinel.read_bytes()
    env = {
        **os.environ,
        "SCHOLAR_RAG_DATABASE_PATH": str(sentinel),
        "SCHOLAR_RAG_AGENT_ID": "do-not-use-ambient-agent",
        "SCHOLAR_RAG_DEFAULT_MODEL": "anthropic",
        "OPENAI_API_KEY": "synthetic-unused-credential",
        "ANTHROPIC_API_KEY": "synthetic-unused-credential",
        "GEMINI_API_KEY": "synthetic-unused-credential",
        "MOONSHOT_API_KEY": "synthetic-unused-credential",
    }
    invalid = {
        "SCHOLAR_RAG_OPENAI_MODEL": "",
        "SCHOLAR_RAG_GEMINI_MODEL": "",
        "SCHOLAR_RAG_MAX_SOURCE_DOCS": "0",
        "SCHOLAR_RAG_MAX_HOPS": "invalid",
        "SCHOLAR_RAG_RETRIEVAL_TIMEOUT_SECONDS": "invalid",
    }
    if ambient == "invalid-environment":
        env.update(invalid)
    elif ambient == "invalid-dotenv":
        for key in invalid:
            env.pop(key, None)
        (tmp_path / ".env").write_text(
            "\n".join(f"{key}={value}" for key, value in invalid.items()), encoding="utf-8"
        )
    command = """
import runpy
import sys
import httpx

def no_network(*args, **kwargs):
    raise AssertionError("Offline comparison demo attempted network access")

httpx.HTTPTransport.handle_request = no_network
httpx.AsyncHTTPTransport.handle_async_request = no_network
sys.argv = ["scripts.demo_run_comparison", "--output-dir", sys.argv[1]]
runpy.run_module("scripts.demo_run_comparison", run_name="__main__")
"""
    output = tmp_path / "output"
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", command, str(output)],
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
    bundle = EvidenceBundle.model_validate_json((output / "baseline.json").read_bytes())
    assert bundle.agent_id == "local-agent"
    assert bundle.generation.provider == "fake"
    assert bundle.configuration.max_source_docs == 50
    assert "synthetic-unused" not in bundle.model_dump_json()
