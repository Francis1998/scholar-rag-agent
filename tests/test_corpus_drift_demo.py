"""Actual-output and offline-isolation contracts for the corpus-drift demonstration."""

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from contextlib import closing
from pathlib import Path

import pytest
from PIL import Image
from scripts.create_corpus_drift_gif import create_gif
from scripts.demo_corpus_drift import OUTPUT_NAMES, PANEL_TITLES, run_demo

from agent.evidence import EvidenceBundle
from storage.corpus_drift import CorpusDriftReport


def test_actual_demo_artifacts_and_reproducible_gif(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    output = tmp_path / "output"
    transcript = run_demo(output)
    assert {file.name for file in output.iterdir()} == set(OUTPUT_NAMES)
    assert (output / "transcript.txt").read_text(encoding="utf-8") == transcript
    bundle = EvidenceBundle.model_validate_json((output / "bundle.json").read_bytes())
    initial = CorpusDriftReport.model_validate_json((output / "unchanged.json").read_bytes())
    drift = CorpusDriftReport.model_validate_json((output / "drift.json").read_bytes())
    assert bundle.generation.provider == "fake"
    assert len(bundle.snapshot.sources) == 5
    assert initial.counts.unchanged == 5
    assert drift.counts.model_dump() == {"total": 5, "unchanged": 1, "changed": 1, "missing": 3}
    assert [source.chunk_id for source in drift.sources] == [
        source.chunk.chunk_id for source in bundle.snapshot.sources
    ]
    assert (output / "drift.json").read_bytes() == (output / "restarted.json").read_bytes()
    checks = json.loads((output / "checks.json").read_text(encoding="utf-8"))
    for fmt, name in (("json", "bundle.json"), ("markdown", "bundle.md")):
        digest = hashlib.sha256((output / name).read_bytes()).hexdigest()
        assert checks["export_sha256"][fmt] == {"before": digest, "after": digest}
    for name in ("report_restart_byte_equal", "events_unchanged", "temporary_database_removed"):
        assert checks[name] is True
    for name in ("drift_agent_calls", "drift_event_writes", "network_calls"):
        assert checks[name] == 0
    assert not list(tmp_path.rglob("*.sqlite3"))
    assert "After edits: unchanged=1, changed=1, missing=3" in transcript
    assert "Drift agent calls: 0; event writes: 0" in transcript
    first, second = tmp_path / "first.gif", tmp_path / "second.gif"
    create_gif(output / "transcript.txt", first)
    create_gif(output / "transcript.txt", second)
    assert first.read_bytes() == second.read_bytes()
    with Image.open(first) as image:
        assert image.n_frames == 4
        assert image.size == (1120, 540)
        assert image.info["loop"] == 0
        for index in range(image.n_frames):
            image.seek(index)
            assert image.info["duration"] == 3500
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        create_gif(output / "transcript.txt", first)


@pytest.mark.parametrize("name", OUTPUT_NAMES)
def test_demo_refuses_output_collisions(tmp_path: Path, name: str) -> None:
    existing = tmp_path / name
    existing.write_text("Preserve this artifact", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_demo(tmp_path)
    assert existing.read_text(encoding="utf-8") == "Preserve this artifact"
    assert {file.name for file in tmp_path.iterdir()} == {name}


def test_demo_and_renderer_refuse_dangling_symlinks(tmp_path: Path) -> None:
    output = tmp_path / "bundle.json"
    output.symlink_to(tmp_path / "does-not-exist")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_demo(tmp_path)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        create_gif(tmp_path / "not-read.txt", output)
    assert output.is_symlink()


@pytest.mark.parametrize("damage", ["unrelated", "horizontal", "vertical"])
def test_gif_rejects_unrelated_or_overflowing_panels(tmp_path: Path, damage: str) -> None:
    if damage == "unrelated":
        transcript = "\n\n".join(["Not this demo"] * 4)
    else:
        body = "W" * 90 if damage == "horizontal" else "too much text " * 1000
        transcript = "\n\n".join(f"{title}\n{body}" for title in PANEL_TITLES)
    source = tmp_path / "transcript.txt"
    source.write_text(transcript, encoding="utf-8")
    destination = tmp_path / "not-created.gif"
    with pytest.raises(ValueError):
        create_gif(source, destination)
    assert not destination.exists()


@pytest.mark.parametrize("ambient", ["environment", "dotenv"])
def test_cli_ignores_ambient_settings_keys_and_database(tmp_path: Path, ambient: str) -> None:
    sentinel = tmp_path / "unrelated.sqlite3"
    with closing(sqlite3.connect(sentinel)) as connection, connection:
        connection.execute("CREATE TABLE marker (value TEXT)")
        connection.execute("INSERT INTO marker VALUES ('keep synthetic data')")
    before = sentinel.read_bytes()
    env = {
        **os.environ,
        "SCHOLAR_RAG_DATABASE_PATH": str(sentinel),
        "SCHOLAR_RAG_AGENT_ID": "must-not-use-ambient-agent",
        "SCHOLAR_RAG_DEFAULT_MODEL": "anthropic",
        "OPENAI_API_KEY": "synthetic-unused-secret",
        "ANTHROPIC_API_KEY": "synthetic-unused-secret",
        "GEMINI_API_KEY": "synthetic-unused-secret",
        "MOONSHOT_API_KEY": "synthetic-unused-secret",
    }
    invalid = {"SCHOLAR_RAG_MAX_HOPS": "invalid", "SCHOLAR_RAG_MAX_SOURCE_DOCS": "0"}
    if ambient == "environment":
        env.update(invalid)
    else:
        for name in invalid:
            env.pop(name, None)
        (tmp_path / ".env").write_text(
            "\n".join(f"{name}={value}" for name, value in invalid.items()), encoding="utf-8"
        )
    guard = """
import runpy
import socket
import sys

def no_network(*args, **kwargs):
    raise AssertionError("The corpus-drift demo attempted network access")

socket.create_connection = no_network
socket.socket.connect = no_network
sys.argv = ["scripts.demo_corpus_drift", "--output-dir", sys.argv[1]]
runpy.run_module("scripts.demo_corpus_drift", run_name="__main__")
"""
    output = tmp_path / "output"
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", guard, str(output)],
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
    bundle = EvidenceBundle.model_validate_json((output / "bundle.json").read_bytes())
    assert bundle.agent_id == "local-agent"
    assert "synthetic-unused-secret" not in bundle.model_dump_json()
