"""Require executed offline artifacts, measured GIFs, and isolation from ambient settings."""

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

from agent.evidence import EvidenceBundle
from agent.evidence_html import render_html

OUTPUT_NAMES = (
    "evidence.json",
    "evidence.md",
    "evidence.html",
    "restarted.html",
    "checks.json",
    "transcript.txt",
)


def test_demo_saves_actual_exports_and_a_reproducible_measured_illustration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.create_evidence_html_gif import create_gif
    from scripts.demo_evidence_html import run_demo

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    output = tmp_path / "output"
    transcript = run_demo(output)
    assert {path.name for path in output.iterdir()} == set(OUTPUT_NAMES)
    assert transcript == (output / "transcript.txt").read_text(encoding="utf-8")
    bundle = EvidenceBundle.model_validate_json((output / "evidence.json").read_bytes())
    assert bundle.generation.provider == "fake" and bundle.generation.model_name is None
    assert len(bundle.snapshot.sources) == 2 and len(bundle.answer.citations) == 2
    html = (output / "evidence.html").read_bytes()
    assert html == render_html(bundle).encode("utf-8")
    assert html == (output / "restarted.html").read_bytes()
    assert b"&lt;script&gt;" in html and b"<script>" not in html
    checks = json.loads((output / "checks.json").read_text(encoding="utf-8"))
    assert checks["html_bytes"] == len(html)
    assert checks["html_sha256"] == {
        "before": hashlib.sha256(html).hexdigest(),
        "after": hashlib.sha256(html).hexdigest(),
    }
    assert checks["source_links"] == 6 and checks["all_fragment_links_resolve"] is True
    assert checks["captured_sources"] == 2 and checks["final_citations"] == 2
    assert checks["events_unchanged"] is True
    assert checks["all_formats_unchanged"] is True
    assert checks["temporary_database_removed"] is True
    for key in (
        "export_agent_or_corpus_calls",
        "export_event_writes",
        "network_calls",
        "remaining_chunks",
    ):
        assert checks[key] == 0
    assert "format=html -> 200" in transcript and "Source links checked: 6" in transcript
    assert "not scientific" in transcript
    assert not list(tmp_path.rglob("*.sqlite3"))
    assert run_demo(tmp_path / "repeat") == transcript
    first, second = tmp_path / "first.gif", tmp_path / "second.gif"
    create_gif(output / "transcript.txt", first)
    create_gif(output / "transcript.txt", second)
    assert first.read_bytes() == second.read_bytes()
    with Image.open(first) as image:
        assert image.is_animated and image.n_frames == 4
        assert image.size == (1120, 540) and image.info["loop"] == 0
        frames = set()
        for index in range(image.n_frames):
            image.seek(index)
            assert image.info["duration"] == 3500
            frames.add(hashlib.sha256(image.convert("RGB").tobytes()).hexdigest())
        assert len(frames) == 4
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        create_gif(output / "transcript.txt", first)


@pytest.mark.parametrize("name", OUTPUT_NAMES)
def test_demo_preserves_existing_named_artifacts(tmp_path: Path, name: str) -> None:
    from scripts.demo_evidence_html import run_demo

    existing = tmp_path / name
    existing.write_text("Keep this existing file", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_demo(tmp_path)
    assert existing.read_text(encoding="utf-8") == "Keep this existing file"
    assert {path.name for path in tmp_path.iterdir()} == {name}


def test_demo_and_gif_refuse_dangling_symlinks(tmp_path: Path) -> None:
    from scripts.create_evidence_html_gif import create_gif
    from scripts.demo_evidence_html import run_demo

    target = tmp_path / "evidence.html"
    target.symlink_to(tmp_path / "not-created")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_demo(tmp_path)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        create_gif(tmp_path / "not-read.txt", target)
    assert target.is_symlink() and not (tmp_path / "not-created").exists()


@pytest.mark.parametrize("damage", ["unrelated", "horizontal", "vertical"])
def test_gif_rejects_wrong_or_clipped_transcript(tmp_path: Path, damage: str) -> None:
    from scripts.create_evidence_html_gif import create_gif
    from scripts.demo_evidence_html import PANEL_TITLES

    if damage == "unrelated":
        transcript = "\n\n".join(["Unrelated demo"] * 4)
    else:
        line = "W" * 90 if damage == "horizontal" else "excessive text " * 1000
        transcript = "\n\n".join(f"{title}\n{line}" for title in PANEL_TITLES)
    source = tmp_path / "transcript.txt"
    source.write_text(transcript, encoding="utf-8")
    target = tmp_path / "reader.gif"
    with pytest.raises(ValueError):
        create_gif(source, target)
    assert not target.exists()


def test_demo_cli_ignores_environment_dotenv_keys_and_user_database(tmp_path: Path) -> None:
    sentinel = tmp_path / "user.sqlite3"
    with closing(sqlite3.connect(sentinel)) as connection, connection:
        connection.execute("CREATE TABLE marker (value TEXT)")
        connection.execute("INSERT INTO marker VALUES ('do not touch')")
    before = sentinel.read_bytes()
    (tmp_path / ".env").write_text("SCHOLAR_RAG_CHUNK_SIZE=not-an-integer\n", encoding="utf-8")
    env = {
        **os.environ,
        "SCHOLAR_RAG_DATABASE_PATH": str(sentinel),
        "SCHOLAR_RAG_AGENT_ID": "do-not-use-ambient-agent",
        "SCHOLAR_RAG_DEFAULT_MODEL": "anthropic",
        "SCHOLAR_RAG_MAX_HOPS": "invalid",
        "SCHOLAR_RAG_MAX_SOURCE_DOCS": "0",
        "OPENAI_API_KEY": "synthetic-unused-key",
        "ANTHROPIC_API_KEY": "synthetic-unused-key",
        "GEMINI_API_KEY": "synthetic-unused-key",
        "MOONSHOT_API_KEY": "synthetic-unused-key",
    }
    output = tmp_path / "output"
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "scripts.demo_evidence_html", "--output-dir", str(output)],
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
    bundle = EvidenceBundle.model_validate_json((output / "evidence.json").read_bytes())
    assert bundle.agent_id == "local-agent" and bundle.generation.provider == "fake"
    for path in output.iterdir():
        assert b"synthetic-unused-key" not in path.read_bytes()
