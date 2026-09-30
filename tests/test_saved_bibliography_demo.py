"""Actual-output, GIF reproducibility, and ambient-settings isolation contracts."""

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
from scripts import demo_saved_bibliography
from scripts.create_saved_bibliography_gif import create_gif
from scripts.demo_saved_bibliography import OUTPUT_NAMES, PANEL_TITLES, run_demo

from agent.evidence import EvidenceBundle
from api.dependencies import AppContainer
from storage.saved_bibliography import SavedBibliography


def test_actual_output_and_reproducible_animated_gif(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    output = tmp_path / "output"
    transcript = run_demo(output)
    assert {file.name for file in output.iterdir()} == set(OUTPUT_NAMES)
    assert (output / "transcript.txt").read_text(encoding="utf-8") == transcript
    evidence = EvidenceBundle.model_validate_json((output / "evidence.json").read_bytes())
    bibliography = SavedBibliography.model_validate_json(
        (output / "bibliography.json").read_bytes()
    )
    assert evidence.generation.provider == "fake"
    assert len(evidence.snapshot.sources) == 4 and len(evidence.answer.citations) == 3
    assert [source.document_id for source in bibliography.sources] == ["paper-a", "paper-b"]
    assert [chunk.chunk_id for chunk in bibliography.sources[0].cited_chunks] == ["a-1", "a-2"]
    assert bibliography.sources[0].metadata["year"] == "2024"
    assert "2025" not in bibliography.bibtex
    assert "uncited-paper" not in bibliography.to_json()
    assert (
        len([warning for warning in bibliography.warnings if warning.startswith("Conflicting")])
        == 1
    )
    assert (output / "bibliography.bib").read_bytes() == bibliography.bibtex.encode("utf-8")
    checks = json.loads((output / "checks.json").read_text(encoding="utf-8"))
    for format, before, after in (
        ("json", "bibliography.json", "restarted.json"),
        ("bibtex", "bibliography.bib", "restarted.bib"),
    ):
        assert (output / before).read_bytes() == (output / after).read_bytes()
        digest = hashlib.sha256((output / before).read_bytes()).hexdigest()
        assert checks["bibliography_sha256"][format] == {"before": digest, "after": digest}
        assert checks["bibliography_bytes"][format] == (output / before).stat().st_size
    for name in ("evidence.json", "evidence.md"):
        assert (
            checks["evidence_sha256"][name]
            == hashlib.sha256((output / name).read_bytes()).hexdigest()
        )
    assert checks["captured_sources"] == 4 and checks["final_citations"] == 3
    assert checks["cited_documents"] == 2 and checks["metadata_conflicts"] == 1
    assert checks["events_unchanged"] is True and checks["temporary_database_removed"] is True
    for name in (
        "export_agent_or_corpus_calls",
        "export_event_writes",
        "network_calls",
        "remaining_corpus_chunks",
    ):
        assert checks[name] == 0
    assert not list(tmp_path.rglob("*.sqlite3"))
    assert bibliography.bibtex.split("\n\n")[0] in transcript
    assert "Export agent/corpus calls: 0; event writes: 0" in transcript
    assert run_demo(tmp_path / "repeat") == transcript
    first, second = tmp_path / "first.gif", tmp_path / "second.gif"
    create_gif(output / "transcript.txt", first)
    create_gif(output / "transcript.txt", second)
    assert first.read_bytes() == second.read_bytes()
    with Image.open(first) as image:
        assert image.is_animated and image.n_frames == 4
        assert image.size == (1120, 540) and image.info["loop"] == 0
        frame_hashes = set()
        for index in range(image.n_frames):
            image.seek(index)
            assert image.info["duration"] == 3500
            frame_hashes.add(hashlib.sha256(image.convert("RGB").tobytes()).hexdigest())
        assert len(frame_hashes) == 4
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        create_gif(output / "transcript.txt", first)


@pytest.mark.parametrize("name", OUTPUT_NAMES)
def test_demo_refuses_colliding_artifacts(tmp_path: Path, name: str) -> None:
    target = tmp_path / name
    target.write_text("Preserve existing content", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_demo(tmp_path)
    assert target.read_text(encoding="utf-8") == "Preserve existing content"
    assert {file.name for file in tmp_path.iterdir()} == {name}


def test_demo_preserves_an_artifact_created_after_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "bibliography.bib"
    seed = demo_saved_bibliography._seed

    def seed_with_late_writer(container: AppContainer) -> None:
        seed(container)
        target.write_bytes(b"Created by another writer after preflight")

    monkeypatch.setattr(demo_saved_bibliography, "_seed", seed_with_late_writer)
    with pytest.raises(FileExistsError):
        run_demo(tmp_path)
    assert target.read_bytes() == b"Created by another writer after preflight"


def test_demo_and_renderer_refuse_dangling_symlinks(tmp_path: Path) -> None:
    target = tmp_path / "bibliography.bib"
    target.symlink_to(tmp_path / "does-not-exist")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_demo(tmp_path)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        create_gif(tmp_path / "not-read.txt", target)
    assert target.is_symlink()


@pytest.mark.parametrize("damage", ["unrelated", "horizontal", "vertical"])
def test_gif_rejects_wrong_or_overflowing_panels(tmp_path: Path, damage: str) -> None:
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
def test_cli_ignores_provider_keys_invalid_settings_and_user_database(
    tmp_path: Path, ambient: str
) -> None:
    sentinel = tmp_path / "unrelated.sqlite3"
    with closing(sqlite3.connect(sentinel)) as connection, connection:
        connection.execute("CREATE TABLE marker (value TEXT)")
        connection.execute("INSERT INTO marker VALUES ('keep this fixture')")
    before = sentinel.read_bytes()
    env = {
        **os.environ,
        "SCHOLAR_RAG_DATABASE_PATH": str(sentinel),
        "SCHOLAR_RAG_AGENT_ID": "must-not-use-ambient-agent",
        "SCHOLAR_RAG_DEFAULT_MODEL": "anthropic",
        "OPENAI_API_KEY": "synthetic-unused-key",
        "ANTHROPIC_API_KEY": "synthetic-unused-key",
        "GEMINI_API_KEY": "synthetic-unused-key",
        "MOONSHOT_API_KEY": "synthetic-unused-key",
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
    output = tmp_path / "output"
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "scripts.demo_saved_bibliography", "--output-dir", str(output)],
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
    evidence = EvidenceBundle.model_validate_json((output / "evidence.json").read_bytes())
    assert evidence.agent_id == "local-agent" and evidence.generation.provider == "fake"
    for path in output.iterdir():
        assert b"synthetic-unused-key" not in path.read_bytes()
