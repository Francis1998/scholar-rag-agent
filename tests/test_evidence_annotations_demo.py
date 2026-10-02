"""Measured API artifacts, renderer bounds, and offline demo isolation."""

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from contextlib import ExitStack, closing
from pathlib import Path
from unittest.mock import Mock

import pytest
from PIL import Image
from scripts import create_evidence_annotations_gif, demo_evidence_annotations
from scripts.create_evidence_annotations_gif import create_gif
from scripts.demo_evidence_annotations import (
    DEMO_QUOTE,
    DEMO_TEXT,
    OUTPUT_NAMES,
    PANEL_TITLES,
    run_demo,
)

from agent.evidence import EvidenceBundle, text_digest
from api.dependencies import AppContainer
from storage.evidence_annotations import (
    AnnotationPage,
    AnnotationSubmission,
    SavedEvidenceAnnotation,
)


def test_real_results_measured_transcript_and_readable_reproducible_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    output = tmp_path / "output"
    transcript = run_demo(output)
    assert {file.name for file in output.iterdir()} == set(OUTPUT_NAMES)
    assert transcript == (output / "transcript.txt").read_text(encoding="utf-8")
    assert run_demo(tmp_path / "repeat") == transcript
    bundle = EvidenceBundle.model_validate_json((output / "evidence.json").read_bytes())
    submission = AnnotationSubmission.model_validate_json((output / "submission.json").read_bytes())
    saved = SavedEvidenceAnnotation.model_validate_json((output / "created.json").read_bytes())
    history = AnnotationPage.model_validate_json((output / "annotations.json").read_bytes())
    assert bundle.generation.provider == "fake"
    assert bundle.snapshot.sources[0].chunk.text == DEMO_TEXT
    assert submission.start == saved.start == DEMO_TEXT.rindex(DEMO_QUOTE)
    assert saved.start != DEMO_TEXT.index(DEMO_QUOTE)
    assert DEMO_TEXT[saved.start : saved.end] == saved.quote == DEMO_QUOTE
    assert saved.document_id == bundle.snapshot.sources[0].chunk.document_id
    assert saved.chunk_id == bundle.snapshot.sources[0].chunk.chunk_id
    assert saved.source_text_sha256 == text_digest(DEMO_TEXT)
    assert (output / "replayed.json").read_bytes() == (output / "created.json").read_bytes()
    assert (output / "restarted.json").read_bytes() == (output / "annotations.json").read_bytes()
    assert len(history.annotations) == 2
    assert history.annotations[-1] == saved and history.next_cursor is None
    first = AnnotationPage.model_validate_json((output / "first-page.json").read_bytes())
    second = AnnotationPage.model_validate_json((output / "second-page.json").read_bytes())
    assert first.annotations + second.annotations == history.annotations
    assert first.next_cursor == first.annotations[0].sequence and second.next_cursor is None
    checks = json.loads((output / "checks.json").read_text())
    assert checks["synthetic_only"] is True and checks["setup_provider"] == "fake"
    assert checks["created_status"] == 201 and checks["replay_status"] == 200
    assert checks["conflict_status"] == 409 and checks["conflict_code"] == "annotation_id_conflict"
    assert checks["selected_start"] == saved.start and checks["selected_end"] == saved.end
    assert checks["first_occurrence_start"] == DEMO_TEXT.index(DEMO_QUOTE)
    assert checks["span_codepoints"] == len(DEMO_QUOTE)
    assert checks["annotation_count"] == 2 and checks["frozen_sources"] == 1
    assert checks["source_text_sha256"] == text_digest(DEMO_TEXT)
    assert checks["first_page_sequences"] == [first.annotations[0].sequence]
    assert checks["next_cursor"] == first.next_cursor
    assert checks["second_page_sequences"] == [saved.sequence]
    for flag in ("events_unchanged", "restart_identical", "temporary_database_removed"):
        assert checks[flag] is True
    for count in (
        "annotation_agent_or_corpus_calls",
        "annotation_event_writes",
        "network_calls",
        "remaining_corpus_chunks",
    ):
        assert checks[count] == 0
    digest = hashlib.sha256((output / "annotations.json").read_bytes()).hexdigest()
    assert checks["history_sha256"] == {"before": digest, "after": digest}
    assert checks["history_bytes"] == (output / "annotations.json").stat().st_size
    assert not list(tmp_path.rglob("*.sqlite3"))
    assert saved.quote in transcript and saved.note in transcript
    assert f"Selected span: [{saved.start}, {saved.end})" in transcript
    first_gif, second_gif = tmp_path / "one.gif", tmp_path / "two.gif"
    create_gif(output / "transcript.txt", first_gif)
    create_gif(output / "transcript.txt", second_gif)
    assert first_gif.read_bytes() == second_gif.read_bytes()
    with Image.open(first_gif) as image:
        assert image.format == "GIF" and image.is_animated and image.n_frames == 4
        assert image.size == (1120, 540) and image.info["loop"] == 0
        hashes = set()
        for frame in range(4):
            image.seek(frame)
            assert image.info["duration"] == 3500
            hashes.add(hashlib.sha256(image.convert("RGB").tobytes()).hexdigest())
        assert len(hashes) == 4
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        create_gif(output / "transcript.txt", first_gif)


@pytest.mark.parametrize("kind", ["directory", "file", "symlink"])
def test_existing_output_targets_are_never_reused_or_overwritten(tmp_path: Path, kind: str) -> None:
    target = tmp_path / "output"
    if kind == "directory":
        target.mkdir()
        (target / "keep.txt").write_text("Existing content")
    elif kind == "file":
        target.write_text("Existing content")
    else:
        target.symlink_to(tmp_path / "absent")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_demo(target)
    if kind == "directory":
        assert [file.name for file in target.iterdir()] == ["keep.txt"]
    elif kind == "file":
        assert target.read_text() == "Existing content"
    else:
        assert target.is_symlink() and not (tmp_path / "absent").exists()


def test_demo_preserves_an_artifact_created_after_directory_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    target = output / "evidence.json"
    block = demo_evidence_annotations._block_annotation_work

    def late_writer(stack: ExitStack, container: AppContainer) -> list[Mock]:
        target.write_bytes(b"Created by another writer")
        return block(stack, container)

    monkeypatch.setattr(demo_evidence_annotations, "_block_annotation_work", late_writer)
    with pytest.raises(FileExistsError):
        run_demo(output)
    assert target.read_bytes() == b"Created by another writer"


@pytest.mark.parametrize("damage", ["unrelated", "empty", "horizontal", "vertical", "oversized"])
def test_renderer_rejects_invalid_or_clipped_transcripts(tmp_path: Path, damage: str) -> None:
    bodies = {
        "unrelated": "\n\n".join(["Not this demo\nbody"] * 4),
        "empty": "\n\n".join(PANEL_TITLES),
        "horizontal": "\n\n".join(f"{title}\n{'W' * 90}" for title in PANEL_TITLES),
        "vertical": "\n\n".join(f"{title}\n{'word ' * 300}" for title in PANEL_TITLES),
        "oversized": "x" * 16385,
    }
    transcript = tmp_path / "transcript.txt"
    transcript.write_text(bodies[damage])
    output = tmp_path / "not-created.gif"
    with pytest.raises(ValueError):
        create_gif(transcript, output)
    assert not output.exists()


def test_renderer_preserves_late_writer_and_dangling_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "transcript.txt"
    source.write_text("\n\n".join(f"{title}\nMeasured body" for title in PANEL_TITLES))
    target = tmp_path / "late.gif"
    original = create_evidence_annotations_gif.render_panels

    def late_writer(panels: list[str], output_path: Path, *, banner: str) -> None:
        original(panels, output_path, banner=banner)
        target.write_bytes(b"Concurrent output")

    monkeypatch.setattr(create_evidence_annotations_gif, "render_panels", late_writer)
    with pytest.raises(FileExistsError):
        create_gif(source, target)
    assert target.read_bytes() == b"Concurrent output"
    dangling = tmp_path / "dangling.gif"
    dangling.symlink_to(tmp_path / "absent")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        create_gif(tmp_path / "not-read.txt", dangling)
    assert dangling.is_symlink()


@pytest.mark.parametrize("ambient", ["environment", "dotenv"])
def test_cli_ignores_ambient_keys_invalid_configuration_and_user_database(
    tmp_path: Path, ambient: str
) -> None:
    sentinel = tmp_path / "user.sqlite3"
    with closing(sqlite3.connect(sentinel)) as connection, connection:
        connection.execute("CREATE TABLE marker (value TEXT)")
        connection.execute("INSERT INTO marker VALUES ('Keep user data')")
    before = sentinel.read_bytes()
    env = {
        **os.environ,
        "SCHOLAR_RAG_DATABASE_PATH": str(sentinel),
        "SCHOLAR_RAG_AGENT_ID": "unused-ambient-agent",
        "SCHOLAR_RAG_DEFAULT_MODEL": "anthropic",
        "OPENAI_API_KEY": "unused-synthetic-key",
        "ANTHROPIC_API_KEY": "unused-synthetic-key",
        "GEMINI_API_KEY": "unused-synthetic-key",
        "MOONSHOT_API_KEY": "unused-synthetic-key",
    }
    invalid = {"SCHOLAR_RAG_MAX_HOPS": "invalid", "SCHOLAR_RAG_MAX_SOURCE_DOCS": "0"}
    if ambient == "environment":
        env.update(invalid)
    else:
        for name in invalid:
            env.pop(name, None)
        (tmp_path / ".env").write_text(
            "\n".join(f"{key}={value}" for key, value in invalid.items())
        )
    output = tmp_path / "output"
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "scripts.demo_evidence_annotations", "--output-dir", str(output)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert sentinel.read_bytes() == before
    assert not (tmp_path / ".scholar-rag-agent.sqlite3").exists()
    bundle = EvidenceBundle.model_validate_json((output / "evidence.json").read_bytes())
    assert bundle.agent_id == "local-agent" and bundle.generation.provider == "fake"
    for path in output.iterdir():
        assert b"unused-synthetic-key" not in path.read_bytes()
