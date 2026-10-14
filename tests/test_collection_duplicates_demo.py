"""Measured artifacts, no-clobber output, and an environment-isolated offline demonstration."""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from PIL import Image
from scripts import create_collection_duplicates_gif, demo_collection_duplicates
from scripts.create_collection_duplicates_gif import create_gif
from scripts.demo_collection_duplicates import OUTPUT_NAMES, PANEL_TITLES, run_demo

from api.dependencies import AppContainer
from storage.collection_duplicates import CollectionDuplicateReport
from storage.paper_collections import PaperCollection


def test_actual_outputs_restart_and_reproducible_gif(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    directory = tmp_path / "output"
    transcript = run_demo(directory)
    assert {path.name for path in directory.iterdir()} == set(OUTPUT_NAMES)
    assert transcript == (directory / "transcript.txt").read_text(encoding="utf-8")
    collection = PaperCollection.model_validate_json((directory / "collection.json").read_bytes())
    report = CollectionDuplicateReport.model_validate_json(
        (directory / "duplicates.json").read_bytes()
    )
    checks = json.loads((directory / "checks.json").read_bytes())
    assert report.collection_id == collection.collection_id
    assert report.inspected_count == checks["selected_documents"] == 7
    assert len(report.groups) == checks["candidate_groups"] == 2
    assert len(report.singletons) == checks["singleton_documents"] == 3
    assert checks["case_sensitive_singletons"] == ["Paper", "paper"]
    for key in (
        "complete_partition",
        "outside_collection_absent",
        "inspection_database_unchanged",
        "python_matches_http",
        "temporary_database_removed",
    ):
        assert checks[key] is True
    for key in ("inspection_and_preview_forbidden_calls", "network_calls", "event_count"):
        assert checks[key] == 0
    assert (
        checks["explicit_human_selected_ids"]
        == checks["preview_document_ids"]
        == ["doi-a", "title-a"]
    )
    preview = json.loads((directory / "preview.json").read_bytes())
    assert {source["chunk"]["document_id"] for source in preview["sources"]} == {"doi-a", "title-a"}
    for format, original, restarted in (
        ("json", "duplicates.json", "restarted.json"),
        ("markdown", "duplicates.md", "restarted.md"),
    ):
        content = (directory / original).read_bytes()
        assert content == (directory / restarted).read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        assert checks["restart_sha256"][format] == {"before": digest, "after": digest}
        assert checks["response_bytes"][format] == len(content)
    assert not list(tmp_path.rglob("*.sqlite3"))
    first, second = tmp_path / "first.gif", tmp_path / "second.gif"
    create_gif(directory / "transcript.txt", first)
    create_gif(directory / "transcript.txt", second)
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
        create_gif(directory / "transcript.txt", first)


@pytest.mark.parametrize("name", OUTPUT_NAMES)
def test_demo_refuses_any_existing_artifact(tmp_path: Path, name: str) -> None:
    target = tmp_path / name
    target.write_bytes(b"Keep this artifact")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_demo(tmp_path)
    assert {path.name for path in tmp_path.iterdir()} == {name}
    assert target.read_bytes() == b"Keep this artifact"


def test_demo_preserves_late_artifact_and_cleans_temporary_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    original = demo_collection_duplicates._seed
    target = tmp_path / "duplicates.json"

    def seed(container: AppContainer) -> tuple[str, ...]:
        target.write_bytes(b"Concurrent artifact")
        return original(container)

    monkeypatch.setattr(demo_collection_duplicates, "_seed", seed)
    with pytest.raises(FileExistsError):
        run_demo(tmp_path)
    assert target.read_bytes() == b"Concurrent artifact"
    assert not list(tmp_path.rglob("*.sqlite3"))


def test_demo_and_gif_refuse_dangling_symlinks(tmp_path: Path) -> None:
    target = tmp_path / "duplicates.json"
    target.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        run_demo(tmp_path)
    with pytest.raises(FileExistsError):
        create_gif(tmp_path / "not-read.txt", target)
    assert target.is_symlink()


def test_gif_preserves_a_destination_created_during_rendering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, target = tmp_path / "transcript.txt", tmp_path / "illustration.gif"
    source.write_text("\n\n".join(PANEL_TITLES), encoding="utf-8")
    original = create_collection_duplicates_gif.render_panels

    def render(panels: list[str], output: Path, *, banner: str) -> None:
        target.write_bytes(b"Concurrent GIF")
        original(panels, output, banner=banner)

    monkeypatch.setattr(create_collection_duplicates_gif, "render_panels", render)
    with pytest.raises(FileExistsError):
        create_gif(source, target)
    assert target.read_bytes() == b"Concurrent GIF"


@pytest.mark.parametrize("damage", ["unrelated", "overflow"])
def test_gif_rejects_invalid_or_overflowing_transcript(tmp_path: Path, damage: str) -> None:
    body = "\n\n".join(
        ["Wrong demo"] * 4
        if damage == "unrelated"
        else [f"{title}\n" + "too much text " * 1000 for title in PANEL_TITLES]
    )
    source, target = tmp_path / "transcript.txt", tmp_path / "not-created.gif"
    source.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError):
        create_gif(source, target)
    assert not target.exists()


@pytest.mark.parametrize("ambient", ["environment", "dotenv"])
def test_cli_ignores_ambient_settings_keys_and_database(tmp_path: Path, ambient: str) -> None:
    sentinel = tmp_path / "unrelated.sqlite3"
    sentinel.write_bytes(b"Never open the ambient database")
    values = {
        "SCHOLAR_RAG_DATABASE_PATH": str(sentinel),
        "SCHOLAR_RAG_DEFAULT_MODEL": "anthropic",
        "SCHOLAR_RAG_MAX_HOPS": "invalid",
        "SCHOLAR_RAG_MAX_SOURCE_DOCS": "0",
        "OPENAI_API_KEY": "synthetic-unused-key",
        "ANTHROPIC_API_KEY": "synthetic-unused-key",
        "GEMINI_API_KEY": "synthetic-unused-key",
        "MOONSHOT_API_KEY": "synthetic-unused-key",
    }
    environment = dict(os.environ)
    if ambient == "environment":
        environment.update(values)
    else:
        for name in values:
            environment.pop(name, None)
        (tmp_path / ".env").write_text(
            "\n".join(f"{key}={value}" for key, value in values.items()), encoding="utf-8"
        )
    output = tmp_path / "output"
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "scripts.demo_collection_duplicates", "--output-dir", str(output)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert sentinel.read_bytes() == b"Never open the ambient database"
    assert not (tmp_path / ".scholar-rag-agent.sqlite3").exists()
    for artifact in output.iterdir():
        assert b"synthetic-unused-key" not in artifact.read_bytes()
