"""The portable notebook showcase is derived from executed synthetic output."""

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image
from scripts.create_notebook_export_gif import create_preview
from scripts.demo_notebook_export import OUTPUT_NAMES, RECORD_COUNT, run_demo

from storage.review_notebook import ReviewNotebook


def test_executed_demo_preserves_all_records_and_produces_distinct_frames(tmp_path: Path) -> None:
    output = tmp_path / "notebook-demo"
    transcript = run_demo(output)
    assert {path.name for path in output.iterdir()} == set(OUTPUT_NAMES)
    assert (output / "transcript.txt").read_text() == transcript
    checks = json.loads((output / "checks.json").read_text())
    notebook = ReviewNotebook.model_validate_json((output / "notebook.json").read_bytes())
    assert len(notebook.reviews) == len(notebook.annotations) == RECORD_COUNT == 23
    assert notebook.evidence.generation.provider == "fake"
    assert (output / "notebook.md").read_bytes() == notebook.to_markdown().encode("utf-8")
    assert checks["first_page_reviews"] == checks["first_page_annotations"] == 20
    assert checks["review_sequences"] == checks["annotation_sequences"] == list(range(23, 0, -1))
    assert checks["decisions"] == ["accepted", "needs_revision", "rejected"]
    assert checks["events_before"] == checks["events_after"] == 8
    assert checks["database_removed"] and checks["offline_json_reproduces_markdown"]
    assert checks["corpus_change_delete_restart_identical"] and checks["synthetic_only"]
    assert checks["socket_connection_calls"] == checks["remaining_corpus_chunks"] == 0
    assert not any(checks["forbidden_call_counts"].values())
    assert not list(output.glob("*.sqlite3"))
    selected = notebook.annotations[0]
    text = notebook.evidence.snapshot.sources[0].chunk.text
    assert selected.start == text.rindex(selected.quote) != text.index(selected.quote)
    assert selected.start == checks["selected_start"] and selected.end == checks["selected_end"]
    assert text[selected.start : selected.end] == selected.quote
    assert checks["source_text_sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert str(selected.annotation_id) in transcript and selected.quote in transcript
    for format, extension in (("json", "json"), ("markdown", "md")):
        content = (output / f"notebook.{extension}").read_bytes()
        assert checks["bytes"][format] == len(content)
        assert checks["sha256"][format] == hashlib.sha256(content).hexdigest()
    gif, still = tmp_path / "preview.gif", tmp_path / "preview.png"
    create_preview(output / "transcript.txt", gif, still)
    with Image.open(gif) as image, Image.open(still) as preview:
        assert image.format == "GIF" and image.is_animated and image.n_frames == 4
        assert image.size == preview.size == (1120, 540)
        assert image.info["loop"] == 0
        assert image.convert("RGB").tobytes() == preview.convert("RGB").tobytes()
        hashes = set()
        for number in range(image.n_frames):
            image.seek(number)
            assert image.info["duration"] == 3500
            hashes.add(hashlib.sha256(image.convert("RGB").tobytes()).hexdigest())
        assert len(hashes) == 4
    with pytest.raises(FileExistsError):
        create_preview(output / "transcript.txt", gif, still)


@pytest.mark.parametrize("kind", ["directory", "file", "symlink"])
def test_demo_does_not_overwrite_existing_output(tmp_path: Path, kind: str) -> None:
    target = tmp_path / "existing"
    if kind == "directory":
        target.mkdir()
    elif kind == "file":
        target.write_text("Keep")
    else:
        target.symlink_to(tmp_path / "absent")
    with pytest.raises(FileExistsError):
        run_demo(target)
    if kind == "file":
        assert target.read_text() == "Keep"


def test_demo_ignores_ambient_configuration_and_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentinel = tmp_path / "user.sqlite3"
    sentinel.write_bytes(b"DO NOT OPEN")
    monkeypatch.setenv("SCHOLAR_RAG_DATABASE_PATH", str(sentinel))
    monkeypatch.setenv("SCHOLAR_RAG_MAX_SOURCE_DOCS", "invalid")
    monkeypatch.setenv("SCHOLAR_RAG_DEFAULT_MODEL", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unused-synthetic-sentinel")
    output = tmp_path / "demo"
    run_demo(output)
    assert sentinel.read_bytes() == b"DO NOT OPEN"
    for artifact in output.iterdir():
        assert b"unused-synthetic-sentinel" not in artifact.read_bytes()


@pytest.mark.parametrize("text", ["unrelated transcript", "x" * 16385])
def test_preview_refuses_unrelated_or_oversized_transcripts(tmp_path: Path, text: str) -> None:
    transcript = tmp_path / "transcript.txt"
    transcript.write_text(text)
    with pytest.raises(ValueError):
        create_preview(transcript, tmp_path / "preview.gif", tmp_path / "preview.png")
    assert not (tmp_path / "preview.gif").exists()
    assert not (tmp_path / "preview.png").exists()


def test_committed_previews_match_the_measured_transcript(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    assets = root / "docs/assets"
    gif, still = tmp_path / "reproduced.gif", tmp_path / "reproduced.png"
    create_preview(assets / "portable-review-notebook.txt", gif, still)
    assert gif.read_bytes() == (assets / "portable-review-notebook.gif").read_bytes()
    assert still.read_bytes() == (assets / "portable-review-notebook.png").read_bytes()
    for document in (root / "README.md", root / "docs/guides/PORTABLE_REVIEW_NOTEBOOK_GUIDE.md"):
        text = document.read_text()
        assert "portable-review-notebook.png" in text
        assert "portable-review-notebook.gif" in text
        assert "portable-review-notebook.txt" in text
