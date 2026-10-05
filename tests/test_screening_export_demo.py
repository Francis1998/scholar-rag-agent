"""The export GIF is derived from actual, reproducible offline downloads."""

import csv
import hashlib
import io
import json
from pathlib import Path

import pytest
from PIL import Image
from scripts.demo_screening_export import create_gif, run_demo


def test_demo_saves_actual_downloads_and_four_measured_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-used")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-be-used")
    monkeypatch.setenv("SCHOLAR_RAG_ANTHROPIC_MODEL", "")
    monkeypatch.setenv("SCHOLAR_RAG_DATABASE_PATH", str(tmp_path / "must-not-exist.sqlite3"))
    output = tmp_path / "demo"
    transcript = run_demo(output)
    data = json.loads((output / "screening-results.json").read_bytes())
    metrics = json.loads((output / "measurements.json").read_bytes())
    rows = list(
        csv.DictReader(
            io.StringIO((output / "screening-results.csv").read_bytes().decode(), newline="")
        )
    )
    assert metrics["synthetic_only"] is True
    assert metrics["queue_page_items"] == 20
    assert metrics["export_members"] == len(data["items"]) == metrics["csv_rows"] == len(rows) == 25
    assert (
        metrics["counts"]
        == data["counts"]
        == {
            "include": 1,
            "exclude": 1,
            "unsure": 1,
            "stale": 1,
            "unscreened": 21,
        }
    )
    assert metrics["included_document_ids"] == data["included_document_ids"]
    assert metrics["stale_include_excluded"] is True
    assert metrics["csv_text_round_trip"] is True
    assert metrics["title_truncations"] == metrics["source_truncations"] == 1
    assert metrics["restart_bytes_equal"] is True
    assert metrics["export_database_bytes_unchanged"] is True
    assert metrics["stale_request_status"] == 409
    for key in (
        "external_http_attempts",
        "generation_attempts",
        "retrieval_attempts",
        "agent_event_count",
    ):
        assert metrics[key] == 0
    for format in ("json", "csv"):
        content = (output / f"screening-results.{format}").read_bytes()
        assert metrics["downloads"][format]["bytes"] == len(content)
        assert metrics["downloads"][format]["sha256"] == hashlib.sha256(content).hexdigest()
        assert str(len(content)) in transcript
    assert transcript == (output / "transcript.txt").read_text()
    assert not (tmp_path / "must-not-exist.sqlite3").exists()
    assert not list(output.glob("*.sqlite3"))
    gif = output / "screening-exports.gif"
    create_gif(transcript, gif)
    with Image.open(gif) as image:
        assert image.format == "GIF"
        assert image.n_frames == 4
        assert image.size == (1120, 540)
        frames = []
        for frame in range(image.n_frames):
            image.seek(frame)
            assert image.info["duration"] == 3500
            frames.append(image.convert("RGB").tobytes())
        assert len(set(frames)) == 4
    assert gif.stat().st_size > 1000
    with pytest.raises(FileExistsError):
        create_gif(transcript, gif)
    with pytest.raises(FileExistsError):
        run_demo(output)


def test_demo_never_overwrites_existing_or_symlinked_outputs(tmp_path: Path) -> None:
    sentinel = tmp_path / "saved.txt"
    sentinel.write_text("keep this file")
    (tmp_path / "screening-results.csv").symlink_to(sentinel)
    with pytest.raises(FileExistsError):
        run_demo(tmp_path)
    assert sentinel.read_text() == "keep this file"
    assert not (tmp_path / "screening-results.json").exists()
