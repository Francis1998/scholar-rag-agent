"""The published illustration must derive from executed, guarded browser responses."""

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image
from scripts.create_collection_browser_gif import create_gif
from scripts.demo_collection_browser import SELECTED_ID, run_demo


def test_executed_demo_and_each_animation_frame(tmp_path: Path) -> None:
    output = tmp_path / "demo"
    transcript = run_demo(output)
    assert transcript == (output / "transcript.txt").read_text()
    checks = json.loads((output / "checks.json").read_text())
    assert checks["members"] == 5
    assert set(checks["counts"].values()) == {1}
    assert checks["current_include_ids"] == [SELECTED_ID]
    assert checks["database_bytes_unchanged"] and checks["restart_html_identical"]
    assert checks["event_count"] == 0
    assert not any(checks["forbidden_call_counts"].values())
    assert checks["temporary_database_removed"]
    assert checks["mismatched_revision_status"] == 409
    image_path = tmp_path / "demo.gif"
    create_gif(output / "transcript.txt", image_path)
    hashes = set()
    with Image.open(image_path) as image:
        assert image.n_frames == 4
        assert image.size == (1120, 540)
        for index in range(image.n_frames):
            image.seek(index)
            assert image.info["duration"] > 0
            hashes.add(hashlib.sha256(image.convert("RGB").tobytes()).hexdigest())
    assert len(hashes) == 4
    with pytest.raises(FileExistsError):
        run_demo(output)
    with pytest.raises(FileExistsError):
        create_gif(output / "transcript.txt", image_path)


def test_renderer_rejects_a_non_demo_transcript(tmp_path: Path) -> None:
    transcript = tmp_path / "invalid.txt"
    transcript.write_text("Not measured output")
    with pytest.raises(ValueError):
        create_gif(transcript, tmp_path / "invalid.gif")
