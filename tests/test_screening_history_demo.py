"""The published GIF is derived from executed, guarded synthetic history reads."""

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image
from scripts.create_screening_history_gif import create_gif
from scripts.demo_screening_history import run_demo


def test_executed_demo_and_distinct_animation_frames(tmp_path: Path) -> None:
    output = tmp_path / "demo"
    transcript = run_demo(output)
    assert transcript == (output / "transcript.txt").read_text()
    checks = json.loads((output / "checks.json").read_text())
    assert checks["legacy_before_items"] == 0
    assert checks["legacy_first_retained_revision"] == 2
    assert checks["legacy_unretained_decision_revisions"] == 1
    assert checks["decisions_newest_first"] == ["unsure", "exclude", "include"]
    assert checks["first_page_revisions"] == [3, 2] and checks["older_page_revisions"] == [1]
    assert checks["rejected_retry_status"] == checks["stale_collection_status"] == 409
    assert checks["current_status"] == "stale"
    for key in (
        "retained_decisions_equal_submissions",
        "python_http_equal",
        "exact_identity_and_return",
        "restart_html_identical",
        "database_bytes_unchanged_during_reads",
        "temporary_database_removed",
    ):
        assert checks[key]
    assert checks["event_count"] == 0
    assert not any(checks["forbidden_call_counts"].values())
    image_path = tmp_path / "history.gif"
    create_gif(output / "transcript.txt", image_path)
    hashes = set()
    with Image.open(image_path) as image:
        assert image.n_frames == 4 and image.size == (1120, 540)
        for index in range(image.n_frames):
            image.seek(index)
            assert image.info["duration"] == 3500
            hashes.add(hashlib.sha256(image.convert("RGB").tobytes()).hexdigest())
    assert len(hashes) == 4
    assert image_path.stat().st_size < 300_000
    with pytest.raises(FileExistsError):
        run_demo(output)
    with pytest.raises(FileExistsError):
        create_gif(output / "transcript.txt", image_path)


def test_renderer_rejects_non_demo_transcript(tmp_path: Path) -> None:
    transcript = tmp_path / "invalid.txt"
    transcript.write_text("Not executed output")
    with pytest.raises(ValueError):
        create_gif(transcript, tmp_path / "invalid.gif")
