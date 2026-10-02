"""The showcase is rendered from actual synthetic API results, without model calls."""

import json
from pathlib import Path

import pytest
from PIL import Image
from scripts.demo_paper_screening import create_gif, run_demo


def test_demo_ignores_live_configuration_and_renders_four_measured_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-used")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-be-used")
    monkeypatch.setenv("SCHOLAR_RAG_ANTHROPIC_MODEL", "")
    monkeypatch.setenv("SCHOLAR_RAG_DATABASE_PATH", str(tmp_path / "must-not-exist.sqlite3"))
    output = tmp_path / "demo"
    transcript = run_demo(output)
    data = json.loads((output / "screening.json").read_text())
    assert data["synthetic_only"] is True
    assert data["initial"]["counts"]["unscreened"] == 3
    assert data["reviewed"]["counts"]["include"] == 1
    assert data["stale"]["counts"]["stale"] == 3
    assert data["stale"]["included_document_ids"] == []
    assert data["stale_write_status"] == 409
    assert data["agent_event_count"] == 0
    assert data["preview_document_ids"] == data["reviewed"]["included_document_ids"]
    assert transcript == (output / "transcript.txt").read_text()
    assert not (tmp_path / "must-not-exist.sqlite3").exists()
    gif = output / "paper-screening.gif"
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
