"""Reproduce actual native preview outcomes and deterministic output illustrations."""

import json
from pathlib import Path

import pytest
from PIL import Image
from scripts.create_retrieval_workbench_gif import create_gif
from scripts.demo_retrieval_workbench import PreviewPage, run_demo


def test_offline_demo_and_media_ignore_ambient_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ambient = tmp_path / "private.sqlite3"
    ambient.write_bytes(b"untouched")
    monkeypatch.setenv("SCHOLAR_RAG_DATABASE_PATH", str(ambient))
    monkeypatch.setenv("SCHOLAR_RAG_MAX_HOPS", "invalid-ambient-value")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-unused-key")
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "measured"
    transcript = run_demo(output)
    checks = json.loads((output / "checks.json").read_bytes())
    assert checks["opening_preview_calls"] == 0
    assert checks["submission_preview_calls"] == 1
    assert checks["prepared_passages"] == 2
    assert checks["minimum_diagnostic"] == {
        "observed_documents": 2,
        "required_documents": 3,
        "passed": False,
    }
    assert checks["agent_events"] == checks["external_connection_calls"] == 0
    assert not any(checks["forbidden_call_counts"].values())
    assert all(value for value in checks.values() if isinstance(value, bool))
    assert (output / "transcript.txt").read_text() == transcript
    assert len(PreviewPage((output / "preview.html").read_text()).passages) == 2
    assert ambient.read_bytes() == b"untouched"
    assert not list(output.glob("*.sqlite3"))
    gif = output / "workbench.gif"
    create_gif(output / "transcript.txt", gif)
    with Image.open(gif) as image:
        assert image.n_frames == 4 and image.size == (1120, 540)
        assert image.info["duration"] == 3500
    with Image.open(gif.with_suffix(".png")) as image:
        assert image.size == (1120, 540)
    repeated = output / "repeat.gif"
    create_gif(output / "transcript.txt", repeated)
    assert repeated.read_bytes() == gif.read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_demo(output)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        create_gif(output / "transcript.txt", gif)


def test_unrelated_transcript_is_not_presented_as_executed_preview(tmp_path: Path) -> None:
    transcript = tmp_path / "wrong.txt"
    transcript.write_text("not measured")
    with pytest.raises(ValueError, match="executed retrieval workbench"):
        create_gif(transcript, tmp_path / "wrong.gif")
