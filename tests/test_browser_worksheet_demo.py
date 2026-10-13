"""Reproduce the integrated offline form workflow and its honestly labeled media."""

import json
from pathlib import Path

import pytest
from PIL import Image
from scripts.create_browser_worksheet_gif import create_gif
from scripts.demo_browser_worksheet import WorksheetPage, run_demo

from agent.research_worksheet import ResearchWorksheet


def test_executed_native_form_demo_and_reproducible_illustration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ambient = tmp_path / "private.sqlite3"
    ambient.write_bytes(b"private sentinel")
    monkeypatch.setenv("SCHOLAR_RAG_DATABASE_PATH", str(ambient))
    monkeypatch.setenv("SCHOLAR_RAG_MAX_HOPS", "invalid-ambient-value")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-unused-key")
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "demo"
    transcript = run_demo(output)
    checks = json.loads((output / "checks.json").read_bytes())
    assert checks["cells"] == 4
    assert checks["questions"] == checks["papers"] == 2
    assert checks["agent_events"] == checks["external_connection_calls"] == 0
    assert not any(checks["forbidden_call_counts"].values())
    assert all(value for value in checks.values() if isinstance(value, bool))
    assert checks["empty_selection_status"] == 422
    worksheet = ResearchWorksheet.model_validate_json((output / "worksheet.json").read_bytes())
    assert len(worksheet.rows) == 2
    assert worksheet.to_markdown() == (output / "worksheet.md").read_text()
    html = (output / "worksheet.html").read_text()
    assert WorksheetPage(html).form_action == "/explore/worksheet"
    assert "EXCLUDED_MARKER" not in html
    assert (output / "transcript.txt").read_text() == transcript
    assert ambient.read_bytes() == b"private sentinel"
    assert not list(output.glob("*.sqlite3"))
    for name, count in checks["output_bytes"].items():
        assert (output / name).stat().st_size == count
    gif = output / "browser-worksheet.gif"
    create_gif(output / "transcript.txt", gif)
    with Image.open(gif) as image:
        assert image.n_frames == 4 and image.size == (1120, 540)
        assert image.info["duration"] == 3500
    with Image.open(gif.with_suffix(".png")) as preview:
        assert preview.size == (1120, 540)
    replay = output / "replayed.gif"
    create_gif(output / "transcript.txt", replay)
    assert gif.read_bytes() == replay.read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_demo(output)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        create_gif(output / "transcript.txt", gif)


def test_illustration_rejects_unrelated_transcript(tmp_path: Path) -> None:
    transcript = tmp_path / "wrong.txt"
    transcript.write_text("Not the executed worksheet demo")
    with pytest.raises(ValueError, match="executed browser worksheet"):
        create_gif(transcript, tmp_path / "wrong.gif")
