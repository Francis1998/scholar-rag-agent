"""Reproduce measured worksheet artifacts under hostile ambient configuration."""

import json
import os
import subprocess
import sys
from importlib import import_module
from pathlib import Path

import httpx
import pytest
from PIL import Image

from agent.research_worksheet import ResearchWorksheet
from tests.test_research_worksheet import denied

KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "MOONSHOT_API_KEY")


def test_offline_demo_exports_measured_worksheet_and_reproducible_gif(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo = import_module("scripts.demo_research_worksheet")
    renderer = import_module("scripts.create_research_worksheet_gif")
    for key in KEYS:
        monkeypatch.setenv(key, "synthetic-unused-environment-key")
    ambient = tmp_path / "must-not-exist.sqlite3"
    monkeypatch.setenv("SCHOLAR_RAG_DATABASE_PATH", str(ambient))
    monkeypatch.setenv("SCHOLAR_RAG_MAX_HOPS", "not-a-valid-integer")
    (tmp_path / ".env").write_text(
        "\n".join(f"{key}=synthetic-unused-dotenv-key" for key in KEYS), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", denied)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", denied)
    output = tmp_path / "demo"
    transcript = demo.run_demo(output)
    worksheet = ResearchWorksheet.model_validate_json((output / "worksheet.json").read_bytes())
    checks = json.loads((output / "checks.json").read_bytes())
    assert len(worksheet.documents) == 2
    assert len(worksheet.rows) == 2
    assert sum(len(row.cells) for row in worksheet.rows) == 4
    assert all(
        cell.status == "passages_returned" and len(cell.passages) == 1
        for row in worksheet.rows
        for cell in row.cells
    )
    assert checks["global_preview_papers"] == 1
    assert checks["worksheet_papers"] == 2
    assert checks["generation_calls"] == checks["agent_events"] == 0
    assert checks["python_http_equal"] is True
    assert checks["restart_equal"] is True
    assert checks["database_unchanged"] is True
    assert checks["temporary_database_removed"] is True
    assert checks["unknown_selection_status"] == checks["unscoped_status"] == 422
    assert checks["json_bytes"] == (output / "worksheet.json").stat().st_size
    assert checks["markdown_bytes"] == (output / "worksheet.md").stat().st_size
    assert checks["json_bytes"] <= 262144 and checks["markdown_bytes"] <= 262144
    assert "Questions: 2; selected papers: 2; cells: 4" in transcript
    assert "Global top-1 represents 1/2 selected papers; worksheet represents 2/2." in transcript
    assert "Live/fake generation calls: 0; agent events: 0" in transcript
    assert "EXCLUDED_MARKER" not in worksheet.to_json()
    assert worksheet.to_markdown() == (output / "worksheet.md").read_text(encoding="utf-8")
    assert transcript == (output / "transcript.txt").read_text(encoding="utf-8")
    assert not ambient.exists() and not list(output.glob("*.sqlite3"))
    for artifact in output.iterdir():
        assert "synthetic-unused" not in artifact.read_text(encoding="utf-8")
    before = (output / "worksheet.json").read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        demo.run_demo(output)
    assert (output / "worksheet.json").read_bytes() == before
    gif = output / "worksheet.gif"
    renderer.create_gif(output / "transcript.txt", gif)
    frames = []
    with Image.open(gif) as image:
        assert image.n_frames == 4 and image.size == (1120, 540)
        assert image.info["loop"] == 0
        for index in range(image.n_frames):
            image.seek(index)
            assert image.info["duration"] == 3500
            frames.append(image.convert("RGB").tobytes())
    assert len(set(frames)) == 4
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        renderer.create_gif(output / "transcript.txt", gif)
    replay = output / "reproduced.gif"
    renderer.create_gif(output / "transcript.txt", replay)
    assert replay.read_bytes() == gif.read_bytes()


@pytest.mark.parametrize("kind", ["unrelated", "oversized"])
def test_renderer_rejects_unrelated_or_oversized_transcripts(tmp_path: Path, kind: str) -> None:
    demo = import_module("scripts.demo_research_worksheet")
    renderer = import_module("scripts.create_research_worksheet_gif")
    source = tmp_path / "transcript.txt"
    panels = ["One", "Two", "Three", "Four"]
    if kind == "oversized":
        panels = [title + "\n" + "\n".join(["line"] * 20) for title in demo.PANEL_TITLES]
    source.write_text("\n\n".join(panels), encoding="utf-8")
    target = tmp_path / "not-created.gif"
    with pytest.raises(ValueError, match=r"worksheet demo|does not fit"):
        renderer.create_gif(source, target)
    assert not target.exists()


def test_published_python_example_is_offline_and_does_not_touch_ambient_storage(
    tmp_path: Path,
) -> None:
    guide = Path(__file__).resolve().parents[1] / "docs/guides/RESEARCH_WORKSHEET_GUIDE.md"
    snippet = guide.read_text(encoding="utf-8").split("uv run python - <<'PY'\n", 1)[1]
    snippet = snippet.split("\nPY\n```", 1)[0]
    ambient = tmp_path / "ambient.sqlite3"
    ambient.write_bytes(b"private sentinel, not an application database")
    env = {
        **os.environ,
        "SCHOLAR_RAG_DATABASE_PATH": str(ambient),
        "SCHOLAR_RAG_MAX_HOPS": "invalid-ambient-value",
        **dict.fromkeys(KEYS, "synthetic-unused-key"),
    }
    guard = (
        "import httpx\nfrom llm.fake import FakeLLMAdapter\n"
        "from llm.router import RoutingLLMAdapter\n"
        "def deny(*args, **kwargs):\n"
        "    raise AssertionError('No HTTP or generation in worksheet example')\n"
        "httpx.HTTPTransport.handle_request = deny\n"
        "httpx.AsyncHTTPTransport.handle_async_request = deny\n"
        "FakeLLMAdapter.generate = deny\nRoutingLLMAdapter.generate = deny\n"
    )
    process = subprocess.run(  # noqa: S603
        [sys.executable, "-c", guard + snippet],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        timeout=30,
    )
    assert process.returncode == 0, process.stderr.decode("utf-8")
    assert "Cells: 4" in process.stdout.decode("utf-8")
    assert "Events: 0" in process.stdout.decode("utf-8")
    assert ambient.read_bytes() == b"private sentinel, not an application database"
    result = ResearchWorksheet.model_validate_json((tmp_path / "worksheet.json").read_bytes())
    assert result.to_markdown() == (tmp_path / "worksheet.md").read_text(encoding="utf-8")
