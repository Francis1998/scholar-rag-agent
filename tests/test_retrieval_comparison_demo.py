"""Reproduce actual policy-comparison API output and its clearly labeled illustration."""

import json
import os
import subprocess
import sys
from importlib import import_module
from pathlib import Path

import httpx
import pytest
from PIL import Image

from agent.retrieval_comparison import RetrievalComparison
from tests.test_retrieval_comparison import denied

KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "MOONSHOT_API_KEY")


def test_offline_demo_measures_new_api_and_renders_reproducible_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo = import_module("scripts.demo_retrieval_comparison")
    renderer = import_module("scripts.create_retrieval_comparison_gif")
    ambient = tmp_path / "must-not-exist.sqlite3"
    monkeypatch.setenv("SCHOLAR_RAG_DATABASE_PATH", str(ambient))
    monkeypatch.setenv("SCHOLAR_RAG_MAX_HOPS", "invalid-ambient-value")
    for key in KEYS:
        monkeypatch.setenv(key, "synthetic-unused-environment-key")
    (tmp_path / ".env").write_text(
        "\n".join(f"{key}=synthetic-unused-dotenv-key" for key in KEYS), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", denied)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", denied)
    output = tmp_path / "demo"
    transcript = demo.run_demo(output)
    result = RetrievalComparison.model_validate_json((output / "comparison.json").read_bytes())
    checks = json.loads((output / "checks.json").read_bytes())
    assert result.baseline.source_count == 5
    assert result.candidate.source_count == 2
    assert result.baseline.document_count == 4
    assert result.candidate.document_count == 2
    assert result.evidence.statistics.removed_count == 3
    assert result.candidate.preview.evidence_assessment.passed is False
    assert checks["comparison_preview_calls"] == 2
    assert checks["model_calls"] == checks["external_http_attempts"] == checks["event_writes"] == 0
    assert checks["agent_events"] == 0
    assert checks["database_unchanged"] is True
    assert checks["python_http_equal"] is True
    assert checks["restart_equal"] is True
    assert checks["temporary_database_removed"] is True
    assert checks["baseline_context_bytes"] == result.baseline.context_utf8_bytes
    assert checks["candidate_context_bytes"] == result.candidate.context_utf8_bytes
    assert checks["context_bytes_delta"] == result.context_bytes_delta
    assert checks["baseline_context_sha256"] == result.baseline.preview.context_sha256
    assert checks["candidate_context_sha256"] == result.candidate.preview.context_sha256
    assert checks["json_bytes"] == (output / "comparison.json").stat().st_size <= 262144
    assert checks["markdown_bytes"] == (output / "comparison.md").stat().st_size <= 262144
    assert result.to_json() == (output / "comparison.json").read_text(encoding="utf-8")
    assert result.to_markdown() == (output / "comparison.md").read_text(encoding="utf-8")
    assert transcript == (output / "transcript.txt").read_text(encoding="utf-8")
    assert "POST /research/compare-retrieval -> HTTP 200" in transcript
    assert "Chunks: 5 -> 2; documents: 4 -> 2" in transcript
    assert "Shared: 2; added: 0; removed: 3" in transcript
    assert "Model calls: 0; external HTTP attempts: 0; event writes: 0" in transcript
    assert not ambient.exists() and not list(output.glob("*.sqlite3"))
    for artifact in output.iterdir():
        assert "synthetic-unused" not in artifact.read_text(encoding="utf-8")
    saved = (output / "comparison.json").read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        demo.run_demo(output)
    assert (output / "comparison.json").read_bytes() == saved
    target = output / "comparison.gif"
    renderer.create_gif(output / "transcript.txt", target)
    frames = []
    with Image.open(target) as image:
        assert image.n_frames == 4
        assert image.size == (1120, 540)
        assert image.info["loop"] == 0
        for index in range(image.n_frames):
            image.seek(index)
            assert image.info["duration"] == 3500
            frames.append(image.convert("RGB").tobytes())
    assert len(set(frames)) == 4
    assert target.stat().st_size < 200000
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        renderer.create_gif(output / "transcript.txt", target)
    replay = output / "reproduced.gif"
    renderer.create_gif(output / "transcript.txt", replay)
    assert replay.read_bytes() == target.read_bytes()


@pytest.mark.parametrize("kind", ["unrelated", "oversized"])
def test_renderer_rejects_unrelated_or_oversized_transcript(tmp_path: Path, kind: str) -> None:
    demo = import_module("scripts.demo_retrieval_comparison")
    renderer = import_module("scripts.create_retrieval_comparison_gif")
    source = tmp_path / "transcript.txt"
    panels = ["One", "Two", "Three", "Four"]
    if kind == "oversized":
        panels = [title + "\n" + "\n".join(["line"] * 20) for title in demo.PANEL_TITLES]
    source.write_text("\n\n".join(panels), encoding="utf-8")
    target = tmp_path / "not-created.gif"
    with pytest.raises(ValueError, match=r"comparison demo|does not fit"):
        renderer.create_gif(source, target)
    assert not target.exists()


def test_demo_refuses_dangling_output_symlinks(tmp_path: Path) -> None:
    demo = import_module("scripts.demo_retrieval_comparison")
    (tmp_path / "comparison.md").symlink_to(tmp_path / "absent.md")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        demo.run_demo(tmp_path)
    assert not (tmp_path / "comparison.json").exists()
    assert not (tmp_path / "absent.md").exists()


def test_published_python_example_stays_offline_and_uses_new_service(tmp_path: Path) -> None:
    guide = Path(__file__).resolve().parents[1] / "docs/guides/RETRIEVAL_COMPARISON_GUIDE.md"
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
        "    raise AssertionError('No HTTP or generation in comparison example')\n"
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
    assert "Chunks: 2 -> 1" in process.stdout.decode("utf-8")
    assert "Minimum passed: False" in process.stdout.decode("utf-8")
    assert "Events: 0" in process.stdout.decode("utf-8")
    assert ambient.read_bytes() == b"private sentinel, not an application database"
    result = RetrievalComparison.model_validate_json((tmp_path / "comparison.json").read_bytes())
    assert result.to_markdown() == (tmp_path / "comparison.md").read_text(encoding="utf-8")
