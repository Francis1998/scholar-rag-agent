"""Execute the published preview demonstration with credentials present and HTTP denied."""

import os
import sqlite3
import subprocess
import sys
from importlib import import_module
from pathlib import Path

import httpx
import pytest
from PIL import Image

from agent.retrieval_preview import RetrievalPreview
from tests.test_retrieval_preview import denied

KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "MOONSHOT_API_KEY")


def test_preview_demo_and_gif_are_real_offline_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo = import_module("scripts.demo_retrieval_preview")
    renderer = import_module("scripts.create_retrieval_preview_gif")
    for name in KEYS:
        monkeypatch.setenv(name, "synthetic-unused-environment-key")
    monkeypatch.setenv("SCHOLAR_RAG_MAX_HOPS", "invalid-ambient-value")
    monkeypatch.setenv("SCHOLAR_RAG_DATABASE_PATH", str(tmp_path / "must-not-exist.sqlite3"))
    (tmp_path / ".env").write_text(
        "\n".join(f"{name}=synthetic-unused-dotenv-key" for name in KEYS), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", denied)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", denied)
    output = tmp_path / "demo"
    transcript = demo.run_demo(output)
    scoped = RetrievalPreview.model_validate_json((output / "scoped.json").read_bytes())
    baseline = RetrievalPreview.model_validate_json((output / "unscoped.json").read_bytes())
    graph = RetrievalPreview.model_validate_json((output / "graph.json").read_bytes())
    unknown = RetrievalPreview.model_validate_json((output / "unknown.json").read_bytes())
    selected = set(scoped.plan.observation.document_ids or ())
    assert len(selected) == 2
    assert {source.chunk.document_id for source in scoped.sources} == selected
    assert not ({source.chunk.document_id for source in baseline.sources} & selected)
    assert {source.path[-1] for source in graph.sources} == {"multihop", "rrf"}
    assert unknown.plan.observation.document_ids == ("not-ingested",)
    assert unknown.sources == []
    assert unknown.context == ""
    assert "Live/fake generation calls: 0" in transcript
    assert "Persisted agent events: 0" in transcript
    assert "Global top-2 selected papers: 0; scoped top-2: 2" in transcript
    assert "Empty scope: HTTP 422; null scope: HTTP 422" in transcript
    assert "Python / HTTP evidence identical: True" in transcript
    assert "Reopened corpus preview identical: True" in transcript
    assert scoped.context_sha256 in transcript
    assert (output / "transcript.txt").read_text(encoding="utf-8") == transcript
    assert not list(tmp_path.glob("*.sqlite3"))
    assert not list(output.glob("*.sqlite3"))
    for artifact in output.iterdir():
        text = artifact.read_text(encoding="utf-8")
        assert "synthetic-unused" not in text
        if artifact.suffix == ".json":
            assert '"run_id"' not in text
            assert '"generation"' not in text
            assert '"answer"' not in text
    original = (output / "scoped.json").read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        demo.run_demo(output)
    assert (output / "scoped.json").read_bytes() == original
    gif = output / "retrieval-preview.gif"
    renderer.create_gif(output / "transcript.txt", gif)
    with Image.open(gif) as image:
        assert image.n_frames == 4
        assert image.size == (1120, 540)
        assert image.info["loop"] == 0
        frames = []
        for index in range(image.n_frames):
            image.seek(index)
            assert image.info["duration"] == 3500
            frames.append(image.convert("RGB").tobytes())
        assert len(set(frames)) == 4
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        renderer.create_gif(output / "transcript.txt", gif)


@pytest.mark.parametrize("kind", ["unrelated", "oversized"])
def test_preview_renderer_rejects_invalid_transcripts(tmp_path: Path, kind: str) -> None:
    demo = import_module("scripts.demo_retrieval_preview")
    renderer = import_module("scripts.create_retrieval_preview_gif")
    source = tmp_path / "input.txt"
    if kind == "unrelated":
        source.write_text("One\n\nTwo\n\nThree\n\nFour", encoding="utf-8")
        message = "retrieval preview demo"
    else:
        source.write_text(
            "\n\n".join(
                [demo.PANEL_TITLES[0] + "\n" + "\n".join(["line"] * 20), *demo.PANEL_TITLES[1:]]
            ),
            encoding="utf-8",
        )
        message = "does not fit"
    with pytest.raises(ValueError, match=message):
        renderer.create_gif(source, tmp_path / "not-created.gif")
    assert not (tmp_path / "not-created.gif").exists()


@pytest.mark.parametrize("tool", ["demo", "gif", "guide"])
def test_preview_tools_do_not_touch_ambient_storage_or_call_generators(
    tmp_path: Path, tool: str
) -> None:
    ambient = tmp_path / "ambient.sqlite3"
    with sqlite3.connect(ambient) as connection:
        connection.execute("CREATE TABLE sentinel (value TEXT)")
        connection.execute("INSERT INTO sentinel VALUES ('unchanged')")
    before = ambient.read_bytes()
    env = {
        **os.environ,
        "SCHOLAR_RAG_DATABASE_PATH": str(ambient),
        "SCHOLAR_RAG_MAX_HOPS": "invalid-ambient-value",
        **dict.fromkeys(KEYS, "synthetic-unused-key"),
    }
    guard = (
        "import sys, httpx\nfrom pathlib import Path\n"
        "from llm.fake import FakeLLMAdapter\n"
        "from llm.router import RoutingLLMAdapter\n"
        "def deny(*args, **kwargs):\n"
        "    raise AssertionError('No network or generation in this example')\n"
        "httpx.AsyncHTTPTransport.handle_async_request = deny\n"
        "httpx.HTTPTransport.handle_request = deny\n"
        "FakeLLMAdapter.generate = deny\nRoutingLLMAdapter.generate = deny\n"
    )
    if tool == "demo":
        command = (
            "from scripts.demo_retrieval_preview import run_demo\nrun_demo(Path(sys.argv[1]))\n"
        )
        arguments = [str(tmp_path / "output")]
    elif tool == "gif":
        command = (
            "from scripts.demo_retrieval_preview import PANEL_TITLES\n"
            "from scripts.create_retrieval_preview_gif import create_gif\n"
            "source = Path(sys.argv[1])\n"
            "source.write_text('\\n\\n'.join(PANEL_TITLES), encoding='utf-8')\n"
            "create_gif(source, Path(sys.argv[2]))\n"
        )
        arguments = [str(tmp_path / "transcript.txt"), str(tmp_path / "output.gif")]
    else:
        guide = Path(__file__).resolve().parents[1] / "docs/guides/RETRIEVAL_PREVIEW_GUIDE.md"
        command = guide.read_text(encoding="utf-8").split("uv run python - <<'PY'\n", 1)[1]
        command = command.split("\nPY\n```", 1)[0]
        arguments = []
    process = subprocess.run(  # noqa: S603
        [sys.executable, "-c", guard + command, *arguments],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        timeout=30,
    )
    assert process.returncode == 0, process.stderr.decode("utf-8")
    assert ambient.read_bytes() == before
    if tool == "guide":
        assert "Sources: 1" in process.stdout.decode("utf-8")
        assert "Events: 0" in process.stdout.decode("utf-8")
