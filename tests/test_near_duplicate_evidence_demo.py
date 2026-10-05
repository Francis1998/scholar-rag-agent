"""Reproduce the published measurements, offline examples, and all committed GIF frames."""

import json
import os
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image
from scripts import create_near_duplicate_evidence_gif as renderer
from scripts import demo_near_duplicate_evidence as demo

from agent.evidence import EvidenceSnapshot
from agent.models import AgentRunResult, AgentState
from agent.retrieval_preview import RetrievalPreview
from llm.fake import FakeLLMAdapter
from tests.test_near_duplicate_evidence import denied

ROOT = Path(__file__).resolve().parents[1]
KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "MOONSHOT_API_KEY")


def test_demo_reproduces_measured_transcript_and_frames_without_models_or_ambient_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key in KEYS:
        monkeypatch.setenv(key, "synthetic-unused-ambient")
    monkeypatch.setenv("SCHOLAR_RAG_DATABASE_PATH", str(tmp_path / "must-not-exist.sqlite3"))
    monkeypatch.setenv("SCHOLAR_RAG_MAX_SOURCE_DOCS", "invalid-ambient")
    (tmp_path / ".env").write_text(
        "\n".join(f"{key}=synthetic-unused-dotenv" for key in KEYS), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(FakeLLMAdapter, "generate", denied)
    output = tmp_path / "demo"
    transcript = demo.run_demo(output)
    baseline = RetrievalPreview.model_validate_json((output / "baseline-preview.json").read_bytes())
    collapsed = RetrievalPreview.model_validate_json(
        (output / "collapsed-preview.json").read_bytes()
    )
    capped = RetrievalPreview.model_validate_json((output / "capped-preview.json").read_bytes())
    failed = AgentRunResult.model_validate_json((output / "failed-run.json").read_bytes())
    events = json.loads((output / "failed-events.json").read_bytes())
    snapshot = EvidenceSnapshot.model_validate(events[-2]["payload"])
    checks = json.loads((output / "checks.json").read_bytes())
    assert failed.state == AgentState.ERROR and failed.answer is None
    assert len(baseline.sources) == 5
    assert len(collapsed.sources) == 3
    assert len(capped.sources) == 2
    assert snapshot.sources == collapsed.sources
    assert snapshot.request.context == collapsed.context
    assert events[-1]["payload"]["payload"]["code"] == "insufficient_evidence_documents"
    assert checks == {
        "baseline_chunks": 5,
        "baseline_documents": 4,
        "baseline_context_bytes": 435,
        "collapsed_chunks": 3,
        "collapsed_documents": 2,
        "collapsed_context_bytes": 264,
        "term_set_threshold_chunks": 4,
        "capped_chunks": 2,
        "capped_documents": 2,
        "surviving_chunk_ids": ["a-distinct", "d-distinct", "a-original"],
        "removed_chunk_ids": ["b-copy", "c-variant"],
        "exact_survivors": True,
        "corpus_unchanged": True,
        "failed_state": "ERROR",
        "failed_assessment": {"required_documents": 4, "observed_documents": 2, "passed": False},
        "failed_snapshot_matches_preview": True,
        "generation_records": 0,
        "model_calls": 0,
        "external_http_attempts": 0,
        "preview_event_writes": 0,
    }
    assert (output / "transcript.txt").read_text(encoding="utf-8") == transcript
    assert (ROOT / "docs/assets/near-duplicate-evidence.txt").read_text(
        encoding="utf-8"
    ) == transcript
    assert demo.run_demo(tmp_path / "second") == transcript
    assert not list(tmp_path.glob("*.sqlite3"))
    assert not list(output.glob("*.sqlite3"))
    for artifact in output.iterdir():
        assert "synthetic-unused" not in artifact.read_text(encoding="utf-8")

    destination = output / "collapse.gif"
    renderer.create_gif(output / "transcript.txt", destination)
    with (
        Image.open(destination) as generated,
        Image.open(ROOT / "docs/assets/near-duplicate-evidence.gif") as committed,
    ):
        assert generated.n_frames == committed.n_frames == 4
        assert generated.size == committed.size == (1120, 540)
        assert generated.info["loop"] == committed.info["loop"] == 0
        frames = []
        for index in range(generated.n_frames):
            generated.seek(index)
            committed.seek(index)
            assert generated.info["duration"] == committed.info["duration"] == 3500
            frame = generated.convert("RGB").tobytes()
            assert frame == committed.convert("RGB").tobytes()
            frames.append(frame)
        assert len(set(frames)) == 4
    original = destination.read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        renderer.create_gif(output / "transcript.txt", destination)
    assert destination.read_bytes() == original


@pytest.mark.parametrize("name", ["failed-events.json", "checks.json", "transcript.txt"])
def test_demo_refuses_existing_outputs_before_creating_a_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    existing = tmp_path / name
    existing.write_text("caller-owned", encoding="utf-8")
    monkeypatch.setattr(demo, "create_app", denied)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        demo.run_demo(tmp_path)
    assert existing.read_text(encoding="utf-8") == "caller-owned"
    assert list(tmp_path.iterdir()) == [existing]


@pytest.mark.parametrize("kind", ["unrelated", "overflow"])
def test_renderer_rejects_wrong_or_overflowing_panels(tmp_path: Path, kind: str) -> None:
    source, destination = tmp_path / "input.txt", tmp_path / "not-created.gif"
    source.write_text(
        "One\n\nTwo\n\nThree\n\nFour"
        if kind == "unrelated"
        else "\n\n".join(
            [demo.PANEL_TITLES[0] + "\n" + "\n".join(["line"] * 20), *demo.PANEL_TITLES[1:]]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"near-duplicate evidence demo|does not fit"):
        renderer.create_gif(source, destination)
    assert not destination.exists()


@pytest.mark.parametrize(("index", "label"), [(0, "API"), (1, "Python")])
def test_complete_guide_examples_run_without_touching_ambient_state_or_generating(
    tmp_path: Path, index: int, label: str
) -> None:
    ambient = tmp_path / "ambient.sqlite3"
    with sqlite3.connect(ambient) as connection:
        connection.execute("CREATE TABLE sentinel (value TEXT)")
        connection.execute("INSERT INTO sentinel VALUES ('unchanged')")
    original = ambient.read_bytes()
    environment = {
        **os.environ,
        "SCHOLAR_RAG_DATABASE_PATH": str(ambient),
        "SCHOLAR_RAG_MAX_HOPS": "invalid-ambient",
        **dict.fromkeys(KEYS, "synthetic-unused-key"),
    }
    guide = (ROOT / "docs/guides/NEAR_DUPLICATE_COLLAPSE_GUIDE.md").read_text(encoding="utf-8")
    examples = re.findall(r"uv run python - <<'PY'\n(.*?)\nPY\n", guide, re.DOTALL)
    assert len(examples) == 2
    guard = (
        "import httpx\n"
        "from llm.fake import FakeLLMAdapter\n"
        "from llm.router import RoutingLLMAdapter\n"
        "def deny(*args, **kwargs):\n"
        "    raise AssertionError('No HTTP or models in this offline example')\n"
        "httpx.HTTPTransport.handle_request = deny\n"
        "httpx.AsyncHTTPTransport.handle_async_request = deny\n"
        "FakeLLMAdapter.generate = deny\n"
        "RoutingLLMAdapter.generate = deny\n"
    )
    process = subprocess.run(  # noqa: S603
        [sys.executable, "-c", guard + examples[index]],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert process.returncode == 0, process.stderr.decode("utf-8")
    assert f"{label}:" in process.stdout.decode("utf-8")
    assert "ERROR without generation" in process.stdout.decode("utf-8")
    assert ambient.read_bytes() == original
