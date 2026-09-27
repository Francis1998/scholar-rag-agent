"""Execute the published minimum-evidence workflow and verify its measured illustration."""

import json
import os
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
from PIL import Image
from scripts import create_minimum_evidence_documents_gif as renderer
from scripts import demo_minimum_evidence_documents as demo

from agent.comparison_models import RunComparison
from agent.evidence import EvidenceBundle, EvidenceSnapshot
from agent.models import AgentRunResult, AgentState
from agent.retrieval_preview import RetrievalPreview
from tests.test_minimum_evidence_documents import denied

ROOT = Path(__file__).resolve().parents[1]
KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "MOONSHOT_API_KEY")


def test_demo_measures_admission_and_regenerates_the_committed_transcript_and_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key in KEYS:
        monkeypatch.setenv(key, "synthetic-unused-ambient-key")
    monkeypatch.setenv("SCHOLAR_RAG_DATABASE_PATH", str(tmp_path / "must-not-exist.sqlite3"))
    monkeypatch.setenv("SCHOLAR_RAG_MAX_SOURCE_DOCS", "invalid-ambient-value")
    (tmp_path / ".env").write_text(
        "\n".join(f"{key}=synthetic-unused-dotenv-key" for key in KEYS), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", denied)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", denied)
    output = tmp_path / "demo"
    transcript = demo.run_demo(output)
    single = RetrievalPreview.model_validate_json((output / "single-preview.json").read_bytes())
    unknown = RetrievalPreview.model_validate_json((output / "unknown-preview.json").read_bytes())
    sufficient = RetrievalPreview.model_validate_json(
        (output / "sufficient-preview.json").read_bytes()
    )
    failed = AgentRunResult.model_validate_json((output / "failed-run.json").read_bytes())
    events = json.loads((output / "failed-events.json").read_bytes())
    snapshot = EvidenceSnapshot.model_validate(events[-2]["payload"])
    bundle = EvidenceBundle.model_validate_json((output / "bundle.json").read_bytes())
    comparison = RunComparison.model_validate_json((output / "comparison.json").read_bytes())
    checks = json.loads((output / "checks.json").read_bytes())
    assert failed.state == AgentState.ERROR and failed.answer is None
    assert len(single.sources) == 3
    assert single.evidence_assessment is not None and not single.evidence_assessment.passed
    assert snapshot.sources == single.sources
    assert snapshot.request.context == single.context
    assert unknown.sources == [] and unknown.plan.observation.document_ids == ("not-ingested",)
    assert sufficient.plan.observation.document_ids == ("paper-a", "paper-b")
    assert sufficient.sources == bundle.snapshot.sources
    assert sufficient.context_sha256 == bundle.snapshot.context_sha256
    assert bundle.generation.provider == "fake"
    assert comparison.any_changes and comparison.changes.evidence_policy_changed
    assert not comparison.changes.context_changed
    assert checks == {
        "single_chunks": 3,
        "single_assessment": {"required_documents": 2, "observed_documents": 1, "passed": False},
        "unknown_assessment": {"required_documents": 1, "observed_documents": 0, "passed": False},
        "failed_state": "ERROR",
        "failed_generations": 0,
        "failed_snapshot_chunks": 3,
        "failed_generation_records": 0,
        "failed_done_events": 0,
        "sufficient_assessment": {"required_documents": 2, "observed_documents": 2, "passed": True},
        "sufficient_chunks": 2,
        "preview_generations": 0,
        "preview_event_writes": 0,
        "preview_matches_query": True,
        "policy_only_change": True,
        "restart_byte_identical": True,
        "query_fake_generations": 2,
        "live_http_attempts": 0,
    }
    assert "min_evidence_documents=2" in (output / "bundle.md").read_text(encoding="utf-8")
    assert (output / "transcript.txt").read_text(encoding="utf-8") == transcript
    assert (ROOT / "docs/assets/minimum-evidence-documents.txt").read_text(
        encoding="utf-8"
    ) == transcript
    assert demo.run_demo(tmp_path / "second") == transcript
    assert not list(tmp_path.glob("*.sqlite3"))
    assert not list(output.glob("*.sqlite3"))
    for artifact in output.iterdir():
        assert "synthetic-unused" not in artifact.read_text(encoding="utf-8")
    destination = output / "minimum.gif"
    renderer.create_gif(output / "transcript.txt", destination)
    with (
        Image.open(destination) as generated,
        Image.open(ROOT / "docs/assets/minimum-evidence-documents.gif") as committed,
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
    renderer.create_gif(output / "transcript.txt", output / "second.gif")
    assert destination.read_bytes() == (output / "second.gif").read_bytes()
    original = destination.read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        renderer.create_gif(output / "transcript.txt", destination)
    assert destination.read_bytes() == original


@pytest.mark.parametrize("name", ["failed-events.json", "bundle.md", "transcript.txt"])
def test_demo_refuses_existing_named_outputs_before_any_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    original = tmp_path / name
    original.write_text("caller-owned", encoding="utf-8")
    monkeypatch.setattr(demo, "create_app", denied)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        demo.run_demo(tmp_path)
    assert original.read_text(encoding="utf-8") == "caller-owned"
    assert list(tmp_path.iterdir()) == [original]


@pytest.mark.parametrize("kind", ["unrelated", "overflow"])
def test_renderer_rejects_unrelated_or_overflowing_transcripts(tmp_path: Path, kind: str) -> None:
    source, destination = tmp_path / "source.txt", tmp_path / "not-created.gif"
    source.write_text(
        "One\n\nTwo\n\nThree\n\nFour"
        if kind == "unrelated"
        else "\n\n".join(
            [demo.PANEL_TITLES[0] + "\n" + "\n".join(["line"] * 20), *demo.PANEL_TITLES[1:]]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"minimum evidence documents demo|does not fit"):
        renderer.create_gif(source, destination)
    assert not destination.exists()


@pytest.mark.parametrize(("index", "label"), [(0, "API"), (1, "Python")])
def test_published_examples_leave_ambient_storage_and_credentials_unused(
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
        "SCHOLAR_RAG_MAX_HOPS": "invalid-ambient-value",
        **dict.fromkeys(KEYS, "synthetic-unused-key"),
    }
    (tmp_path / ".env").write_text(
        "\n".join(f"{key}=synthetic-unused-dotenv-key" for key in KEYS), encoding="utf-8"
    )
    guide = (ROOT / "docs/guides/MINIMUM_EVIDENCE_DOCUMENTS_GUIDE.md").read_text(encoding="utf-8")
    examples = re.findall(r"uv run python - <<'PY'\n(.*?)\nPY\n", guide, re.DOTALL)
    assert len(examples) == 2
    guard = (
        "import httpx\n"
        "def deny(*args, **kwargs):\n"
        "    raise AssertionError('No external HTTP in this offline example')\n"
        "httpx.HTTPTransport.handle_request = deny\n"
        "httpx.AsyncHTTPTransport.handle_async_request = deny\n"
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
    output = process.stdout.decode("utf-8")
    assert f"{label} blocked: ERROR observed: 1" in output
    assert f"{label} admitted: DONE sources: 2" in output
    assert ambient.read_bytes() == original
