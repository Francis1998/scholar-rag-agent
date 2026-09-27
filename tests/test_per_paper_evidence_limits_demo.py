"""Execute the published synthetic demo/examples and render their measured transcript."""

import os
import re
import sqlite3
import subprocess
import sys
from importlib import import_module
from pathlib import Path

import httpx
import pytest
from PIL import Image

from agent.comparison_models import RunComparison
from agent.evidence import EvidenceBundle
from agent.retrieval_preview import RetrievalPreview
from tests.test_per_paper_evidence_limits import denied

KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "MOONSHOT_API_KEY")
ROOT = Path(__file__).resolve().parents[1]


def test_demo_records_real_offline_counts_provenance_and_reproducible_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo = import_module("scripts.demo_per_paper_evidence_limits")
    renderer = import_module("scripts.create_per_paper_evidence_limits_gif")
    for key in KEYS:
        monkeypatch.setenv(key, "synthetic-unused-ambient-key")
    monkeypatch.setenv("SCHOLAR_RAG_MAX_SOURCE_DOCS", "invalid-ambient-value")
    monkeypatch.setenv("SCHOLAR_RAG_DATABASE_PATH", str(tmp_path / "must-not-exist.sqlite3"))
    (tmp_path / ".env").write_text(
        "\n".join(f"{key}=synthetic-unused-dotenv-key" for key in KEYS), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", denied)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", denied)
    output = tmp_path / "demo"
    transcript = demo.run_demo(output)
    baseline = RetrievalPreview.model_validate_json((output / "baseline.json").read_bytes())
    limited = RetrievalPreview.model_validate_json((output / "limited.json").read_bytes())
    collection = RetrievalPreview.model_validate_json((output / "collection.json").read_bytes())
    unknown = RetrievalPreview.model_validate_json((output / "unknown.json").read_bytes())
    bundle = EvidenceBundle.model_validate_json((output / "bundle.json").read_bytes())
    comparison = RunComparison.model_validate_json((output / "comparison.json").read_bytes())
    assert len(baseline.sources) == 6 and len(limited.sources) == 3
    assert len({source.chunk.title for source in baseline.sources}) == 1
    assert {source.chunk.document_id for source in collection.sources} == {"paper-a", "paper-b"}
    assert collection.plan.observation.document_ids == ("paper-a", "paper-b")
    assert unknown.sources == [] and unknown.plan.observation.document_ids == ("not-ingested",)
    assert limited.sources == bundle.snapshot.sources
    assert limited.context_sha256 == bundle.snapshot.context_sha256
    assert bundle.generation.provider == "fake"
    assert bundle.plan.observation.evidence_policy == limited.plan.observation.evidence_policy
    assert comparison.changes.evidence_policy_changed and comparison.any_changes
    assert not comparison.changes.context_changed
    assert "max_chunks_per_document=1" in (output / "bundle.md").read_text(encoding="utf-8")
    for measurement in (
        "Omitted: paper-a=3, paper-b=2, paper-c=1",
        "Cap 1: paper-a=1, paper-b=1, paper-c=1",
        "Hybrid / graph calls: omitted=1/1; capped=1/1",
        "Preview generation calls=0; preview event writes=0",
        "Explicit null -> HTTP 422; boolean -> HTTP 422",
        "Exports and comparison after restart: byte-identical=True",
        "Query fake generations=3; live HTTP attempts=0",
        limited.context_sha256,
    ):
        assert measurement in transcript
    assert (output / "transcript.txt").read_text(encoding="utf-8") == transcript
    assert (ROOT / "docs/assets/per-paper-evidence-limits.txt").read_text(
        encoding="utf-8"
    ) == transcript
    assert not list(tmp_path.glob("*.sqlite3"))
    assert not list(output.glob("*.sqlite3"))
    for artifact in output.iterdir():
        assert "synthetic-unused" not in artifact.read_text(encoding="utf-8")
    assert demo.run_demo(tmp_path / "second-demo") == transcript
    gif = output / "limits.gif"
    renderer.create_gif(output / "transcript.txt", gif)
    with Image.open(gif) as image:
        assert image.n_frames == 4 and image.size == (1120, 540)
        assert image.info["loop"] == 0
        frames = []
        for index in range(image.n_frames):
            image.seek(index)
            assert image.info["duration"] == 3500
            frames.append(image.convert("RGB").tobytes())
        assert len(set(frames)) == 4
    renderer.create_gif(output / "transcript.txt", output / "second.gif")
    assert gif.read_bytes() == (output / "second.gif").read_bytes()
    with Image.open(ROOT / "docs/assets/per-paper-evidence-limits.gif") as committed:
        assert committed.n_frames == 4 and committed.size == (1120, 540)
    before = gif.read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        renderer.create_gif(output / "transcript.txt", gif)
    assert gif.read_bytes() == before


@pytest.mark.parametrize("name", ["baseline.json", "bundle.md", "transcript.txt"])
def test_demo_refuses_caller_outputs_before_creating_an_app(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    demo = import_module("scripts.demo_per_paper_evidence_limits")
    original = tmp_path / name
    original.write_text("caller-owned", encoding="utf-8")
    monkeypatch.setattr(demo, "create_app", denied)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        demo.run_demo(tmp_path)
    assert original.read_text(encoding="utf-8") == "caller-owned"
    assert list(tmp_path.iterdir()) == [original]


@pytest.mark.parametrize("kind", ["unrelated", "overflow"])
def test_renderer_rejects_unrelated_or_overflowing_transcripts(tmp_path: Path, kind: str) -> None:
    demo = import_module("scripts.demo_per_paper_evidence_limits")
    renderer = import_module("scripts.create_per_paper_evidence_limits_gif")
    source, destination = tmp_path / "source.txt", tmp_path / "not-created.gif"
    text = (
        "One\n\nTwo\n\nThree\n\nFour"
        if kind == "unrelated"
        else "\n\n".join(
            [demo.PANEL_TITLES[0] + "\n" + "\n".join(["line"] * 20), *demo.PANEL_TITLES[1:]]
        )
    )
    source.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=r"per-paper evidence limits demo|does not fit"):
        renderer.create_gif(source, destination)
    assert not destination.exists()


@pytest.mark.parametrize(("example", "expected"), [(0, "API sources: 2"), (1, "Python sources: 2")])
def test_published_examples_ignore_ambient_storage_and_keys(
    tmp_path: Path, example: int, expected: str
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
    (tmp_path / ".env").write_text(
        "\n".join(f"{key}=synthetic-unused-dotenv-key" for key in KEYS), encoding="utf-8"
    )
    guide = (ROOT / "docs/guides/PER_PAPER_EVIDENCE_LIMITS_GUIDE.md").read_text(encoding="utf-8")
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
        [sys.executable, "-c", guard + examples[example]],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert process.returncode == 0, process.stderr.decode("utf-8")
    assert expected in process.stdout.decode("utf-8")
    assert ambient.read_bytes() == before
