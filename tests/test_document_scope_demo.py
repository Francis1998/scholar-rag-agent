"""The scope showcase uses real offline API results, not invented research output."""

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
from PIL import Image
from scripts.create_document_scope_gif import create_gif
from scripts.demo_document_scope import PANEL_TITLES, run_demo

from agent.evidence import EvidenceBundle
from tests.test_evidence_export import no_live_call


def test_scope_demo_and_animation_are_measured_offline_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", no_live_call)
    names = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "MOONSHOT_API_KEY")
    for name in names:
        monkeypatch.setenv(name, "synthetic-unused-environment-key")
    monkeypatch.setenv("SCHOLAR_RAG_DEFAULT_MODEL", "anthropic")
    (tmp_path / ".env").write_text(
        "\n".join(f"{name}=synthetic-unused-dotenv-key" for name in names), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    output_dir = tmp_path / "demo"
    transcript = run_demo(output_dir)
    bundle = EvidenceBundle.model_validate_json((output_dir / "scoped.json").read_bytes())
    baseline = EvidenceBundle.model_validate_json((output_dir / "unscoped.json").read_bytes())
    unknown = EvidenceBundle.model_validate_json((output_dir / "unknown.json").read_bytes())
    scope = bundle.plan.observation.document_ids
    assert scope is not None and len(scope) == 2
    assert {source.chunk.document_id for source in bundle.snapshot.sources} == set(scope)
    assert all(source.chunk.document_id not in scope for source in baseline.snapshot.sources)
    assert {c.document_id for c in bundle.answer.citations} == set(scope)
    assert unknown.snapshot.sources == []
    assert unknown.answer.citations == []
    assert unknown.answer.ungrounded
    assert bundle.generation.provider == "fake"
    assert "## Document scope" in (output_dir / "scoped.md").read_text(encoding="utf-8")
    assert "Global top-2 contains selected papers: 0" in transcript
    assert "Scoped top-2 contains selected papers: 2" in transcript
    assert "unscoped chunks=3, scoped chunks=1" in transcript
    assert "Allowed tail via excluded bridge: blocked" in transcript
    assert "Empty list -> HTTP 422; null -> HTTP 422" in transcript
    assert "Reloaded scoped query: 2 selected papers" in transcript
    assert "JSON and Markdown after restart: byte-identical" in transcript
    assert bundle.snapshot.context_sha256 in transcript
    assert (output_dir / "transcript.txt").read_text(encoding="utf-8") == transcript
    for name in ("scoped.json", "scoped.md", "unscoped.json", "unknown.json", "transcript.txt"):
        assert "synthetic-unused" not in (output_dir / name).read_text(encoding="utf-8")
    assert not list(output_dir.glob("*.sqlite3"))
    original = (output_dir / "scoped.json").read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_demo(output_dir)
    assert (output_dir / "scoped.json").read_bytes() == original

    gif = output_dir / "document-scope.gif"
    create_gif(output_dir / "transcript.txt", gif)
    with Image.open(gif) as image:
        assert image.n_frames == 4
        assert image.size == (1120, 540)
        assert image.info["loop"] == 0
        frames = []
        for index in range(4):
            image.seek(index)
            assert image.info["duration"] == 3500
            frames.append(image.convert("RGB").tobytes())
        assert len(set(frames)) == 4


@pytest.mark.parametrize(
    "text",
    [
        "Unrelated transcript.",
        "One\n\nTwo\n\nThree\n\nFour",
        "\n\n".join([PANEL_TITLES[0], "", PANEL_TITLES[2], PANEL_TITLES[3]]),
    ],
)
def test_scope_gif_rejects_unrelated_transcripts(tmp_path: Path, text: str) -> None:
    source = tmp_path / "unrelated.txt"
    source.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="scope demo"):
        create_gif(source, tmp_path / "not-created.gif")
    assert not (tmp_path / "not-created.gif").exists()


def test_scope_gif_refuses_to_clip_oversized_transcripts(tmp_path: Path) -> None:
    source = tmp_path / "oversized.txt"
    source.write_text(
        "\n\n".join([PANEL_TITLES[0] + "\n" + "\n".join(["line"] * 20), *PANEL_TITLES[1:]]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not fit"):
        create_gif(source, tmp_path / "not-created.gif")
    assert not (tmp_path / "not-created.gif").exists()


@pytest.mark.parametrize("tool", ["demo", "gif"])
def test_scope_tools_do_not_initialize_an_ambient_database(tmp_path: Path, tool: str) -> None:
    """A fresh import must not modify a user's unrelated environment-selected DB."""
    ambient = tmp_path / "unrelated.sqlite3"
    with sqlite3.connect(ambient) as connection:
        connection.execute("CREATE TABLE sentinel (value TEXT)")
        connection.execute("INSERT INTO sentinel VALUES ('unchanged')")
    before = ambient.read_bytes()
    env = {
        **os.environ,
        "SCHOLAR_RAG_DATABASE_PATH": str(ambient),
        **dict.fromkeys(
            ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "MOONSHOT_API_KEY"), ""
        ),
    }
    guard = (
        "import sys, httpx\nfrom pathlib import Path\n"
        "def no_network(*args, **kwargs):\n"
        "    raise AssertionError('Synthetic scope tools must not use the network')\n"
        "httpx.AsyncHTTPTransport.handle_async_request = no_network\n"
    )
    if tool == "demo":
        command = "from scripts.demo_document_scope import run_demo\nrun_demo(Path(sys.argv[1]))\n"
        arguments = [str(tmp_path / "output")]
    else:
        transcript = tmp_path / "transcript.txt"
        transcript.write_text("\n\n".join(PANEL_TITLES), encoding="utf-8")
        command = (
            "from scripts.create_document_scope_gif import create_gif\n"
            "create_gif(Path(sys.argv[1]), Path(sys.argv[2]))\n"
        )
        arguments = [str(transcript), str(tmp_path / "output.gif")]
    process = subprocess.run(  # noqa: S603
        [sys.executable, "-c", guard + command, *arguments],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        timeout=30,
    )
    assert process.returncode == 0, process.stderr.decode("utf-8")
    with sqlite3.connect(ambient) as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master")}
    assert tables == {"sentinel"}
    assert ambient.read_bytes() == before
