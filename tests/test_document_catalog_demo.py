"""The document-catalog showcase measures real local API behavior."""

import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
from PIL import Image
from scripts.create_document_catalog_gif import create_gif
from scripts.demo_document_catalog import PANEL_TITLES, run_demo

from agent.evidence import EvidenceBundle
from storage.document_catalog import DocumentCatalogPage
from tests.test_document_catalog import _no_work


def test_catalog_demo_is_offline_and_generates_measured_animation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _no_work)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", _no_work)
    keys = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "MOONSHOT_API_KEY")
    for key in keys:
        monkeypatch.setenv(key, "unused-synthetic-secret")
    monkeypatch.setenv("SCHOLAR_RAG_DEFAULT_MODEL", "anthropic")
    monkeypatch.setenv("SCHOLAR_RAG_MAX_HOPS", "invalid-ambient-value")
    ambient = tmp_path / "must-not-exist.sqlite3"
    monkeypatch.setenv("SCHOLAR_RAG_DATABASE_PATH", str(ambient))
    (tmp_path / ".env").write_text(
        "\n".join(f"{key}=unused-synthetic-secret" for key in keys), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "demo"
    transcript = run_demo(output)
    first = DocumentCatalogPage.model_validate_json((output / "page-1.json").read_bytes())
    second = DocumentCatalogPage.model_validate_json((output / "page-2.json").read_bytes())
    selected = DocumentCatalogPage.model_validate_json((output / "selected.json").read_bytes())
    bundle = EvidenceBundle.model_validate_json((output / "scoped-evidence.json").read_bytes())
    assert len(first.documents) == 2 and len(second.documents) == 1
    assert first.next_cursor and second.next_cursor is None
    assert len(selected.documents) == 1
    assert {s.chunk.document_id for s in bundle.snapshot.sources} == {
        selected.documents[0].document_id
    }
    assert bundle.generation.provider == "fake"
    assert "Agent events created by browsing: 0" in transcript
    assert "First page after reopening SQLite: byte-identical" in transcript
    assert (output / "transcript.txt").read_text(encoding="utf-8") == transcript
    assert not ambient.exists()
    assert not list(output.glob("*.sqlite3"))
    for path in output.iterdir():
        assert "unused-synthetic-secret" not in path.read_text(encoding="utf-8")
    before = (output / "scoped-evidence.json").read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_demo(output)
    assert (output / "scoped-evidence.json").read_bytes() == before

    gif = output / "catalog.gif"
    create_gif(output / "transcript.txt", gif)
    with Image.open(gif) as image:
        assert image.size == (1120, 540)
        assert image.n_frames == 4
        assert image.info["loop"] == 0
        frames = []
        for index in range(4):
            image.seek(index)
            assert image.info["duration"] == 3500
            frames.append(image.convert("RGB").tobytes())
        assert len(set(frames)) == 4


@pytest.mark.parametrize("text", ["unrelated", "One\n\nTwo\n\nThree\n\nFour"])
def test_catalog_gif_rejects_unrelated_transcripts(tmp_path: Path, text: str) -> None:
    transcript = tmp_path / "transcript.txt"
    transcript.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="document catalog demo"):
        create_gif(transcript, tmp_path / "not-created.gif")
    assert not (tmp_path / "not-created.gif").exists()


def test_catalog_gif_refuses_clipped_or_overwritten_artifacts(tmp_path: Path) -> None:
    transcript = tmp_path / "transcript.txt"
    transcript.write_text(
        "\n\n".join([PANEL_TITLES[0] + "\n" + "\n".join(["line"] * 20), *PANEL_TITLES[1:]]),
        encoding="utf-8",
    )
    output = tmp_path / "catalog.gif"
    with pytest.raises(ValueError, match="does not fit"):
        create_gif(transcript, output)
    assert not output.exists()
    transcript.write_text("\n\n".join(PANEL_TITLES), encoding="utf-8")
    create_gif(transcript, output)
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        create_gif(transcript, output)
    assert output.read_bytes() == original


def test_catalog_guide_python_example_runs_without_ambient_settings(tmp_path: Path) -> None:
    guide = Path(__file__).resolve().parents[1] / "docs/guides/DOCUMENT_CATALOG_GUIDE.md"
    snippet = guide.read_text(encoding="utf-8").split("uv run python - <<'PY'\n", 1)[1]
    snippet = snippet.split("\nPY\n```", 1)[0]
    ambient = tmp_path / "must-not-exist.sqlite3"
    env = {
        **os.environ,
        "SCHOLAR_RAG_DATABASE_PATH": str(ambient),
        "SCHOLAR_RAG_MAX_HOPS": "invalid-ambient-value",
        **dict.fromkeys(
            ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "MOONSHOT_API_KEY"),
            "unused-synthetic-secret",
        ),
    }
    guard = (
        "import httpx\n"
        "def no_network(*args, **kwargs):\n"
        "    raise AssertionError('The catalog example must remain offline')\n"
        "httpx.AsyncHTTPTransport.handle_async_request = no_network\n"
        "httpx.HTTPTransport.handle_request = no_network\n"
    )
    process = subprocess.run(
        [sys.executable, "-"],
        input=guard + snippet,
        cwd=tmp_path,
        env=env,
        check=False,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert process.returncode == 0, process.stderr
    assert "Stored chunks: 1" in process.stdout
    assert "State: DONE" in process.stdout
    assert not ambient.exists()
