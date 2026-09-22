"""Measured showcase, environment isolation, temporary cleanup, and runnable guide."""

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import NoReturn

import pytest
from PIL import Image
from scripts import demo_paper_collections as demo
from scripts.create_paper_collections_gif import create_gif

from agent.evidence import EvidenceBundle
from agent.retrieval_preview import RetrievalPreview
from llm.providers import AnthropicAdapter, GeminiAdapter, KimiAdapter, OpenAIAdapter
from storage.paper_collections import CollectionPage, PaperCollection, SQLitePaperCollections

KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "MOONSHOT_API_KEY")


def denied(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("Synthetic collection examples must not call live providers.")


def tracked_temporary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[Path]:
    paths: list[Path] = []

    def create(*, prefix: str) -> TemporaryDirectory[str]:
        directory = TemporaryDirectory(prefix=prefix, dir=tmp_path)
        paths.append(Path(directory.name))
        return directory

    monkeypatch.setattr(demo, "TemporaryDirectory", create)
    return paths


def test_demo_measures_real_results_and_renders_original_animation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ambient = tmp_path / "ambient.sqlite3"
    with sqlite3.connect(ambient) as connection:
        connection.execute("CREATE TABLE sentinel (value TEXT)")
        connection.execute("INSERT INTO sentinel VALUES ('private-ambient-data')")
    before = ambient.read_bytes()
    for key in KEYS:
        monkeypatch.setenv(key, "synthetic-unused-secret")
    monkeypatch.setenv("SCHOLAR_RAG_DATABASE_PATH", str(ambient))
    monkeypatch.setenv("SCHOLAR_RAG_MAX_HOPS", "invalid-environment-setting")
    monkeypatch.setenv("SCHOLAR_RAG_DEFAULT_MODEL", "anthropic")
    (tmp_path / ".env").write_text(
        "\n".join(f"{key}=synthetic-unused-dotenv-secret" for key in KEYS)
        + "\nSCHOLAR_RAG_MAX_SOURCE_DOCS=invalid-dotenv-setting\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    for adapter in (OpenAIAdapter, AnthropicAdapter, GeminiAdapter, KimiAdapter):
        monkeypatch.setattr(adapter, "generate", denied)
    temporary = tracked_temporary(monkeypatch, tmp_path)
    output = tmp_path / "demo"
    transcript = demo.run_demo(output)
    checks = json.loads((output / "checks.json").read_bytes())
    assert checks["crud_preview_generation_calls"] == checks["crud_preview_events"] == 0
    assert checks["total_fake_generation_calls"] == 1
    assert checks["collection_restart_identical"] and checks["preview_restart_identical"]
    assert checks["stale_update_status"] == 409
    assert checks["deleted_query_status"] == 404
    assert checks["remaining_documents"] == 3 and checks["remaining_runs"] == 1
    assert checks["post_query_event_delta"] == 0
    assert checks["json_export_unchanged"] and checks["markdown_export_unchanged"]
    assert checks["temporary_database_removed"]
    assert len(temporary) == 1 and all(not path.exists() for path in temporary)
    assert ambient.read_bytes() == before
    assert not list(output.glob("*.sqlite3"))
    collection = PaperCollection.model_validate_json((output / "collection.json").read_bytes())
    replacement = PaperCollection.model_validate_json((output / "replacement.json").read_bytes())
    assert replacement.collection_id == collection.collection_id
    assert (collection.revision, replacement.revision) == (1, 2)
    assert list(collection.document_ids) == checks["original_document_ids"]
    assert list(replacement.document_ids) == checks["updated_document_ids"]
    first = CollectionPage.model_validate_json((output / "page-1.json").read_bytes())
    second = CollectionPage.model_validate_json((output / "page-2.json").read_bytes())
    assert len(first.collections) == len(second.collections) == 1
    assert first.next_cursor and second.next_cursor is None
    original = RetrievalPreview.model_validate_json((output / "preview-original.json").read_bytes())
    updated = RetrievalPreview.model_validate_json((output / "preview-updated.json").read_bytes())
    bundle = EvidenceBundle.model_validate_json((output / "evidence.json").read_bytes())
    assert len(original.sources) == checks["original_source_count"] == 2
    assert len(updated.sources) == checks["updated_source_count"] == 1
    assert {s.chunk.document_id for s in original.sources} == set(collection.document_ids)
    assert {s.chunk.document_id for s in updated.sources} == set(replacement.document_ids)
    assert bundle.snapshot.sources == original.sources
    assert bundle.snapshot.request.context == original.context
    assert bundle.generation.provider == "fake"
    assert "EXCLUDED_MARKER" not in original.context
    assert (output / "transcript.txt").read_text(encoding="utf-8") == transcript
    assert "Stale revision rejected: HTTP 409" in transcript
    assert "Temporary SQLite removed: True" in transcript
    for path in output.iterdir():
        text = path.read_text(encoding="utf-8")
        assert "synthetic-unused" not in text and "private-ambient-data" not in text
    assert {path.name for path in output.iterdir()} == set(demo.ARTIFACT_NAMES)
    assert demo.run_demo(tmp_path / "second") == transcript
    assert all(not path.exists() for path in temporary)
    previous = (output / "evidence.json").read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        demo.run_demo(output)
    assert (output / "evidence.json").read_bytes() == previous

    gif = output / "paper-collections.gif"
    create_gif(output / "transcript.txt", gif)
    with Image.open(gif) as image:
        assert image.size == (1120, 540) and image.n_frames == 4
        assert image.info["loop"] == 0
        frames = []
        for index in range(4):
            image.seek(index)
            assert image.info["duration"] == 3500
            frames.append(image.convert("RGB").tobytes())
        assert len(set(frames)) == 4
    before_gif = gif.read_bytes()
    with pytest.raises(FileExistsError):
        create_gif(output / "transcript.txt", gif)
    assert gif.read_bytes() == before_gif


def test_demo_cleans_temporary_database_even_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    temporary = tracked_temporary(monkeypatch, tmp_path)
    monkeypatch.setattr(SQLitePaperCollections, "create", denied)
    output = tmp_path / "failed"
    with pytest.raises(AssertionError, match="Synthetic collection examples"):
        demo.run_demo(output)
    assert temporary and all(not path.exists() for path in temporary)
    assert not list(output.iterdir())


def test_demo_refuses_dangling_output_symlinks(tmp_path: Path) -> None:
    output = tmp_path / "demo"
    output.mkdir()
    missing = tmp_path / "unrelated.txt"
    (output / "checks.json").symlink_to(missing)
    with pytest.raises(FileExistsError):
        demo.run_demo(output)
    assert not missing.exists()


@pytest.mark.parametrize("kind", ["unrelated", "oversized"])
def test_renderer_refuses_invalid_or_clipped_transcripts(tmp_path: Path, kind: str) -> None:
    source = tmp_path / "transcript.txt"
    if kind == "unrelated":
        text = "One\n\nTwo\n\nThree\n\nFour"
        message = "paper collections demo"
    else:
        text = "\n\n".join(
            [demo.PANEL_TITLES[0] + "\n" + "\n".join(["line"] * 20), *demo.PANEL_TITLES[1:]]
        )
        message = "does not fit"
    source.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        create_gif(source, tmp_path / "not-created.gif")
    assert not (tmp_path / "not-created.gif").exists()


@pytest.mark.parametrize("tool", ["demo", "gif", "guide"])
def test_demo_cli_renderer_and_published_python_ignore_ambient_settings(
    tmp_path: Path, tool: str
) -> None:
    ambient = tmp_path / "ambient.sqlite3"
    with sqlite3.connect(ambient) as connection:
        connection.execute("CREATE TABLE sentinel (value TEXT)")
    before = ambient.read_bytes()
    env = {
        **os.environ,
        "SCHOLAR_RAG_DATABASE_PATH": str(ambient),
        "SCHOLAR_RAG_MAX_HOPS": "invalid-ambient",
        **dict.fromkeys(KEYS, "synthetic-unused-key"),
    }
    (tmp_path / ".env").write_text("SCHOLAR_RAG_MAX_SOURCE_DOCS=invalid-dotenv\n", encoding="utf-8")
    guard = (
        "import httpx, runpy, sys\nfrom pathlib import Path\n"
        "from llm.providers import OpenAIAdapter, AnthropicAdapter, GeminiAdapter, KimiAdapter\n"
        "def deny(*args, **kwargs):\n"
        "    raise AssertionError('No network or live provider in this demo')\n"
        "httpx.AsyncHTTPTransport.handle_async_request = deny\n"
        "httpx.HTTPTransport.handle_request = deny\n"
        "for adapter in (OpenAIAdapter, AnthropicAdapter, GeminiAdapter, KimiAdapter):\n"
        "    adapter.generate = deny\n"
    )
    if tool == "demo":
        command = "runpy.run_module('scripts.demo_paper_collections', run_name='__main__')\n"
        arguments = ["--output-dir", str(tmp_path / "output")]
    elif tool == "gif":
        transcript = tmp_path / "transcript.txt"
        transcript.write_text("\n\n".join(demo.PANEL_TITLES), encoding="utf-8")
        command = "runpy.run_module('scripts.create_paper_collections_gif', run_name='__main__')\n"
        arguments = ["--transcript", str(transcript), "--output", str(tmp_path / "output.gif")]
    else:
        guide = Path(__file__).resolve().parents[1] / "docs/guides/PAPER_COLLECTIONS_GUIDE.md"
        command = guide.read_text(encoding="utf-8").split("uv run python - <<'PY'\n", 1)[1]
        command = command.split("\nPY\n```", 1)[0]
        arguments = []
    process = subprocess.run(  # noqa: S603
        [sys.executable, "-c", guard + command, *arguments],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert process.returncode == 0, process.stderr
    assert ambient.read_bytes() == before
    assert "synthetic-unused" not in process.stdout
    if tool == "guide":
        assert "Selected sources: 1" in process.stdout
        assert "State: DONE" in process.stdout and "Revision: 2" in process.stdout
        assert "Surviving history: True" in process.stdout
    elif tool == "demo":
        assert "Total model calls: 1 (the explicit fake query only)" in process.stdout
