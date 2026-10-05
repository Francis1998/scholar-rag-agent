"""Measured synthetic outputs, reproducible illustration, and offline isolation."""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from PIL import Image
from scripts.create_literal_search_gif import create_gif
from scripts.demo_literal_search import (
    DEMO_LITERAL,
    DEMO_QUERY,
    DEMO_TEXT,
    OUTPUT_NAMES,
    PANEL_TITLES,
    run_demo,
)

from storage.literal_search import LiteralSearchPage


def test_measured_workflow_and_reproducible_four_frame_illustration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    output = tmp_path / "output"
    transcript = run_demo(output)
    assert {path.name for path in output.iterdir()} == set(OUTPUT_NAMES)
    assert (output / "transcript.txt").read_text() == transcript
    assert run_demo(tmp_path / "repeat") == transcript
    first = LiteralSearchPage.model_validate_json((output / "page-1.json").read_bytes())
    second = LiteralSearchPage.model_validate_json((output / "page-2.json").read_bytes())
    third = LiteralSearchPage.model_validate_json((output / "page-3.json").read_bytes())
    literal = LiteralSearchPage.model_validate_json((output / "literal.json").read_bytes())
    assert first.matches[0].match_start == DEMO_TEXT.index(DEMO_QUERY) > 800
    assert (
        first.matches[0].excerpt
        == DEMO_TEXT[first.matches[0].excerpt_start : first.matches[0].excerpt_end]
    )
    assert [match.chunk_id for page in (first, second, third) for match in page.matches] == [
        "chunk-1",
        "chunk-2",
        "chunk-3",
    ]
    assert len(literal.matches) == 2 and literal.query == DEMO_LITERAL
    assert (output / "python.json").read_bytes() == (output / "page-1.json").read_bytes()
    assert (output / "restarted.json").read_bytes() == (output / "page-1.json").read_bytes()
    checks = json.loads((output / "checks.json").read_text())
    assert checks["synthetic_only"] and checks["temporary_database_removed"]
    assert checks["whole_corpus_matches"] == 4 and checks["selected_matches"] == 3
    assert checks["page_sizes"] == [1, 1, 1]
    assert checks["match_start"] == first.matches[0].match_start
    assert checks["match_end"] == first.matches[0].match_end
    assert checks["title_characters"] == 300 and checks["title_truncated"]
    assert checks["empty_matches"] == 0 and checks["literal_matches"] == 2
    assert checks["first_page_bytes"] == (output / "page-1.json").stat().st_size
    assert checks["mismatch_status"] == 422 and checks["mismatch_code"] == "invalid_search_cursor"
    assert checks["python_http_identical"] and checks["restart_identical"]
    assert checks["network_calls"] == checks["search_model_retrieval_or_write_calls"] == 0
    assert checks["events_before"] == checks["events_after"] == 0
    assert checks["database_sha256"]["before"] == checks["database_sha256"]["after"]
    assert not list(tmp_path.rglob("*.sqlite3"))
    assert str(first.matches[0].match_start) in transcript and DEMO_QUERY in transcript
    first_gif, repeated_gif = tmp_path / "one.gif", tmp_path / "two.gif"
    create_gif(output / "transcript.txt", first_gif)
    create_gif(output / "transcript.txt", repeated_gif)
    assert first_gif.read_bytes() == repeated_gif.read_bytes()
    with Image.open(first_gif) as image:
        assert image.format == "GIF" and image.is_animated and image.n_frames == 4
        assert image.size == (1120, 540) and image.info["loop"] == 0
        frames = set()
        for index in range(image.n_frames):
            image.seek(index)
            assert image.info["duration"] == 3500
            frames.add(hashlib.sha256(image.convert("RGB").tobytes()).hexdigest())
        assert len(frames) == 4
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        create_gif(output / "transcript.txt", first_gif)


@pytest.mark.parametrize("kind", ["directory", "file", "symlink"])
def test_demo_refuses_existing_output_targets(tmp_path: Path, kind: str) -> None:
    output = tmp_path / "output"
    if kind == "directory":
        output.mkdir()
        (output / "keep.txt").write_text("Keep")
    elif kind == "file":
        output.write_text("Keep")
    else:
        output.symlink_to(tmp_path / "absent")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_demo(output)
    if kind == "directory":
        assert (output / "keep.txt").read_text() == "Keep"
    elif kind == "file":
        assert output.read_text() == "Keep"
    else:
        assert output.is_symlink() and not (tmp_path / "absent").exists()


@pytest.mark.parametrize("kind", ["titles", "empty", "oversized", "wide", "tall"])
def test_renderer_rejects_unmeasured_or_clipped_transcripts(tmp_path: Path, kind: str) -> None:
    content = {
        "titles": "Not this demonstration",
        "empty": "\n\n".join(PANEL_TITLES),
        "oversized": "x" * 16385,
        "wide": "\n\n".join(f"{title}\n{'W' * 90}" for title in PANEL_TITLES),
        "tall": "\n\n".join(f"{title}\n{'word ' * 300}" for title in PANEL_TITLES),
    }[kind]
    transcript = tmp_path / "transcript.txt"
    transcript.write_text(content)
    output = tmp_path / "not-created.gif"
    with pytest.raises(ValueError):
        create_gif(transcript, output)
    assert not output.exists()


def test_cli_ignores_ambient_keys_invalid_settings_and_private_database(tmp_path: Path) -> None:
    sentinel = tmp_path / "private.sqlite3"
    sentinel.write_bytes(b"Keep private database unchanged")
    (tmp_path / ".env").write_text("SCHOLAR_RAG_MAX_HOPS=invalid")
    environment = {
        **os.environ,
        "SCHOLAR_RAG_DATABASE_PATH": str(sentinel),
        "SCHOLAR_RAG_MAX_SOURCE_DOCS": "invalid",
        "SCHOLAR_RAG_DEFAULT_MODEL": "anthropic",
        "OPENAI_API_KEY": "unused-synthetic-key",
        "ANTHROPIC_API_KEY": "unused-synthetic-key",
        "GEMINI_API_KEY": "unused-synthetic-key",
        "MOONSHOT_API_KEY": "unused-synthetic-key",
    }
    output = tmp_path / "output"
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "scripts.demo_literal_search", "--output-dir", str(output)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert sentinel.read_bytes() == b"Keep private database unchanged"
    assert not (tmp_path / ".scholar-rag-agent.sqlite3").exists()
    checks = json.loads((output / "checks.json").read_text())
    assert checks["network_calls"] == checks["search_model_retrieval_or_write_calls"] == 0
    for path in output.iterdir():
        assert b"unused-synthetic-key" not in path.read_bytes()
