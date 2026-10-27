"""Reproduce the isolated native workflow and verify its guide and executed screenshot assets."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image
from scripts.create_frozen_annotation_gif import create_preview
from scripts.demo_frozen_annotation import (
    CHUNK_ID,
    DOCUMENT_ID,
    FRAME_PAGES,
    NOTE,
    QUOTE,
    RUN_ID,
    TEXT,
    AuthoringPage,
    run_demo,
)

from storage.evidence_annotations import AnnotationPage, SavedEvidenceAnnotation
from tests.test_corpus_explorer import ExplorerAPI
from tests.test_corpus_explorer import api as api
from tests.test_review_notebook import TEXT as TEST_TEXT
from tests.test_review_notebook import fixture_run

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs/assets"
GUIDE = ROOT / "docs/guides/FROZEN_ANNOTATION_WORKBENCH_GUIDE.md"


def test_demo_reproduces_exact_preview_retry_and_restart_contract(tmp_path: Path) -> None:
    output = tmp_path / "demo"
    transcript = run_demo(output)
    checks = json.loads((output / "checks.json").read_text())
    assert checks["run_id"] == RUN_ID
    assert checks["start"] == TEXT.rindex(QUOTE) == 139 and checks["end"] == 175
    assert checks["first_occurrence"] < checks["start"]
    assert checks["annotations"] == 1
    assert checks["save_status"] == checks["retry_status"] == 303
    assert checks["rejected_statuses"] == {
        "cross_origin": 403,
        "duplicate_field": 422,
        "changed_uuid_payload": 409,
    }
    assert checks["create_service_attempts_before_restart"] == 3
    assert checks["events_before"] == checks["events_after"] == 8
    assert not any(checks["forbidden_call_counts"].values())
    for key in (
        "get_preview_database_unchanged",
        "rejected_writes_database_unchanged",
        "retry_record_unchanged",
        "python_api_equal",
        "corpus_change_delete_restart_identical",
    ):
        assert checks[key] is True
    record = AnnotationPage.model_validate_json(
        (output / "annotations.json").read_bytes()
    ).annotations[0]
    assert record.document_id == DOCUMENT_ID and record.chunk_id == CHUNK_ID
    assert record.note == NOTE and record.quote == QUOTE
    page = AuthoringPage((output / "saved-quote.html").read_bytes().decode())
    assert page.passages == [TEXT] and page.highlights == [QUOTE]
    assert page.values["annotation-note"] == [NOTE]
    assert (output / "deleted-restart.html").read_bytes() == (
        output / "saved-quote.html"
    ).read_bytes()
    assert transcript == (ASSETS / "frozen-annotation-workbench.txt").read_text()
    assert run_demo(tmp_path / "repeat") == transcript
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_demo(output)
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_demo(alias)


def test_cli_ignores_ambient_private_settings_and_network(tmp_path: Path) -> None:
    sentinel = tmp_path / "private.sqlite3"
    sentinel.write_bytes(b"untouched")
    (tmp_path / ".env").write_text("SCHOLAR_RAG_MAX_HOPS=invalid\n")
    output = tmp_path / "demo"
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "scripts.demo_frozen_annotation", "--output-dir", str(output)],
        cwd=tmp_path,
        env={
            **os.environ,
            "SCHOLAR_RAG_DATABASE_PATH": str(sentinel),
            "SCHOLAR_RAG_DEFAULT_MODEL": "anthropic",
            "SCHOLAR_RAG_MAX_HOPS": "invalid",
            "OPENAI_API_KEY": "synthetic-unused",
            "ANTHROPIC_API_KEY": "synthetic-unused",
            "GEMINI_API_KEY": "synthetic-unused",
            "MOONSHOT_API_KEY": "synthetic-unused",
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert sentinel.read_bytes() == b"untouched"
    assert all(b"synthetic-unused" not in path.read_bytes() for path in output.iterdir())


def test_committed_animation_and_opt_in_preview_are_four_distinct_browser_frames() -> None:
    with Image.open(ASSETS / "frozen-annotation-workbench.gif") as image:
        assert image.size == (1280, 960) and image.n_frames == 4
        frames = set()
        for index in range(4):
            image.seek(index)
            assert image.info["duration"] == 3000
            frames.add(image.convert("RGB").tobytes())
        assert len(frames) == 4
    with Image.open(ASSETS / "frozen-annotation-workbench.png") as image:
        assert image.format == "PNG" and image.size == (1280, 960)


def test_screenshot_assembler_refuses_overwrites_and_bad_sizes(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    for index, name in enumerate(FRAME_PAGES):
        Image.new("RGB", (1280, 960), (index * 60, 30, 90)).save(frames / f"{name}.png")
    gif, still = tmp_path / "new.gif", tmp_path / "new.png"
    create_preview(frames, gif, still)
    assert still.read_bytes() == (frames / "frozen-source.png").read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        create_preview(frames, gif, still)
    with pytest.raises(ValueError, match="different paths"):
        create_preview(frames, tmp_path / "same", tmp_path / "same")
    Image.new("RGB", (10, 10)).save(frames / "preview.png")
    with pytest.raises(ValueError, match="1280x960"):
        create_preview(frames, tmp_path / "bad.gif", tmp_path / "bad.png")
    assert not (tmp_path / "bad.gif").exists()


def test_guide_indexes_and_inspiration_are_explicit() -> None:
    text = GUIDE.read_text()
    for phrase in (
        "## Native browser workflow",
        "## API contract",
        "## Python",
        "## Portfolio",
        "## Reproduce",
        "same-origin",
        "DNS rebinding",
        "no authentication",
        "workflow inspiration only",
        "https://github.com/Future-House/paper-qa",
        "https://github.com/hypothesis/client",
        "https://web.hypothes.is/help/annotation-basics/",
        "2026-10-26",
        "GPT-6 Astra",
        "Sonnet 5.5",
        "Gemini 3.8 Flash",
        "Kimi K3.1 Pro",
        "four actual local browser",
        "not a continuous",
        "65,536",
        "16,384",
        "303",
    ):
        assert phrase in text
    for path in (ROOT / "README.md", ROOT / "docs/README.md", ROOT / "docs/DEMO.md"):
        assert GUIDE.name in path.read_text()
    for suffix in ("png", "gif", "txt"):
        assert f"frozen-annotation-workbench.{suffix}" in text
    assert "](../assets/frozen-annotation-workbench.gif)" in text


def test_documented_python_preview_and_explicit_confirmation_work(
    api: ExplorerAPI, tmp_path: Path
) -> None:
    bundle = fixture_run(api)
    source = bundle.snapshot.sources[0]
    start = TEST_TEXT.rindex("Repeated quote")
    code = GUIDE.read_text().split("```python\n", 1)[1].split("```", 1)[0]
    for old, new in (
        ('"/absolute/path/existing-evidence.sqlite3"', repr(str(api.path))),
        ('"COPY_EXACT_RUN_ID"', repr(bundle.run_id)),
        ('"COPY_EXACT_CHUNK_ID"', repr(source.chunk.chunk_id)),
        ("COPY_START_CODE_POINT", str(start)),
        ("COPY_END_CODE_POINT", str(start + 14)),
    ):
        code = code.replace(old, new)
    before = api.path.read_bytes()
    rejected = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        cwd=tmp_path,
        input="CANCEL\n",
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert rejected.returncode != 0 and "Not saved." in rejected.stderr
    assert api.path.read_bytes() == before
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        cwd=tmp_path,
        input="SAVE\n",
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    saved = SavedEvidenceAnnotation.model_validate_json(
        result.stdout.split("Type SAVE to persist this exact preview: ", 1)[1]
    )
    assert saved.start == start and saved.quote == "Repeated quote"
    assert api.container.evidence_annotations.list_annotations(bundle.run_id).annotations == [saved]
