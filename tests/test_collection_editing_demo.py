"""Reproduce the exact delta, restart and nonempty historical preservation measurements."""

import json
from pathlib import Path

import pytest
from PIL import Image, ImageChops
from scripts.create_collection_editor_gif import create_preview
from scripts.demo_collection_editing import FIXTURE, run_demo, serve_demo
from scripts.demo_frozen_annotation import DOCUMENT_ID


def test_executed_editor_demo_preserves_history_and_matches_committed_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ambient = tmp_path / "not-the-demo.sqlite3"
    monkeypatch.setenv("SCHOLAR_RAG_DATABASE_PATH", str(ambient))
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-not-a-credential")
    output = tmp_path / "demo"
    transcript = run_demo(output)
    checks = json.loads((output / "checks.json").read_text())
    assert checks["fixture"] == FIXTURE
    assert checks["before_document_ids"] == [DOCUMENT_ID, "01-retained", "03-off-page"]
    assert checks["after_document_ids"] == ["01-retained", "03-off-page", "02-addition"]
    assert checks["before_revision"] == 1 and checks["after_revision"] == 2
    assert checks["save_status"] == 303 and checks["preview_status"] == 200
    assert checks["stale_save_statuses"] == [409, 409]
    assert checks["protected_row_counts"] == {
        "documents": 4,
        "chunks": 4,
        "events": 8,
        "decisions": 2,
        "history": 3,
    }
    assert checks["screening_counts_after"] == {
        "stale": 1,
        "unscreened": 2,
        "include": 0,
        "exclude": 0,
        "unsure": 0,
    }
    assert not any(checks["forbidden_call_counts"].values())
    for key in (
        "get_and_preview_database_unchanged",
        "rejected_writes_database_unchanged",
        "python_json_equal",
        "restart_collection_equal",
        "restart_html_identical",
        "protected_table_rows_unchanged",
        "frozen_evidence_unchanged",
    ):
        assert checks[key] is True
    assert not ambient.exists()
    assert (output / "transcript.txt").read_text() == transcript
    assert (
        Path(__file__).resolve().parents[1] / "docs/assets/collection-editor.txt"
    ).read_text() == transcript
    gif, still = tmp_path / "preview.gif", tmp_path / "preview.png"
    create_preview(output / "transcript.txt", gif, still)
    assert gif.stat().st_size < 500_000
    with Image.open(gif) as animation, Image.open(still) as preview:
        assert animation.n_frames == 4 and animation.size == (1120, 540)
        assert ImageChops.difference(animation.convert("RGB"), preview).getbbox() is None
    with pytest.raises(FileExistsError):
        create_preview(output / "transcript.txt", gif, still)
    before = (output / "corpus.sqlite3").read_bytes()
    with pytest.raises(FileExistsError):
        run_demo(output)
    assert (output / "corpus.sqlite3").read_bytes() == before


def test_demo_refuses_symlinks_invalid_transcripts_and_unrelated_serve_targets(
    tmp_path: Path,
) -> None:
    target = tmp_path / "existing"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(FileExistsError):
        run_demo(link)
    (target / "checks.json").write_text('{"fixture":"unrelated"}')
    with pytest.raises(ValueError, match="existing native collection editing demo"):
        serve_demo(target, 8765)
    transcript = tmp_path / "invalid.txt"
    transcript.write_text("Not a measured native collection demo")
    with pytest.raises(ValueError, match="four bounded"):
        create_preview(transcript, tmp_path / "bad.gif", tmp_path / "bad.png")
