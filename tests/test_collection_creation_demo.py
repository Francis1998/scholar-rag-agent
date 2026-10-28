"""Reproduce the measured native workflow without external HTTP or ambient credentials."""

import json
from pathlib import Path

import pytest
from PIL import Image, ImageChops
from scripts.create_collection_builder_gif import create_preview
from scripts.demo_collection_creation import FIXTURE, SELECTED_IDS, run_demo, serve_demo


def test_demo_measures_exact_atomic_creation_and_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ambient = tmp_path / "not-the-demo.sqlite3"
    monkeypatch.setenv("SCHOLAR_RAG_DATABASE_PATH", str(ambient))
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-not-a-credential")
    output = tmp_path / "demo"
    transcript = run_demo(output)
    checks = json.loads((output / "checks.json").read_text())
    assert checks["fixture"] == FIXTURE
    assert checks["corpus_documents"] == 5
    assert checks["filtered_page_sizes"] == [2, 1]
    assert checks["preselected_checkboxes"] == 0
    assert checks["selected_document_ids"] == list(SELECTED_IDS)
    assert checks["selected_count"] == 2 and checks["collection_count"] == 1
    assert checks["redirect_status"] == 303 and checks["revision"] == 1
    assert checks["create_service_attempts"] == 3
    assert checks["rejected_statuses"] == {
        "duplicate-name": 409,
        "empty-selection": 422,
        "unknown-document": 422,
        "cross-origin": 403,
    }
    for key in (
        "get_database_unchanged",
        "rejected_writes_database_unchanged",
        "python_json_equal",
        "restart_selection_equal",
        "restart_html_identical",
        "restart_database_unchanged",
    ):
        assert checks[key] is True
    assert checks["events"] == 0 and not any(checks["forbidden_call_counts"].values())
    assert checks["handoff_pages"] == ["screening", "search", "worksheet"]
    assert (output / "transcript.txt").read_text() == transcript
    committed = Path(__file__).resolve().parents[1] / "docs/assets/collection-builder.txt"
    assert committed.read_text() == transcript
    assert "Native POST: 303" in transcript and "forbidden calls: 0" in transcript
    assert (output / "corpus.sqlite3").is_file() and not ambient.exists()
    gif, still = tmp_path / "preview.gif", tmp_path / "preview.png"
    create_preview(output / "transcript.txt", gif, still)
    with Image.open(gif) as animation, Image.open(still) as preview:
        assert animation.n_frames == 4 and animation.size == (1120, 540)
        assert ImageChops.difference(animation.convert("RGB"), preview).getbbox() is None
    with pytest.raises(FileExistsError):
        create_preview(output / "transcript.txt", gif, still)
    before = (output / "corpus.sqlite3").read_bytes()
    with pytest.raises(FileExistsError):
        run_demo(output)
    assert (output / "corpus.sqlite3").read_bytes() == before


def test_demo_refuses_symlinks_and_unrelated_serve_targets(tmp_path: Path) -> None:
    target = tmp_path / "existing"
    target.mkdir()
    link = tmp_path / "linked"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(FileExistsError):
        run_demo(link)
    (target / "checks.json").write_text('{"fixture":"unrelated"}')
    with pytest.raises(ValueError, match="existing native collection creation demo"):
        serve_demo(target, 8765)
    transcript = tmp_path / "invalid.txt"
    transcript.write_text("Not a measured native collection demo")
    with pytest.raises(ValueError, match="four bounded"):
        create_preview(transcript, tmp_path / "bad.gif", tmp_path / "bad.png")
    assert not (tmp_path / "bad.gif").exists()
