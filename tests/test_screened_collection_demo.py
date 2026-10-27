"""The screened-scope illustration must come from measured offline behavior."""

import json
from pathlib import Path

import pytest
from PIL import Image
from scripts.demo_screened_collection import ARTIFACT_NAMES, create_gif, run_demo


def test_demo_ignores_ambient_configuration_and_renders_actual_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "MOONSHOT_API_KEY"):
        monkeypatch.setenv(name, "must-not-be-used")
    monkeypatch.setenv("SCHOLAR_RAG_DEFAULT_MODEL", "not-a-provider")
    monkeypatch.setenv("SCHOLAR_RAG_OPENAI_MODEL", "")
    untouched = tmp_path / "must-not-exist.sqlite3"
    monkeypatch.setenv("SCHOLAR_RAG_DATABASE_PATH", str(untouched))
    output = tmp_path / "demo"
    transcript = run_demo(output)
    measured = json.loads((output / "screened-demo.json").read_text())
    preview = json.loads((output / "screened-preview.json").read_text())
    evidence = json.loads((output / "screened-evidence.json").read_text())
    assert measured["synthetic_only"] is True
    assert len(measured["ordinary_document_ids"]) == 3
    assert len(measured["screened_document_ids"]) == 1
    assert measured["screened_document_ids"] == measured["python_selected_document_ids"]
    assert preview["plan"]["observation"]["document_ids"] == measured["screened_document_ids"]
    assert evidence["plan"]["observation"]["document_ids"] == measured["screened_document_ids"]
    assert measured["selection_survived_restart"] is True
    assert measured["previews_read_only"] is True
    assert measured["preview_event_count"] == 0
    assert measured["saved_state"] == "DONE"
    assert measured["generation_provider"] == "fake"
    assert measured["empty_preview_status"] == measured["empty_query_status"] == 409
    assert measured["rejections_read_only"] is True
    assert measured["stale_revision_status"] == 409
    assert measured["saved_evidence_unchanged"] is True
    assert not untouched.exists()
    assert transcript == (output / "transcript.txt").read_text()
    gif = output / "screened-collection.gif"
    create_gif(transcript, gif)
    with Image.open(gif) as image:
        assert image.format == "GIF"
        assert image.size == (1120, 540)
        assert image.n_frames == 4
        frames = []
        for index in range(image.n_frames):
            image.seek(index)
            assert image.info["duration"] == 3500
            frames.append(image.convert("RGB").tobytes())
        assert len(set(frames)) == 4
    with pytest.raises(FileExistsError):
        create_gif(transcript, gif)
    with pytest.raises(FileExistsError):
        run_demo(output)


@pytest.mark.parametrize("name", ARTIFACT_NAMES)
def test_demo_preserves_existing_artifacts(tmp_path: Path, name: str) -> None:
    target = tmp_path / name
    target.write_text("keep this artifact", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run_demo(tmp_path)
    assert target.read_text() == "keep this artifact"
