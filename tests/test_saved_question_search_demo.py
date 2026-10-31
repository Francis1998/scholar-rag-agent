"""Execute the offline measurements and validate their opt-in illustration artifacts."""

import json
from pathlib import Path

import pytest
from PIL import Image
from scripts.demo_saved_question_search import NEEDLE, create_preview, run_demo


def test_executed_demo_records_restart_navigation_and_zero_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SCHOLAR_RAG_DATABASE_PATH", str(tmp_path / "do-not-create.sqlite3"))
    monkeypatch.setenv("SCHOLAR_RAG_DEFAULT_MODEL", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "synthetic-unused-key")
    monkeypatch.setenv("SCHOLAR_RAG_MAX_SOURCE_DOCS", "invalid-ambient-value")
    output = tmp_path / "demo"
    transcript = run_demo(output)
    assert not (tmp_path / "do-not-create.sqlite3").exists()
    assert (output / "transcript.txt").read_text(encoding="utf-8") == transcript
    checks = json.loads((output / "checks.json").read_bytes())
    assert checks["all_runs"] == 6 and checks["question_matches"] == 3
    assert [len(page) for page in checks["done_pages"]] == [1, 1]
    assert len({run_id for page in checks["done_pages"] for run_id in page}) == 2
    assert checks["long_question_match_start"] > checks["preview_characters"] == 300
    assert checks["lowercase_matches"] == checks["regex_like_matches"] == 0
    assert checks["empty_api_status"] == 422
    assert checks["native_return_consumers"] == 6 and checks["mount_prefix"] == "/lab"
    assert checks["restart_identical"] and checks["schema_and_database_unchanged"]
    assert checks["events_before"] == checks["events_after"] == 22
    assert not any(checks["forbidden_call_counts"].values())
    assert checks["network_connection_calls"] == 0
    page = json.loads((output / "done-next.json").read_bytes())
    assert NEEDLE not in page["runs"][0]["query_summary"]
    assert page["runs"][0]["query_truncated"]
    assert (output / "history-next.html").read_bytes() == (output / "restarted.html").read_bytes()
    assert "synthetic-unused-key" not in (output / "reader.html").read_text(encoding="utf-8")
    create_preview(output)
    with Image.open(output / "saved-question-search.gif") as image:
        assert image.n_frames == 4 and image.size == (1120, 540)
        for index in range(image.n_frames):
            image.seek(index)
            assert image.info["duration"] == 3500
    with Image.open(output / "saved-question-search.png") as image:
        assert image.size == (1120, 540)
    before = {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_demo(output)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        create_preview(output)
    assert before == {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}


def test_illustration_rejects_unrelated_or_overflowing_text(tmp_path: Path) -> None:
    from scripts.demo_saved_question_search import PANEL_TITLES

    source = tmp_path / "transcript.txt"
    source.write_text("unrelated", encoding="utf-8")
    with pytest.raises(ValueError, match="four measured"):
        create_preview(tmp_path)
    source.write_text(
        "\n\n".join(title + "\n" + "W " * 2000 for title in PANEL_TITLES), encoding="utf-8"
    )
    with pytest.raises(ValueError, match=r"fit|width"):
        create_preview(tmp_path)
    assert not (tmp_path / "saved-question-search.gif").exists()
    assert not (tmp_path / "saved-question-search.png").exists()
