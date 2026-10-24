"""Published progress illustrations must come from measured offline report downloads."""

import hashlib
import json
import re
import sys
from pathlib import Path

import pytest
from PIL import Image
from scripts.create_screening_report_gif import create_preview
from scripts.demo_screening_report import MEMBER_IDS, OUTPUT_NAMES, run_demo

from storage.screening_report import ScreeningReport


def test_demo_downloads_reproduce_offline_and_render_four_distinct_frames(tmp_path: Path) -> None:
    output = tmp_path / "demo"
    transcript = run_demo(output)
    assert {path.name for path in output.iterdir()} == set(OUTPUT_NAMES)
    assert transcript == (output / "transcript.txt").read_text()
    report = ScreeningReport.model_validate_json((output / "report.json").read_bytes())
    checks = json.loads((output / "checks.json").read_text())
    assert report.snapshot.total_documents == 9
    assert report.resolved_documents == 5 and report.needs_review_documents == 4
    assert report.resolved_document_ids == MEMBER_IDS[:5]
    assert report.needs_review_document_ids == MEMBER_IDS[5:]
    assert [group.count for group in report.exclusion_reasons] == [2, 1, 1]
    assert checks["visible_members"] == 1 and checks["report_members"] == 9
    assert checks["stale_revision_http_status"] == 409
    assert checks["restart_identical"] and checks["database_bytes_unchanged"]
    assert checks["database_removed"] and checks["offline_json_reproduces_markdown"]
    assert checks["event_count"] == 0
    assert not any(checks["forbidden_call_counts"].values())
    assert (output / "report.md").read_bytes() == report.to_markdown().encode("utf-8")
    assert (
        hashlib.sha256((output / "snapshot.json").read_bytes()).hexdigest()
        == report.snapshot_sha256
    )
    for name, size in checks["output_bytes"].items():
        content = (output / name).read_bytes()
        assert len(content) == size
        assert checks["sha256"][name] == hashlib.sha256(content).hexdigest()
    gif, still = tmp_path / "preview.gif", tmp_path / "preview.png"
    create_preview(output / "transcript.txt", gif, still)
    with Image.open(gif) as animation, Image.open(still) as preview:
        assert animation.n_frames == 4 and animation.info["loop"] == 0
        assert animation.size == preview.size == (1120, 540)
        assert animation.convert("RGB").tobytes() == preview.convert("RGB").tobytes()
        frames = set()
        for index in range(4):
            animation.seek(index)
            assert animation.info["duration"] == 3500
            frames.add(hashlib.sha256(animation.convert("RGB").tobytes()).hexdigest())
        assert len(frames) == 4
    with pytest.raises(FileExistsError):
        create_preview(output / "transcript.txt", gif, still)


@pytest.mark.parametrize("kind", ["directory", "file", "symlink"])
def test_demo_never_overwrites_user_output(tmp_path: Path, kind: str) -> None:
    target = tmp_path / "existing"
    if kind == "directory":
        target.mkdir()
    elif kind == "file":
        target.write_text("Keep this")
    else:
        target.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        run_demo(target)
    if kind == "file":
        assert target.read_text() == "Keep this"


def test_demo_ignores_ambient_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = tmp_path / "user.sqlite3"
    sentinel.write_bytes(b"DO NOT OPEN")
    monkeypatch.setenv("SCHOLAR_RAG_DATABASE_PATH", str(sentinel))
    monkeypatch.setenv("SCHOLAR_RAG_DEFAULT_MODEL", "not-a-provider")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unused-private-fixture")
    output = tmp_path / "demo"
    run_demo(output)
    assert sentinel.read_bytes() == b"DO NOT OPEN"
    assert all(b"unused-private-fixture" not in path.read_bytes() for path in output.iterdir())


@pytest.mark.parametrize("text", ["Not the executed demo", "x" * 16385])
def test_preview_refuses_wrong_or_oversized_transcript(tmp_path: Path, text: str) -> None:
    transcript = tmp_path / "transcript.txt"
    transcript.write_text(text)
    with pytest.raises(ValueError):
        create_preview(transcript, tmp_path / "preview.gif", tmp_path / "preview.png")
    assert not (tmp_path / "preview.gif").exists()
    assert not (tmp_path / "preview.png").exists()


def test_committed_preview_and_links_match_executed_transcript(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    assets = root / "docs/assets"
    gif, still = tmp_path / "preview.gif", tmp_path / "preview.png"
    create_preview(assets / "screening-progress-report.txt", gif, still)
    assert gif.read_bytes() == (assets / "screening-progress-report.gif").read_bytes()
    with Image.open(still) as actual, Image.open(assets / "screening-progress-report.png") as saved:
        assert actual.size == saved.size
        assert actual.convert("RGB").tobytes() == saved.convert("RGB").tobytes()
    for document in (root / "README.md", root / "docs/guides/SCREENING_REPORT_GUIDE.md"):
        text = document.read_text()
        assert "screening-progress-report.png" in text
        assert "screening-progress-report.gif" in text
        assert "screening-progress-report.txt" in text


def test_standalone_python_guide_saves_real_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    guide = Path(__file__).resolve().parents[1] / "docs/guides/SCREENING_REPORT_GUIDE.md"
    blocks = re.findall(r"```python\n(.*?)```", guide.read_text(), re.DOTALL)
    output = tmp_path / "guide"
    monkeypatch.setattr(sys, "argv", ["screening_report_example.py", str(output)])
    exec(compile(blocks[0], str(guide), "exec"), {})  # noqa: S102
    report = ScreeningReport.model_validate_json((output / "report.json").read_bytes())
    assert report.resolved_documents == 1 and report.needs_review_documents == 1
    assert report.exclusion_reasons[0].document_ids == ("synthetic-outside",)
    assert (output / "report.md").read_bytes() == report.to_markdown().encode("utf-8")
