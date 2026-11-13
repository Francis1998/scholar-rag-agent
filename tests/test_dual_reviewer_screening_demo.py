"""Published dual-reviewer illustrations and numbers must come from measured offline runs."""

import hashlib
import json
import re
import sys
from pathlib import Path

import pytest
from PIL import Image
from scripts.create_dual_reviewer_screening_gif import create_preview
from scripts.demo_dual_reviewer_screening import OUTPUT_NAMES, run_demo

from storage.dual_screening import AgreementReport, ScreeningConflicts, ScreeningResolution

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs/assets"
GUIDE = ROOT / "docs/guides/DUAL_REVIEWER_SCREENING_GUIDE.md"


def test_demo_measures_blinding_resolution_and_kappa_and_renders_four_frames(
    tmp_path: Path,
) -> None:
    output = tmp_path / "demo"
    transcript = run_demo(output)
    assert {path.name for path in output.iterdir()} == set(OUTPUT_NAMES)
    assert transcript == (output / "transcript.txt").read_text()
    checks = json.loads((output / "checks.json").read_text())
    assert checks["blind_items_hidden"] == 12 and not checks["blind_queue_has_r1_reasons"]
    assert checks["team_view_while_blind"] == [409, "screening_blind_mode_on"]
    assert checks["member_page_hidden"] and checks["lifted_after_first_decision"] == 1
    assert checks["decisions_unchanged"] and checks["kappa_unchanged_by_resolutions"]
    assert checks["report_matches_agreement"] and checks["browser_matches_api"]
    assert checks["restart_identical"] and checks["database_removed"]
    assert not any(checks["forbidden_call_counts"].values())
    agreement = AgreementReport.model_validate_json((output / "agreement.json").read_bytes())
    primary = agreement.pairs[0].statistics
    assert primary.confusion == ((4, 1, 0), (1, 5, 0), (0, 1, 0))
    assert primary.kappa == pytest.approx((0.75 - 67 / 144) / (1 - 67 / 144), abs=1e-6)
    assert agreement.conflicts.unresolved == 1 and agreement.conflicts.resolved == 2
    conflicts = ScreeningConflicts.model_validate_json((output / "conflicts.json").read_bytes())
    assert [item.document_id for item in conflicts.items] == [
        "near-miss-05",
        "near-miss-11",
        "near-miss-12",
    ]
    resolution = ScreeningResolution.model_validate_json((output / "resolution.json").read_bytes())
    assert resolution.resolver == "lead" and resolution.decision == "exclude"
    for name, size in checks["output_bytes"].items():
        content = (output / name).read_bytes()
        assert len(content) == size
        assert checks["sha256"][name] == hashlib.sha256(content).hexdigest()
    member = (output / "r2-member-blind.html").read_text()
    assert "EXPLICIT HUMAN WRITE / INDEPENDENT DUAL REVIEW" in member
    assert 'id="others-hidden"' in member
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
    assert (ASSETS / "dual-reviewer-screening.txt").read_text() == transcript


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
    monkeypatch.setenv("OPENAI_API_KEY", "unused-private-fixture")
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


def test_committed_preview_and_links_match_the_transcript(tmp_path: Path) -> None:
    gif, still = tmp_path / "preview.gif", tmp_path / "preview.png"
    create_preview(ASSETS / "dual-reviewer-screening.txt", gif, still)
    assert gif.read_bytes() == (ASSETS / "dual-reviewer-screening.gif").read_bytes()
    assert (ASSETS / "dual-reviewer-screening.gif").stat().st_size < 800_000
    with Image.open(still) as actual, Image.open(ASSETS / "dual-reviewer-screening.png") as saved:
        assert actual.size == saved.size
        assert actual.convert("RGB").tobytes() == saved.convert("RGB").tobytes()
    for document in (ROOT / "README.md", GUIDE):
        text = document.read_text()
        for name in (
            "dual-reviewer-screening.png",
            "dual-reviewer-screening.gif",
            "dual-reviewer-screening.txt",
        ):
            assert name in text, (document.name, name)


def test_standalone_python_guide_example_runs(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    blocks = re.findall(r"```python\n(.*?)```", GUIDE.read_text(), re.DOTALL)
    monkeypatch.setattr(sys, "argv", ["dual_reviewer_example.py"])
    exec(compile(blocks[0], str(GUIDE), "exec"), {})  # noqa: S102
    output = capsys.readouterr().out
    expected = re.search(r"Expected output:\n\n```text\n(.*?)```", GUIDE.read_text(), re.DOTALL)
    assert expected is not None and output == expected.group(1)
