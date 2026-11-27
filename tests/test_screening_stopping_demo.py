"""Published stopping-rule illustrations and numbers must come from measured offline runs."""

import hashlib
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

import pytest
from PIL import Image
from scripts.create_screening_stopping_gif import create_preview
from scripts.demo_screening_stopping import OUTPUT_NAMES, run_demo

from storage.screening_stopping import ScreeningStoppingAssessment

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs/assets"
GUIDE = ROOT / "docs/guides/SCREENING_STOPPING_RULES_GUIDE.md"


def test_demo_measures_safe_buscar_and_renders_four_frames(tmp_path: Path) -> None:
    output = tmp_path / "demo"
    transcript = run_demo(output)
    assert {path.name for path in output.iterdir()} == set(OUTPUT_NAMES)
    assert transcript == (output / "transcript.txt").read_text()
    checks = json.loads((output / "checks.json").read_text())
    assert checks["pilot"] == [10, 2, 20]
    assert (checks["checkpoint_screened"], checks["checkpoint_found"]) == (43, 20)
    assert checks["sample"] == [43, 57, 2] and checks["projection"] == 44
    # K_hat = 2 and no relevant sampled: p(n) = (57 - n)(56 - n) / (57 * 56).
    assert [step[0] for step in checks["progression"]] == list(range(45))
    for n, k, p in checks["progression"]:
        assert k == 0 and Fraction(p) == Fraction((57 - n) * (56 - n), 57 * 56)
    assert checks["final_p"] == "13/266" and checks["screened_total"] == 87
    assert checks["final_bounds"] == [1.0, pytest.approx(20 / 21)]
    assert checks["browser_matches_api"] and checks["restart_identical"]
    assert checks["unchanged_by_read"] and checks["database_removed"]
    assert checks["post_status"] == [405, "screening_stopping_method_not_allowed"]
    assert not any(checks["forbidden_call_counts"].values())
    final = ScreeningStoppingAssessment.model_validate_json(
        (output / "sample-final.json").read_bytes()
    )
    assert final.recall_test.status == "met" and final.safe_checkpoint.status == "met"
    for name, size in checks["output_bytes"].items():
        content = (output / name).read_bytes()
        assert len(content) == size
        assert checks["sha256"][name] == hashlib.sha256(content).hexdigest()
    page = (output / "stopping.html").read_text()
    assert "READ-ONLY / SCREENING STOPPING RULES" in page and 'id="recall-status">met<' in page
    gif, still = tmp_path / "preview.gif", tmp_path / "preview.png"
    create_preview(output / "transcript.txt", gif, still)
    with Image.open(gif) as animation, Image.open(still) as preview:
        assert animation.n_frames == 4 and animation.info["loop"] == 0
        assert animation.size == preview.size == (1120, 540)
    with pytest.raises(FileExistsError):
        create_preview(output / "transcript.txt", gif, still)
    assert (ASSETS / "screening-stopping-rules.txt").read_text() == transcript


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


def test_committed_preview_and_links_match_the_transcript(tmp_path: Path) -> None:
    gif, still = tmp_path / "preview.gif", tmp_path / "preview.png"
    create_preview(ASSETS / "screening-stopping-rules.txt", gif, still)
    assert gif.read_bytes() == (ASSETS / "screening-stopping-rules.gif").read_bytes()
    assert (ASSETS / "screening-stopping-rules.gif").stat().st_size < 800_000
    with Image.open(still) as actual, Image.open(ASSETS / "screening-stopping-rules.png") as saved:
        assert actual.convert("RGB").tobytes() == saved.convert("RGB").tobytes()
    for document in (ROOT / "README.md", GUIDE):
        text = document.read_text()
        for name in (
            "screening-stopping-rules.png",
            "screening-stopping-rules.gif",
            "screening-stopping-rules.txt",
        ):
            assert name in text, (document.name, name)


def test_standalone_python_guide_example_runs(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    blocks = re.findall(r"```python\n(.*?)```", GUIDE.read_text(), re.DOTALL)
    monkeypatch.setattr(sys, "argv", ["stopping_rules_example.py"])
    exec(compile(blocks[0], str(GUIDE), "exec"), {})  # noqa: S102
    output = capsys.readouterr().out
    expected = re.search(r"Expected output:\n\n```text\n(.*?)```", GUIDE.read_text(), re.DOTALL)
    assert expected is not None and output == expected.group(1)
