"""The published reference-import illustration must come from a measured offline run."""

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image
from scripts.create_reference_import_gif import create_preview
from scripts.demo_reference_import import OUTPUT_NAMES, run_demo

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs/assets"


def test_demo_imports_previews_commits_and_renders_four_frames(tmp_path: Path) -> None:
    output = tmp_path / "demo"
    transcript = run_demo(output)
    assert {path.name for path in output.iterdir()} == set(OUTPUT_NAMES)
    checks = json.loads((output / "checks.json").read_text())
    assert checks["preview"] == {
        "records-read": 5,
        "records-skipped": 1,
        "duplicates-in-import": 1,
        "duplicates-in-collection": 0,
        "papers-to-add": 3,
    }
    assert checks["skipped_detail"] == [1, 2, 11, "missing_title"]
    assert checks["duplicate_detail"] == [2, 1, 1, 1, "doi"]
    assert (checks["documents_before_commit"], checks["collections_before_commit"]) == (0, 0)
    assert checks["commit"] == [303, "/explore/collection"] and checks["view_lists_members"]
    assert (checks["members"], checks["revision"], checks["documents_after_commit"]) == (3, 1, 3)
    assert checks["files"] == [["pubmed-search.ris", 3, 2, 0], ["zotero-library.json", 2, 1, 1]]
    assert checks["provenance"] == ["pubmed-search.ris", "1", "1", True]
    assert checks["reimport"] == [0, "nothing_new"]
    assert checks["reimport_reasons"] == {"doi": 3, "pmid": 1}
    assert checks["previously_imported"] == [True, True]
    assert checks["repeat_commit"] == [409, "reference_import_nothing_new"]
    assert checks["unchanged_after_repeat"] and checks["queue"] == [3, 0, 0]
    assert checks["labels"]["included"] == 1
    assert not any(checks["forbidden_call_counts"].values()) and checks["database_removed"]
    for name, size in checks["output_bytes"].items():
        content = (output / name).read_bytes()
        assert len(content) == size
        assert checks["sha256"][name] == hashlib.sha256(content).hexdigest()
    assert "EXPLICIT HUMAN WRITE / REFERENCE IMPORT" in (output / "preview.html").read_text()
    gif, still = tmp_path / "preview.gif", tmp_path / "preview.png"
    create_preview(output / "transcript.txt", gif, still)
    with Image.open(gif) as animation, Image.open(still) as preview:
        assert animation.n_frames == 4 and animation.info["loop"] == 0
        assert animation.size == preview.size == (1120, 540)
    with pytest.raises(FileExistsError):
        create_preview(output / "transcript.txt", gif, still)
    assert (ASSETS / "reference-import.txt").read_text() == transcript


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
