"""Every GIF embedded in README/docs Markdown must be a real, viewable demo.

87 guide demos once shipped as 1x1-pixel (or truncated) placeholders, so readers
saw nothing where a demo was promised. Re-render a guide demo with
``python scripts/render_guide_demos.py --only <name>.gif``.
"""

from __future__ import annotations

import io
import re
import struct
from pathlib import Path

import pytest
from PIL import Image

from tests.test_documentation_navigation import local_target, markdown_prose

ROOT = Path(__file__).resolve().parents[1]
MIN_WIDTH, MIN_HEIGHT, MIN_BYTES = 200, 100, 2048
GIF_REFERENCE = re.compile(
    r"""(?:\]\(|\bsrc\s*=\s*["'])([^)\s"'<>]+\.gif)(?=[)\s"'#?])""", re.IGNORECASE
)


def referenced_gifs() -> dict[Path, set[str]]:
    """Map each local GIF referenced from README/docs prose to the files citing it."""
    documents = sorted(ROOT.glob("*.md")) + sorted((ROOT / "docs").rglob("*.md"))
    references: dict[Path, set[str]] = {}
    for document in documents:
        prose = markdown_prose(document.read_text(encoding="utf-8"))
        for target in GIF_REFERENCE.findall(prose):
            path = local_target(document, target)
            if path is not None:
                references.setdefault(path, set()).add(document.relative_to(ROOT).as_posix())
    return references


def gif_problem(path: Path) -> str | None:
    """Return why ``path`` is not a real demo GIF, or ``None`` when it is one."""
    if not path.is_file():
        return "missing"
    data = path.read_bytes()
    if len(data) < 10 or data[:6] not in (b"GIF87a", b"GIF89a"):
        return "not a valid GIF"
    width, height = struct.unpack("<HH", data[6:10])
    if width < MIN_WIDTH or height < MIN_HEIGHT:
        return f"{width}x{height} px, below {MIN_WIDTH}x{MIN_HEIGHT}"
    if len(data) < MIN_BYTES:
        return f"{len(data)} bytes, below {MIN_BYTES}"
    try:
        with Image.open(path) as image:
            image.load()
    except (OSError, SyntaxError) as error:
        return f"does not decode: {error}"
    return None


def test_scan_sees_readme_and_guide_demos() -> None:
    references = referenced_gifs()

    assert len(references) > 300
    assert any("README.md" in documents for documents in references.values())
    assert (
        "docs/guides/ABSTRACT_KEYWORD_BOOST_GUIDE.md"
        in references[ROOT / "docs/assets/abstract-keyword-boost.gif"]
    )


def test_every_referenced_gif_is_a_real_demo() -> None:
    problems = []
    for path, documents in sorted(referenced_gifs().items()):
        problem = gif_problem(path)
        if problem is not None:
            shown = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
            problems.append(f"{shown}: {problem} (in {', '.join(sorted(documents))})")

    assert not problems, (
        f"{len(problems)} referenced GIFs are placeholders or broken; re-render with "
        "scripts/render_guide_demos.py:\n" + "\n".join(problems)
    )


def _gif(width: int, height: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("P", (width, height)).save(buffer, format="GIF")
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (b"GIF", "not a valid GIF"),
        (b"\x89PNG\r\n\x1a\n" + b"\0" * 4096, "not a valid GIF"),
        (_gif(1, 1), "1x1 px, below 200x100"),
        (_gif(640, 40), "640x40 px, below 200x100"),
        (_gif(320, 200), "bytes, below 2048"),
    ],
    ids=["truncated", "png", "one-pixel", "too-short", "too-small-file"],
)
def test_gif_problem_flags_placeholders(tmp_path: Path, data: bytes, expected: str) -> None:
    path = tmp_path / "demo.gif"
    path.write_bytes(data)

    problem = gif_problem(path)

    assert problem is not None and expected in problem


def test_gif_problem_accepts_real_gifs_and_reports_missing_files(tmp_path: Path) -> None:
    assert gif_problem(tmp_path / "absent.gif") == "missing"
    assert gif_problem(ROOT / "docs/assets/demo.gif") is None
