"""Keep the README and documentation catalog navigable as sections are added.

Earlier padding runs appended 21 ``### <Name>CueExtractor`` sections after
"Quality gates" and "License", so they rendered as sub-sections of both, buried
the license, cited guides only as unclickable code paths, and embedded GIFs that
were byte-identical copies (blank frames or another extractor's title card).
The catalog likewise stranded half of its cue guides under "Historical project
records" in a table without a header row. New, well-formed sections may appear
anywhere before the closing sections; only the documented order is fixed.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from tests.test_documentation_navigation import local_target, markdown_prose

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
CATALOG = ROOT / "docs/README.md"

README_ORDER = (
    "Start offline",
    "Choose a workflow",
    "What actually runs",
    "Boundaries to understand",
    "Offline evidence extractors",
    "Documentation",
    "Quality gates",
    "License",
)
LEAF_SECTIONS = ("Quality gates", "License")
CUE_SECTION = "Clinical and statistical evidence cues"
HISTORY_SECTION = "Historical project records"
CUE_GUIDE = re.compile(r"guides/[A-Z0-9_]+_CUES?_EXTRACTOR_GUIDE\.md$")
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
LINK_TARGET = re.compile(r"\]\(([^)\s]+)\)")
TABLE_DELIMITER = re.compile(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")


def headings(text: str) -> list[tuple[int, str, int]]:
    """Return ``(level, title, line_index)`` for headings outside fenced code."""
    found: list[tuple[int, str, int]] = []
    fence: str | None = None
    for index, line in enumerate(text.splitlines()):
        marker = re.match(r"^\s*(```|~~~)", line)
        if marker:
            fence = None if fence == marker.group(1) else fence or marker.group(1)
            continue
        match = None if fence else HEADING.match(line)
        if match:
            found.append((len(match.group(1)), match.group(2), index))
    return found


def section_lines(text: str, title: str) -> list[str]:
    """Return the lines of the section headed ``title``, up to its next peer."""
    lines = text.splitlines()
    marks = headings(text)
    for position, (level, heading, start) in enumerate(marks):
        if heading == title:
            end = next((line for lvl, _, line in marks[position + 1 :] if lvl <= level), len(lines))
            return lines[start + 1 : end]
    raise AssertionError(f"missing section: {title}")


def subsections(text: str, title: str) -> list[str]:
    """Return the headings nested under the ``## title`` section, ignoring code."""
    marks = headings(text)
    start = next(i for i, (level, name, _) in enumerate(marks) if level == 2 and name == title)
    nested: list[str] = []
    for level, name, _ in marks[start + 1 :]:
        if level <= 2:
            break
        nested.append(f"{'#' * level} {name}")
    return nested


def slug(title: str) -> str:
    """GitHub's heading anchor: plain text, lowercased, punctuation dropped."""
    plain = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", title).replace("`", "")
    plain = re.sub(r"[^\w\- ]", "", plain.replace("*", "").strip().lower())
    return plain.replace(" ", "-")


def anchors(document: Path) -> set[str]:
    """Return every heading anchor GitHub generates for ``document``."""
    seen: dict[str, int] = {}
    result: set[str] = set()
    for _, title, _ in headings(document.read_text(encoding="utf-8")):
        base = slug(title)
        count = seen.get(base, 0)
        seen[base] = count + 1
        result.add(base if count == 0 else f"{base}-{count}")
    return result


def h2_titles(text: str) -> list[str]:
    return [title for level, title, _ in headings(text) if level == 2]


def test_readme_keeps_documented_top_level_order_with_license_last() -> None:
    titles = h2_titles(README.read_text(encoding="utf-8"))

    missing = [title for title in README_ORDER if titles.count(title) != 1]
    assert not missing, f"README needs exactly one '## ' section for each of: {missing}"
    positions = [titles.index(title) for title in README_ORDER]
    assert positions == sorted(positions), f"README sections out of order: {titles}"
    assert titles[-1] == "License", f"'## License' must be last, not {titles[-1]!r}"


@pytest.mark.parametrize("title", LEAF_SECTIONS)
def test_readme_closing_sections_have_no_subsections(title: str) -> None:
    nested = subsections(README.read_text(encoding="utf-8"), title)

    assert not nested, f"'## {title}' must not contain sub-sections:\n" + "\n".join(nested)


def test_readme_links_guides_instead_of_quoting_their_paths() -> None:
    text = re.sub(r"```.*?```|~~~.*?~~~", "", README.read_text(encoding="utf-8"), flags=re.S)
    quoted = sorted(set(re.findall(r"`([^`\s]+\.md)`", text)))

    assert not quoted, "Link these README guide paths instead of quoting them:\n" + "\n".join(
        quoted
    )


def test_readme_anchor_links_resolve_to_headings() -> None:
    prose = markdown_prose(README.read_text(encoding="utf-8"))
    checked, broken = 0, []
    for target in LINK_TARGET.findall(prose):
        path, _, fragment = target.partition("#")
        document = local_target(README, path) if path else README
        if fragment and document is not None and document.suffix == ".md":
            checked += 1
            if not document.is_file() or fragment not in anchors(document):
                broken.append(target)

    assert checked >= 5
    assert not broken, "README anchors without a matching heading:\n" + "\n".join(broken)


def test_readme_gifs_are_distinct_demos_not_copies() -> None:
    prose = markdown_prose(README.read_text(encoding="utf-8"))
    by_digest: dict[str, set[str]] = {}
    for target in LINK_TARGET.findall(prose):
        path = local_target(README, target)
        if path is not None and path.suffix.lower() == ".gif" and path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            by_digest.setdefault(digest, set()).add(path.relative_to(ROOT).as_posix())
    copies = [sorted(paths) for paths in by_digest.values() if len(paths) > 1]

    assert by_digest
    assert not copies, "README links byte-identical GIFs under different names:\n" + "\n".join(
        ", ".join(paths) for paths in copies
    )


def test_catalog_files_every_cue_guide_under_the_cue_section() -> None:
    text = markdown_prose(CATALOG.read_text(encoding="utf-8"))
    cue_section = "\n".join(section_lines(text, CUE_SECTION))
    filed = {local_target(CATALOG, target) for target in LINK_TARGET.findall(cue_section)}
    linked = {
        local_target(CATALOG, target)
        for target in LINK_TARGET.findall(text)
        if CUE_GUIDE.search(target)
    }
    stray = sorted(str(path.relative_to(ROOT)) for path in linked - filed if path is not None)

    assert len(linked) > 100
    assert not stray, f"File these cue guides under '### {CUE_SECTION}':\n" + "\n".join(stray)


def test_catalog_ends_with_historical_records_without_guides() -> None:
    text = CATALOG.read_text(encoding="utf-8")
    history = "\n".join(section_lines(text, HISTORY_SECTION))
    guides = [target for target in LINK_TARGET.findall(history) if "guides/" in target]

    assert h2_titles(text)[-1] == HISTORY_SECTION
    assert not guides, f"'## {HISTORY_SECTION}' must not list guides:\n" + "\n".join(guides)


@pytest.mark.parametrize("document", [README, CATALOG], ids=["README", "catalog"])
def test_markdown_tables_start_with_a_header_row(document: Path) -> None:
    lines = re.sub(
        r"```.*?```|~~~.*?~~~",
        lambda block: "\n" * block.group(0).count("\n"),
        document.read_text(encoding="utf-8"),
        flags=re.S,
    ).splitlines()
    headless = [
        f"line {index + 1}: {line[:80]}"
        for index, line in enumerate(lines)
        if line.startswith("|")
        and (index == 0 or not lines[index - 1].startswith("|"))
        and not (index + 1 < len(lines) and TABLE_DELIMITER.match(lines[index + 1]))
    ]

    assert not headless, "Tables need a header and delimiter row:\n" + "\n".join(headless)


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Model lineup (verified 2026-11-19)", "model-lineup-verified-2026-11-19"),
        ("6. Present a portfolio demonstration", "6-present-a-portfolio-demonstration"),
        ("`/query` and [the API](API.md)", "query-and-the-api"),
    ],
)
def test_slug_matches_github_heading_anchors(title: str, expected: str) -> None:
    assert slug(title) == expected


def test_heading_scan_ignores_fenced_code() -> None:
    text = "# Title\n```bash\n# not a heading\n```\n## Real\n~~~\n## Fake\n~~~\n### Nested"

    assert [(level, title) for level, title, _ in headings(text)] == [
        (1, "Title"),
        (2, "Real"),
        (3, "Nested"),
    ]
    assert section_lines(text, "Real")[-1] == "### Nested"
    assert subsections(text, "Real") == ["### Nested"]
