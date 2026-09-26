"""Offline navigation checks for the README, catalog, and existing guide assets."""

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "docs/README.md"
INLINE_LINK_PATTERN = re.compile(r"(?<!!)\[[^\]\n]*\]\(([^)\s]+)\)")


def markdown_prose(text: str) -> str:
    """Exclude code examples and comments from navigable Markdown."""
    text = re.sub(r"```.*?```|~~~.*?~~~|<!--.*?-->", "", text, flags=re.DOTALL)
    return re.sub(r"(`+).*?\1", "", text, flags=re.DOTALL)


def local_target(document: Path, target: str) -> Path | None:
    url = urlsplit(target)
    if url.scheme or url.netloc:
        return None
    return (document.parent / unquote(url.path)).resolve() if url.path else document


@pytest.mark.parametrize("document", ["README.md", "docs/README.md"])
def test_entrypoint_starts_with_project_heading(document: str) -> None:
    text = (ROOT / document).read_text(encoding="utf-8")
    assert text.lstrip().startswith("# "), f"{document} must start with its H1, not announcements"


def test_catalog_links_every_existing_guide() -> None:
    text = markdown_prose(CATALOG.read_text(encoding="utf-8"))
    targets = INLINE_LINK_PATTERN.findall(text)
    linked = {local_target(CATALOG, target) for target in targets}
    guides = set((ROOT / "docs/guides").glob("*.md"))
    assert guides
    missing = sorted(str(guide.relative_to(ROOT)) for guide in guides - linked)
    assert not missing, "Add clickable relative catalog links for:\n" + "\n".join(missing)


def test_local_markdown_links_and_assets_exist() -> None:
    documents = sorted(ROOT.glob("*.md")) + sorted((ROOT / "docs").rglob("*.md"))
    missing: list[str] = []
    for document in documents:
        text = markdown_prose(document.read_text(encoding="utf-8"))
        # Match destinations separately so nested badge links check both image and link.
        for target in re.findall(r"\]\(([^)\s]+)\)", text):
            path = local_target(document, target)
            if path is not None and (not path.is_relative_to(ROOT) or not path.exists()):
                missing.append(f"{document.relative_to(ROOT)} -> {target}")
    assert not missing, "Broken local links or assets:\n" + "\n".join(missing)


def test_catalog_discovery_only_counts_clickable_links() -> None:
    text = markdown_prose(
        """
`[Inline example](guides/INLINE.md)`
```markdown
[Fenced example](guides/FENCED.md)
```
~~~markdown
[Another fenced example](guides/TILDE.md)
~~~
<!-- [Hidden link](guides/HIDDEN.md) -->
![Illustration, not a guide link](guides/IMAGE.md)
[Readable guide](guides/READABLE.md)
[`Code-formatted label`](guides/CODE_LABEL.md)
"""
    )
    assert INLINE_LINK_PATTERN.findall(text) == ["guides/READABLE.md", "guides/CODE_LABEL.md"]


@pytest.mark.parametrize("target", ["https://example.com/guide", "//example.com/image.gif"])
def test_external_targets_are_not_local_files(target: str) -> None:
    assert local_target(CATALOG, target) is None


def test_local_targets_preserve_relative_paths_without_query_or_fragment() -> None:
    assert local_target(CATALOG, "guides/A%20B.md?raw=1#usage") == ROOT / "docs/guides/A B.md"
    assert local_target(CATALOG, "#start-here") == CATALOG
