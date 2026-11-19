"""Guard user-facing prose against retired model-stack boilerplate.

Guides, module docstrings, and rendered markdown footers used to advertise
"GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2" long after providers moved
to GPT-6, Claude 5.5, Gemini 4, and Kimi K3.1. Superseded models stay
available but may only be named beside their successors: Claude Fable 5.2
replaced Fable 5.1 on 2026-11-09, and on 2026-11-17 GPT-6.2 Astra replaced
GPT-6 Astra and GPT-6.2 Sol replaced GPT-6.1 Sol. For GPT-6 Astra only the API
ID is guarded, which lineups, defaults, and rollbacks use; its prose name
remains in helper docs that list it among compatible models. A line that
enumerates Gemini 4 Carbon and Hydrogen must also name Neon (2026-11-18).
Historical records (CHANGELOG, daily improvement and PR logs) are exempt;
tests are not scanned.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

RETIRED_PROSE = re.compile(
    r"\bGPT-5\.5\b|\bGPT-4o\b|\bClaude\s+(?:Sonnet|Opus)\s+4\.\d\b"
    r"|\bSonnet\s+4\.6\b|\bKimi\s+K2\b(?!\.\d)|\bGemini\s+2\.5\b"
)
# A line matching the first pattern must also match the second.
REQUIRED_COMPANIONS = {
    "claude-fable-5-1": (
        re.compile(r"\bFable\s+5\.1\b|\bclaude-fable-5-1\b"),
        re.compile(r"\bFable\s+5\.2\b|\bclaude-fable-5-2\b"),
    ),
    "gpt-6-astra": (
        re.compile(r"\bgpt-6-astra\b"),
        re.compile(r"\bgpt-6\.2-astra\b"),
    ),
    "gpt-6.1-sol": (
        re.compile(r"\b6\.1\s+Sol\b|\bgpt-6\.1-sol\b"),
        re.compile(r"\b6\.2\s+Sol\b|\bgpt-6\.2-sol\b"),
    ),
    "gemini-4-lineup": (
        re.compile(
            r"(?:\bCarbon\b|gemini-4-carbon).*(?:\bHydrogen\b|gemini-4-hydrogen)"
            r"|(?:\bHydrogen\b|gemini-4-hydrogen).*(?:\bCarbon\b|gemini-4-carbon)"
        ),
        re.compile(r"\bNeon\b|\bgemini-4-neon\b"),
    ),
}

EXEMPT = {
    Path("CHANGELOG.md"),
    Path("DAILY_IMPROVEMENTS.md"),
    Path("docs/PULL_SHARK_LOG.md"),
}


def _prose_files() -> list[Path]:
    """Return the user-facing docs and non-test sources to scan."""
    files = [*ROOT.glob("*.md"), *ROOT.glob("docs/**/*.md")]
    files += [*ROOT.glob("src/**/*.py"), *ROOT.glob("scripts/**/*.py")]
    return sorted(p for p in files if p.relative_to(ROOT) not in EXEMPT)


def test_scan_covers_guides_and_sources() -> None:
    """The guard must actually see the guide catalog and retrieval sources."""
    rels = {p.relative_to(ROOT).as_posix() for p in _prose_files()}

    assert "README.md" in rels
    assert "src/retrieval/related_works.py" in rels
    assert sum(r.startswith("docs/guides/") for r in rels) > 100


def test_no_retired_model_stack_in_prose() -> None:
    """Docs, docstrings, and rendered footers name the current model stack."""
    offenders = [
        f"{path.relative_to(ROOT)}:{lineno}: {line.strip()[:120]}"
        for path in _prose_files()
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if RETIRED_PROSE.search(line)
    ]

    assert offenders == [], "retired model names in prose:\n" + "\n".join(offenders[:40])


@pytest.mark.parametrize("rule", REQUIRED_COMPANIONS)
def test_lineup_prose_names_the_current_models(rule: str) -> None:
    """Lineups must not present superseded models as current or omit Gemini 4 Neon."""
    stale, current = REQUIRED_COMPANIONS[rule]
    offenders = [
        f"{path.relative_to(ROOT)}:{lineno}: {line.strip()[:120]}"
        for path in _prose_files()
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if stale.search(line) and not current.search(line)
    ]

    assert offenders == [], f"stale {rule} prose:\n" + "\n".join(offenders[:40])
