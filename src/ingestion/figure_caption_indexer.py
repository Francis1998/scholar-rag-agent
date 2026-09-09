"""Index PDF-like figure and table captions from plain text blocks.

Extracts caption lines that match scholarly PDF patterns such as
``Figure 1: ...`` / ``Fig. 2. ...`` and ``Table 3: ...``. Fills a
PaperQA / Unstructured figure-extraction gap for RAG with a local,
deterministic caption indexer (no LLM or layout model). Indexed captions
can feed GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 grounding.
Distinct from PDF OCR hooks and live DOI connectors.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_CAPTION_RE = re.compile(
    r"(?P<label>Figure|Fig\.|Table)\s+(?P<number>\d+)\s*(?P<sep>[:.\-\u2013\u2014])\s*(?P<caption>\S.*)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class FigureCaption:
    """One extracted figure or table caption with source offset."""

    kind: str
    number: int
    caption: str
    char_offset: int


class FigureCaptionIndexer:
    """Extract figure/table captions from PDF-like text blocks.

    Scans line-oriented text for caption markers and returns stable
    :class:`FigureCaption` records ordered by appearance. Offline indexer for
    GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 RAG pipelines —
    PaperQA/Unstructured figure-extraction gap.
    """

    def extract(self, text: str) -> list[FigureCaption]:
        """Return figure/table captions found in ``text``.

        Empty or blank input yields an empty list. Matching is deterministic:
        same text always produces the same ordered list of
        ``(kind, number, caption, char_offset)`` records. ``kind`` is
        ``\"figure\"`` or ``\"table\"`` (lowercase).
        """
        if not text or not text.strip():
            return []

        results: list[FigureCaption] = []
        # Walk line by line while tracking absolute character offsets so
        # callers can highlight the caption span in the source document.
        offset = 0
        for line in text.splitlines(keepends=True):
            stripped = line.strip()
            match = _CAPTION_RE.match(stripped) if stripped else None
            if match:
                label = match.group("label").lower()
                kind = "table" if label.startswith("table") else "figure"
                # Offset of the match within the original line (after leading ws).
                leading = len(line) - len(line.lstrip(" \t"))
                caption_text = match.group("caption").strip()
                # Drop trailing keepends characters from caption payload.
                caption_text = caption_text.rstrip("\r\n").strip()
                results.append(
                    FigureCaption(
                        kind=kind,
                        number=int(match.group("number")),
                        caption=caption_text,
                        char_offset=offset + leading,
                    )
                )
            offset += len(line)
        return results
