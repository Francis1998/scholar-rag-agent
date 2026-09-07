"""Deterministic literature-review outlines from ranked retrieval results.

Fills a PaperQA-style review synthesis gap: PaperQA can gather paper evidence,
but this module builds a structured outline of sections from ranked
``SearchResult`` / ``Document`` lists without calling an LLM. Optional later
narrative drafting can use GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.
Distinct from retrieval gates and from live DOI connectors.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from retrieval.models import Document, SearchResult

_SECTION_DEFS: tuple[tuple[str, frozenset[str], str], ...] = (
    (
        "Background & Motivation",
        frozenset({"background", "introduction", "intro", "motivation", "related"}),
        "Foundational context and motivating prior work.",
    ),
    (
        "Methods & Approaches",
        frozenset({"methods", "method", "approach", "methodology", "experimental"}),
        "Study designs, algorithms, and experimental setups.",
    ),
    (
        "Key Findings",
        frozenset({"results", "findings", "result", "evaluation", "experiments"}),
        "Primary empirical or theoretical findings.",
    ),
    (
        "Limitations & Open Questions",
        frozenset({"limitations", "limitation", "discussion", "future", "open"}),
        "Gaps, caveats, and unresolved questions.",
    ),
    (
        "Synthesis Opportunities",
        frozenset({"conclusion", "conclusions", "summary", "synthesis"}),
        "Cross-paper themes suitable for review synthesis.",
    ),
)

_SECTION_META_FIELDS = ("section", "section_type", "paper_section")


@dataclass(frozen=True)
class OutlineItem:
    """One paper/document placed in an outline section."""

    document_id: str
    title: str
    score: float
    source: str


@dataclass(frozen=True)
class OutlineSection:
    """One literature-review outline section with assigned papers."""

    title: str
    note: str
    items: tuple[OutlineItem, ...]


@dataclass(frozen=True)
class LiteratureReviewOutline:
    """Complete deterministic literature-review outline."""

    topic: str
    sections: tuple[OutlineSection, ...]

    def to_markdown(self) -> str:
        """Render the outline as markdown suitable for review drafts."""
        lines = [f"# Literature Review Outline: {self.topic or 'Untitled'}"]
        lines.append("")
        lines.append(
            "_Deterministic outline from ranked evidence. Optional narrative "
            "synthesis can use GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / "
            "Kimi K2 (PaperQA-style review synthesis gap)._"
        )
        for section in self.sections:
            lines.append("")
            lines.append(f"## {section.title}")
            lines.append(section.note)
            if not section.items:
                lines.append("- _(no papers assigned)_")
                continue
            for item in section.items:
                score = f"{item.score:.3f}" if item.score else "n/a"
                lines.append(f"- [{item.document_id}] {item.title} (score={score})")
        return "\n".join(lines).rstrip() + "\n"


def _meta_section(metadata: dict[str, str]) -> str:
    for field in _SECTION_META_FIELDS:
        value = metadata.get(field, "").strip().lower()
        if value:
            return value
    return ""


def _match_section_index(section_label: str) -> int | None:
    token = section_label.replace("-", " ").replace("_", " ").strip().lower()
    parts = set(token.split())
    for index, (_title, keywords, _note) in enumerate(_SECTION_DEFS):
        if token in keywords or parts & keywords:
            return index
    return None


class LiteratureReviewOutliner:
    """Build deterministic literature-review outline sections.

    Papers are deduplicated by ``document_id`` (highest score wins). When
    ``section`` / ``section_type`` metadata matches a known heading, the paper
    is placed there; otherwise papers are distributed round-robin across
    sections in rank order. Inputs are not mutated. Outline-only module for
    GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 pipelines — does not
    generate narrative prose (PaperQA review synthesis gap).
    """

    def outline(
        self,
        sources: Sequence[SearchResult | Document],
        *,
        topic: str = "",
        top_k: int | None = None,
    ) -> LiteratureReviewOutline:
        """Return a literature-review outline from ranked sources.

        Args:
            sources: Ranked ``SearchResult`` hits and/or ``Document`` records.
            topic: Optional review topic heading.
            top_k: Optional cap on unique documents after dedupe.

        Raises:
            ValueError: If ``top_k`` is not ``None`` and ``<= 0``.
        """
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be > 0 when provided")

        best: dict[str, tuple[OutlineItem, str]] = {}
        order: list[str] = []
        for source in sources:
            if isinstance(source, SearchResult):
                document_id = source.chunk.document_id
                title = source.chunk.title
                score = source.score
                meta = source.chunk.metadata
                origin = source.chunk.source
            else:
                document_id = source.document_id
                title = source.title
                score = 0.0
                meta = source.metadata
                origin = source.source
            key = document_id.strip().lower()
            if not key:
                continue
            item = OutlineItem(
                document_id=document_id,
                title=title.strip() or document_id,
                score=score,
                source=origin,
            )
            label = _meta_section(dict(meta))
            current = best.get(key)
            if current is None:
                best[key] = (item, label)
                order.append(key)
            elif item.score > current[0].score:
                best[key] = (item, label)

        ranked_keys = sorted(order, key=lambda key: best[key][0].score, reverse=True)
        if top_k is not None:
            ranked_keys = ranked_keys[:top_k]

        buckets: list[list[OutlineItem]] = [[] for _ in _SECTION_DEFS]
        unassigned: list[OutlineItem] = []
        for key in ranked_keys:
            item, label = best[key]
            index = _match_section_index(label) if label else None
            if index is None:
                unassigned.append(item)
            else:
                buckets[index].append(item)

        for offset, item in enumerate(unassigned):
            buckets[offset % len(buckets)].append(item)

        sections = tuple(
            OutlineSection(title=title, note=note, items=tuple(items))
            for (title, _keywords, note), items in zip(_SECTION_DEFS, buckets, strict=True)
        )
        return LiteratureReviewOutline(topic=topic.strip(), sections=sections)
