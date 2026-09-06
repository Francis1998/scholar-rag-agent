"""Paper-level citation graph indexing and retrieval expansion.

Distinct from entity co-mention GraphRAG (:mod:`retrieval.graph`) and from
:class:`~retrieval.citation_count_boost.CitationCountBooster` (score-only).
Builds a local directed graph of papers from chunk metadata and expands seed
hits along citing / cited edges — inspired by Semantic Scholar citation graphs
and PaperQA-style paper expansion, without requiring a live API call.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict

from retrieval.models import Chunk, SearchResult

_SPLIT_PATTERN = re.compile(r"[,;|\s]+")
_DOI_FIELDS = ("doi", "paper_doi", "work_doi")
_OUT_FIELDS = ("references", "cites", "reference_dois", "cites_dois")
_IN_FIELDS = ("cited_by", "cited_by_dois", "citing_dois")


def _normalize_doi(value: str) -> str:
    """Normalize a DOI or paper id for graph keys."""
    candidate = value.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if candidate.startswith(prefix):
            candidate = candidate[len(prefix) :]
            break
    return candidate.strip().strip(".")


def _split_ids(raw: str) -> list[str]:
    """Split a metadata multi-id string into normalized identifiers."""
    if not raw.strip():
        return []
    seen: set[str] = set()
    out: list[str] = []
    for part in _SPLIT_PATTERN.split(raw):
        key = _normalize_doi(part)
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _paper_id(chunk: Chunk) -> str:
    """Return the preferred paper key for a chunk (DOI, else document_id)."""
    for field in _DOI_FIELDS:
        value = chunk.metadata.get(field, "").strip()
        if value:
            return _normalize_doi(value)
    return chunk.document_id.strip().lower()


class CitationGraphIndex:
    """In-memory directed citation graph over locally indexed chunks."""

    def __init__(self) -> None:
        """Create an empty citation graph index."""
        self._chunks_by_paper: dict[str, list[Chunk]] = defaultdict(list)
        self._cites: dict[str, set[str]] = defaultdict(set)
        self._cited_by: dict[str, set[str]] = defaultdict(set)

    def index_chunks(self, chunks: list[Chunk]) -> None:
        """Index chunks and directed citation edges from metadata.

        Outgoing edges are read from ``references`` / ``cites`` /
        ``reference_dois`` / ``cites_dois``. Incoming edges are read from
        ``cited_by`` / ``cited_by_dois`` / ``citing_dois``. Missing fields are
        ignored. Inputs are not mutated.
        """
        for chunk in chunks:
            paper = _paper_id(chunk)
            self._chunks_by_paper[paper].append(
                Chunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    title=chunk.title,
                    text=chunk.text,
                    source=chunk.source,
                    metadata=dict(chunk.metadata),
                )
            )
            for field in _OUT_FIELDS:
                for target in _split_ids(chunk.metadata.get(field, "")):
                    if target == paper:
                        continue
                    self._cites[paper].add(target)
                    self._cited_by[target].add(paper)
            for field in _IN_FIELDS:
                for source in _split_ids(chunk.metadata.get(field, "")):
                    if source == paper:
                        continue
                    self._cited_by[paper].add(source)
                    self._cites[source].add(paper)

    def neighbours(self, paper_id: str, direction: str = "both") -> set[str]:
        """Return one-hop neighbour paper ids for ``direction``."""
        key = _normalize_doi(paper_id)
        if direction == "cites":
            return set(self._cites.get(key, ()))
        if direction == "cited_by":
            return set(self._cited_by.get(key, ()))
        if direction != "both":
            raise ValueError("direction must be one of: both, cites, cited_by")
        return set(self._cites.get(key, ())) | set(self._cited_by.get(key, ()))

    def chunks_for_paper(self, paper_id: str) -> list[Chunk]:
        """Return indexed chunks for a paper id."""
        return list(self._chunks_by_paper.get(_normalize_doi(paper_id), ()))


class CitationGraphExpander:
    """Expand retrieval hits along local paper citation edges.

    Inspired by Semantic Scholar citation-graph expansion and PaperQA-style
    related-paper gathering. Operates entirely on an in-memory
    :class:`CitationGraphIndex` (no network). Distinct from entity multi-hop
    GraphRAG. Local expander for GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
    Kimi K2 pipelines (not a DOI connector).
    """

    def __init__(
        self,
        index: CitationGraphIndex,
        *,
        hop_decay: float = 0.5,
        direction: str = "both",
    ) -> None:
        """Create a citation-graph expander.

        Args:
            index: Populated citation graph index.
            hop_decay: Multiplicative score decay per hop (``(0, 1]``).
            direction: Edge direction to follow: ``both``, ``cites``, or
                ``cited_by``.

        Raises:
            ValueError: If ``hop_decay`` is non-finite or outside ``(0, 1]``,
                or if ``direction`` is invalid.
        """
        if not math.isfinite(hop_decay) or not 0.0 < hop_decay <= 1.0:
            raise ValueError("hop_decay must be a finite number in (0.0, 1.0]")
        if direction not in {"both", "cites", "cited_by"}:
            raise ValueError("direction must be one of: both, cites, cited_by")
        self._index = index
        self._hop_decay = hop_decay
        self._direction = direction

    def expand(
        self,
        results: list[SearchResult],
        *,
        max_hops: int = 1,
        top_k: int | None = None,
        include_seeds: bool = True,
    ) -> list[SearchResult]:
        """Expand seed hits along citation edges up to ``max_hops``.

        Seed scores are preserved when ``include_seeds`` is true. Each hop
        multiplies the prior paper score by ``hop_decay`` once (so hop 2 is
        ``seed * hop_decay**2``). Duplicate chunk ids keep the highest score.
        Inputs are not mutated.
        """
        if max_hops < 0:
            raise ValueError("max_hops must be >= 0")
        if top_k is not None and top_k <= 0:
            return []
        if not results:
            return []

        merged: dict[str, SearchResult] = {}
        seed_papers: dict[str, float] = {}

        for result in results:
            paper = _paper_id(result.chunk)
            seed_papers[paper] = max(seed_papers.get(paper, 0.0), result.score)
            if include_seeds:
                merged[result.chunk.chunk_id] = SearchResult(
                    chunk=result.chunk,
                    score=result.score,
                    retriever="citation_graph",
                    path=[*result.path, result.retriever],
                )

        frontier = dict(seed_papers)
        visited = set(frontier)
        for _hop in range(1, max_hops + 1):
            if not frontier:
                break
            next_frontier: dict[str, float] = {}
            for paper, paper_score in frontier.items():
                for neighbour in self._index.neighbours(paper, self._direction):
                    if neighbour in visited:
                        continue
                    neighbour_score = paper_score * self._hop_decay
                    next_frontier[neighbour] = max(
                        next_frontier.get(neighbour, 0.0), neighbour_score
                    )
                    for chunk in self._index.chunks_for_paper(neighbour):
                        existing = merged.get(chunk.chunk_id)
                        if existing is None or neighbour_score > existing.score:
                            merged[chunk.chunk_id] = SearchResult(
                                chunk=chunk,
                                score=neighbour_score,
                                retriever="citation_graph",
                                path=[paper, neighbour],
                            )
            visited.update(next_frontier)
            frontier = next_frontier

        ranked = sorted(merged.values(), key=lambda item: item.score, reverse=True)
        if top_k is None:
            return ranked
        return ranked[:top_k]
