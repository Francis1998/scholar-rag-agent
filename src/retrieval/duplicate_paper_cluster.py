"""Local candidate duplicate grouping by DOI, document ID, or fuzzy title.

These heuristic groups are not proof of scientific work identity. Distinct from
:class:`~retrieval.near_duplicate_collapse.NearDuplicateCollapser` and
:class:`~retrieval.paraphrase_collapse.ParaphraseCollapser`, which collapse
chunk-level retrieval hits. Clustering does not call a model or DOI service.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher

from retrieval.models import Chunk, Document, SearchResult

_DOI_FIELDS = ("doi", "paper_doi", "work_doi")
_WHITESPACE = re.compile(r"\s+")


def _normalize_doi(value: str) -> str:
    candidate = value.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if candidate.startswith(prefix):
            candidate = candidate[len(prefix) :]
            break
    return candidate.strip().strip(".")


def _normalize_title(title: str) -> str:
    return _WHITESPACE.sub(" ", title.strip().lower())


@dataclass(frozen=True)
class PaperRef:
    """One paper reference used for candidate duplicate grouping."""

    document_id: str
    title: str
    doi: str = ""
    source: str = ""


@dataclass(frozen=True)
class PaperCluster:
    """One candidate group linked by local duplicate heuristics."""

    cluster_id: str
    papers: tuple[PaperRef, ...]
    match_reasons: tuple[str, ...]

    @property
    def size(self) -> int:
        """Return the number of papers in this cluster."""
        return len(self.papers)


class DuplicatePaperClusterer:
    """Group candidate duplicates by normalized DOI, exact ID, or fuzzy title.

    Nonblank ``document_id`` values match only identical raw strings; case and
    surrounding whitespace are significant. Empty or whitespace-only IDs never
    create an ID match. Normalized DOI matches and titles that reach
    ``title_threshold`` via :class:`difflib.SequenceMatcher` also merge, even
    with distinct IDs. Uses union-find so transitive matches share one candidate
    cluster. Paper-level only; does not collapse retrieval hits or prove work
    identity.
    """

    def __init__(self, title_threshold: float = 0.92) -> None:
        """Create a paper-level duplicate clusterer.

        Args:
            title_threshold: Inclusive ``SequenceMatcher`` ratio in
                ``[0.0, 1.0]`` for fuzzy title matches. Defaults to ``0.92``.

        Raises:
            ValueError: If ``title_threshold`` is non-finite or outside
                ``[0.0, 1.0]``.
        """
        if not math.isfinite(title_threshold) or not 0.0 <= title_threshold <= 1.0:
            raise ValueError("title_threshold must be a finite number within [0.0, 1.0]")
        self._title_threshold = title_threshold

    def cluster(
        self,
        papers: Sequence[PaperRef | Document | Chunk | SearchResult],
    ) -> list[PaperCluster]:
        """Return candidate duplicate clusters sorted by descending size then id.

        Singleton papers are included so callers can inspect the full
        partitioning. Inputs are not mutated.
        """
        refs = [self._to_ref(item) for item in papers]
        refs = [ref for ref in refs if ref.document_id.strip() or ref.title.strip() or ref.doi]
        if not refs:
            return []

        parent = list(range(len(refs)))
        reasons: dict[tuple[int, int], str] = {}

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int, reason: str) -> None:
            root_left = find(left)
            root_right = find(right)
            if root_left == root_right:
                return
            if root_left < root_right:
                parent[root_right] = root_left
            else:
                parent[root_left] = root_right
            key = (min(left, right), max(left, right))
            reasons[key] = reason

        doi_index: dict[str, int] = {}
        doc_index: dict[str, int] = {}
        for index, ref in enumerate(refs):
            doi = _normalize_doi(ref.doi) if ref.doi else ""
            if doi:
                prior = doi_index.get(doi)
                if prior is None:
                    doi_index[doi] = index
                else:
                    union(prior, index, "doi")
            doc_id = ref.document_id
            if doc_id.strip():
                prior = doc_index.get(doc_id)
                if prior is None:
                    doc_index[doc_id] = index
                else:
                    union(prior, index, "document_id")

        titles = [_normalize_title(ref.title) for ref in refs]
        for left in range(len(refs)):
            if not titles[left]:
                continue
            for right in range(left + 1, len(refs)):
                if not titles[right]:
                    continue
                if find(left) == find(right):
                    continue
                ratio = SequenceMatcher(None, titles[left], titles[right]).ratio()
                if ratio >= self._title_threshold:
                    union(left, right, f"title:{ratio:.3f}")

        buckets: dict[int, list[int]] = {}
        for index in range(len(refs)):
            buckets.setdefault(find(index), []).append(index)

        clusters: list[PaperCluster] = []
        for root, members in sorted(buckets.items(), key=lambda item: (item[0],)):
            member_refs = tuple(refs[index] for index in members)
            match_reasons = tuple(
                sorted(
                    {
                        reason
                        for (left, right), reason in reasons.items()
                        if find(left) == root and find(right) == root
                    }
                )
            )
            if not match_reasons and len(members) == 1:
                match_reasons = ("singleton",)
            cluster_id = f"cluster-{root:04d}"
            clusters.append(
                PaperCluster(
                    cluster_id=cluster_id,
                    papers=member_refs,
                    match_reasons=match_reasons,
                )
            )

        clusters.sort(key=lambda cluster: (-cluster.size, cluster.cluster_id))
        return clusters

    def duplicate_clusters(
        self,
        papers: Sequence[PaperRef | Document | Chunk | SearchResult],
    ) -> list[PaperCluster]:
        """Return only clusters with two or more papers."""
        return [cluster for cluster in self.cluster(papers) if cluster.size >= 2]

    @staticmethod
    def _to_ref(item: PaperRef | Document | Chunk | SearchResult) -> PaperRef:
        if isinstance(item, PaperRef):
            return item
        if isinstance(item, SearchResult):
            chunk = item.chunk
            doi = ""
            for field in _DOI_FIELDS:
                value = chunk.metadata.get(field, "").strip()
                if value:
                    doi = value
                    break
            return PaperRef(
                document_id=chunk.document_id,
                title=chunk.title,
                doi=doi,
                source=chunk.source,
            )
        if isinstance(item, Chunk):
            doi = ""
            for field in _DOI_FIELDS:
                value = item.metadata.get(field, "").strip()
                if value:
                    doi = value
                    break
            return PaperRef(
                document_id=item.document_id,
                title=item.title,
                doi=doi,
                source=item.source,
            )
        doi = ""
        for field in _DOI_FIELDS:
            value = item.metadata.get(field, "").strip()
            if value:
                doi = value
                break
        return PaperRef(
            document_id=item.document_id,
            title=item.title,
            doi=doi,
            source=item.source,
        )
