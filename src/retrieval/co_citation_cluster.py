"""Cluster papers that share citation neighbors above a threshold.

Given papers with citation-neighbor ID sets, groups papers whose shared
neighbor overlap meets ``min_shared`` and optional Jaccard thresholds.
Offline only — distinct from :class:`~retrieval.citation_graph.CitationGraphIndex`
(directed expand along citing/cited edges) and
:class:`~retrieval.contradiction_cluster.ContradictionClusterFinder`
(claim polarity clusters). Fills a ResearchRabbit / Semantic Scholar
co-citation browsing gap without an LLM or network call. Optional later
narrative can use GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

_SPLIT = re.compile(r"[,;|\s]+")


@dataclass(frozen=True)
class CoCitationCluster:
    """One cluster of papers linked by shared citation neighbors."""

    cluster_id: str
    paper_ids: tuple[str, ...]
    shared_neighbor_count: int
    reasons: tuple[str, ...]


class CoCitationClusterFinder:
    """Cluster papers that co-cite overlapping neighbor sets.

    Offline co-citation clusters for GPT-5.5 / Claude Sonnet 4.6 /
    Gemini 3.x / Kimi K2 pipelines — ResearchRabbit/Semantic Scholar
    co-citation gap. Distinct from
    :class:`~retrieval.citation_graph.CitationGraphIndex` (edge expansion)
    and :class:`~retrieval.contradiction_cluster.ContradictionClusterFinder`
    (supporting vs contradicting claim clusters). Never mutates inputs and
    never calls the network.
    """

    def __init__(
        self,
        *,
        min_shared: int = 2,
        min_jaccard: float = 0.0,
    ) -> None:
        """Create a co-citation cluster finder.

        Args:
            min_shared: Minimum shared neighbor count required to link two
                papers (must be ``>= 1``). Defaults to ``2``.
            min_jaccard: Inclusive Jaccard threshold on neighbor sets in
                ``[0.0, 1.0]``. Defaults to ``0.0`` (shared-count only).

        Raises:
            ValueError: If ``min_shared`` or ``min_jaccard`` is invalid.
        """
        if not isinstance(min_shared, int) or min_shared < 1:
            raise ValueError("min_shared must be an integer >= 1")
        if not math.isfinite(min_jaccard) or not 0.0 <= min_jaccard <= 1.0:
            raise ValueError("min_jaccard must be a finite number within [0.0, 1.0]")
        self._min_shared = min_shared
        self._min_jaccard = min_jaccard

    def find(
        self,
        papers: Sequence[Mapping[str, object]],
    ) -> tuple[CoCitationCluster, ...]:
        """Return co-citation clusters for ``papers``.

        Each mapping should provide ``paper_id`` / ``id`` and a neighbor
        collection under ``neighbors`` or ``citation_neighbors`` (set, list,
        tuple, or comma-separated string). Multi-paper clusters are returned
        first (descending size, then id); singletons follow. Empty input
        yields an empty tuple. Inputs are not mutated.
        """
        if not papers:
            return ()

        parsed: list[tuple[str, frozenset[str]]] = []
        for index, paper in enumerate(papers):
            paper_id = self._resolve_id(paper, index)
            neighbors = self._neighbors(paper)
            parsed.append((paper_id, neighbors))

        parent = list(range(len(parsed)))

        def find_root(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i: int, j: int) -> None:
            ri, rj = find_root(i), find_root(j)
            if ri != rj:
                parent[rj] = ri

        link_shared: dict[tuple[int, int], int] = {}
        for i in range(len(parsed)):
            for j in range(i + 1, len(parsed)):
                shared = parsed[i][1] & parsed[j][1]
                shared_count = len(shared)
                if shared_count < self._min_shared:
                    continue
                union_size = len(parsed[i][1] | parsed[j][1])
                jaccard = shared_count / union_size if union_size else 0.0
                if jaccard < self._min_jaccard:
                    continue
                union(i, j)
                link_shared[(i, j)] = shared_count

        buckets: dict[int, list[int]] = {}
        for i in range(len(parsed)):
            buckets.setdefault(find_root(i), []).append(i)

        clusters: list[CoCitationCluster] = []
        for _root, members in buckets.items():
            member_ids = tuple(parsed[i][0] for i in members)
            if len(members) == 1:
                clusters.append(
                    CoCitationCluster(
                        cluster_id=f"co-cite-{parsed[members[0]][0]}",
                        paper_ids=member_ids,
                        shared_neighbor_count=0,
                        reasons=("Singleton — below co-citation link threshold.",),
                    )
                )
                continue
            max_shared = 0
            for i_idx, i in enumerate(members):
                for j in members[i_idx + 1 :]:
                    a, b = (i, j) if i < j else (j, i)
                    max_shared = max(max_shared, link_shared.get((a, b), 0))
            if max_shared == 0:
                for i_idx, i in enumerate(members):
                    for j in members[i_idx + 1 :]:
                        max_shared = max(max_shared, len(parsed[i][1] & parsed[j][1]))
            clusters.append(
                CoCitationCluster(
                    cluster_id=f"co-cite-{'-'.join(sorted(member_ids))}",
                    paper_ids=tuple(sorted(member_ids)),
                    shared_neighbor_count=max_shared,
                    reasons=(
                        f"Shared >= {self._min_shared} citation neighbors "
                        f"(max pairwise shared={max_shared}).",
                    ),
                )
            )

        clusters.sort(key=lambda c: (-len(c.paper_ids), c.cluster_id))
        return tuple(clusters)

    @staticmethod
    def _resolve_id(paper: Mapping[str, object], index: int) -> str:
        for key in ("paper_id", "id", "doi", "document_id"):
            value = paper.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return f"paper-{index}"

    @staticmethod
    def _neighbors(paper: Mapping[str, object]) -> frozenset[str]:
        raw = paper.get("neighbors")
        if raw is None:
            raw = paper.get("citation_neighbors")
        if raw is None:
            return frozenset()
        if isinstance(raw, str):
            return frozenset(p for p in _SPLIT.split(raw) if p)
        if isinstance(raw, (set, frozenset, list, tuple)):
            out: set[str] = set()
            for item in raw:
                text = str(item).strip()
                if text:
                    out.add(text)
            return frozenset(out)
        return frozenset()
