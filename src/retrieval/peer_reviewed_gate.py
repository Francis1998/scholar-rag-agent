"""Gate that keeps peer-reviewed results when required."""

from retrieval.models import SearchResult

_TRUTHY = frozenset({"true", "1", "yes", "peer_reviewed", "peer-reviewed"})


class PeerReviewedGate:
    """Keep results marked peer-reviewed in chunk metadata.

    When *require_peer_reviewed* is true (default), only results whose
    ``metadata["peer_reviewed"]`` is truthy are kept.  When false, all
    results pass through with provenance rewritten.

    Inspired by scholarly RAG stacks that prefer peer-reviewed sources
    over preprints when summarizing evidence.
    Inputs are not mutated.  Local postprocessor for GPT-5.5 /
    Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 pipelines (not a DOI
    connector).
    """

    def __init__(self, require_peer_reviewed: bool = True) -> None:
        """Create a peer-reviewed gate.

        Args:
            require_peer_reviewed: When true, drop non-peer-reviewed hits.
        """
        if not isinstance(require_peer_reviewed, bool):
            raise ValueError("require_peer_reviewed must be a bool")
        self._require = require_peer_reviewed

    @staticmethod
    def _is_peer_reviewed(metadata: dict[str, str]) -> bool:
        raw = metadata.get("peer_reviewed", "")
        return str(raw).strip().lower() in _TRUTHY

    def gate(
        self,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Return peer-reviewed results (or all when not required)."""
        if not results:
            return []

        kept: list[SearchResult] = []
        limit = len(results) if top_k is None else min(top_k, len(results))
        if limit <= 0:
            return []

        for r in results:
            if self._require and not self._is_peer_reviewed(r.chunk.metadata):
                continue
            kept.append(
                SearchResult(
                    chunk=r.chunk,
                    score=r.score,
                    retriever="peer_reviewed_gate",
                    path=[*r.path, r.retriever],
                )
            )
            if len(kept) >= limit:
                break
        return kept
