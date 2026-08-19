"""Expand child-chunk retrieval hits to in-memory parent documents."""

from collections.abc import Mapping

from retrieval.models import Chunk, Document, SearchResult

ParentValue = Document | Chunk | str


class ParentDocumentExpander:
    """Replace child chunk hits with deduplicated full parent document text."""

    def __init__(self, parents: Mapping[str, ParentValue]) -> None:
        """Create an expander from a parent map keyed by child ``document_id``.

        Parent values may be normalized ``Document`` objects, parent ``Chunk``
        objects, or raw text strings. The mapping is copied at construction so
        later caller mutations do not change expansion behavior.

        Raises:
            TypeError: If a parent value has an unsupported type.
        """
        for document_id, parent in parents.items():
            if not isinstance(parent, (Document, Chunk, str)):
                raise TypeError(f"parent {document_id!r} must be a Document, Chunk, or string")
        self._parents = dict(parents)

    def expand(self, results: list[SearchResult], top_k: int | None = None) -> list[SearchResult]:
        """Return ranked parent results for child hits with available parents.

        Multiple child hits for one parent collapse to the highest-scoring hit.
        Missing parent ids are skipped. Parent score ties retain the order in
        which their first child appeared.
        """
        limit = len(results) if top_k is None else min(top_k, len(results))
        if not results or limit <= 0:
            return []

        best_hits: dict[str, SearchResult] = {}
        for result in results:
            document_id = result.chunk.document_id
            if document_id not in self._parents:
                continue
            current = best_hits.get(document_id)
            if current is None or result.score > current.score:
                best_hits[document_id] = result

        expanded = [
            SearchResult(
                chunk=self._parent_chunk(document_id, hit),
                score=hit.score,
                retriever="parent_document",
                path=[*hit.path, hit.retriever],
            )
            for document_id, hit in best_hits.items()
        ]
        return sorted(expanded, key=lambda result: result.score, reverse=True)[:limit]

    def _parent_chunk(self, document_id: str, child_hit: SearchResult) -> Chunk:
        parent = self._parents[document_id]
        if isinstance(parent, Document):
            return Chunk(
                chunk_id=f"parent:{parent.document_id}",
                document_id=parent.document_id,
                title=parent.title,
                text=parent.text,
                source=parent.source,
                metadata=dict(parent.metadata),
            )
        if isinstance(parent, Chunk):
            return parent.model_copy(deep=True)
        return Chunk(
            chunk_id=f"parent:{document_id}",
            document_id=document_id,
            title=child_hit.chunk.title,
            text=parent,
            source=child_hit.chunk.source,
            metadata=dict(child_hit.chunk.metadata),
        )
