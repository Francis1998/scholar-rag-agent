"""GraphRAG entity extraction and graph retrieval."""

import re
from itertools import combinations

from retrieval.models import Chunk, Entity, EntityEdge, SearchResult
from storage.graph_store import SQLiteGraphStore

_TERM_PATTERN = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9\-]{2,}|[a-z]+(?:ase|tion|ology|omics|graph|model|agent))\b"
)


class SpacyEntityExtractor:
    """Extract named entities with spaCy when installed and a deterministic fallback otherwise."""

    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        """Create an extractor and lazily load spaCy when available."""
        self._nlp = None
        try:
            import spacy

            self._nlp = spacy.load(model_name)
        except Exception:
            self._nlp = None

    def extract(self, text: str) -> list[Entity]:
        """Extract entities from text."""
        if self._nlp is not None:
            document = self._nlp(text)
            return [Entity(name=entity.text, label=entity.label_) for entity in document.ents]
        seen: set[str] = set()
        entities: list[Entity] = []
        for match in _TERM_PATTERN.finditer(text):
            name = match.group(0).strip()
            if name.lower() not in seen:
                seen.add(name.lower())
                entities.append(Entity(name=name, label="FALLBACK_TERM"))
        return entities


class GraphRAGBuilder:
    """Build an entity relationship graph over ingested chunks."""

    def __init__(
        self, graph_store: SQLiteGraphStore, extractor: SpacyEntityExtractor | None = None
    ) -> None:
        """Create a graph builder."""
        self._graph_store = graph_store
        self._extractor = extractor or SpacyEntityExtractor()

    def index_chunks(self, chunks: list[Chunk]) -> None:
        """Extract entities and co-mention edges for chunks."""
        for chunk in chunks:
            entities = self._extractor.extract(chunk.text)
            normalized_names = sorted({entity.name for entity in entities})
            self._graph_store.add_mentions(chunk, entities)
            edges = [
                EntityEdge(source=left, target=right, chunk_id=chunk.chunk_id)
                for left, right in combinations(normalized_names, 2)
            ]
            self._graph_store.add_edges(edges)


class GraphRetriever:
    """Retrieve chunks connected to seed entities in the graph."""

    def __init__(self, graph_store: SQLiteGraphStore) -> None:
        """Create a graph retriever."""
        self._graph_store = graph_store

    async def retrieve(self, seed_entities: list[str], limit: int = 10) -> list[SearchResult]:
        """Return chunks directly mentioning seed entities."""
        results: list[SearchResult] = []
        for chunk in self._graph_store.chunks_for_entities(seed_entities, limit=limit):
            results.append(
                SearchResult(chunk=chunk, score=1.0, retriever="graph", path=seed_entities)
            )
        return results[:limit]
