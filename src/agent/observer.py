"""Query observation and intent classification."""

import re

from agent.models import QueryIntent, QueryObservation

_ENTITY_PATTERN = re.compile(r"\b[A-Z][A-Za-z0-9\-]{2,}(?:\s+[A-Z][A-Za-z0-9\-]{2,})*\b")


class QueryAnalyzer:
    """Classify research query intent and extract coarse entities."""

    def analyze(self, query: str) -> QueryObservation:
        """Return a structured observation for a user query."""
        normalized_query = query.strip()
        lowered_query = normalized_query.lower()
        if any(token in lowered_query for token in ("compare", "versus", " vs ", "difference")):
            intent = QueryIntent.COMPARISON
        elif any(
            token in lowered_query for token in ("hypothesis", "validate", "support", "refute")
        ):
            intent = QueryIntent.HYPOTHESIS_VALIDATION
        elif any(
            token in lowered_query
            for token in ("synthesize", "summarize", "literature", "overview")
        ):
            intent = QueryIntent.SYNTHESIS
        else:
            intent = QueryIntent.FACTUAL_LOOKUP
        entities = sorted({match.group(0) for match in _ENTITY_PATTERN.finditer(normalized_query)})
        return QueryObservation(original_query=normalized_query, intent=intent, entities=entities)
