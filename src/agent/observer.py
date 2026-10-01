"""Query observation and intent classification."""

import re

from agent.models import QueryIntent, QueryObservation

_ENTITY_PATTERN = re.compile(r"\b[A-Z][A-Za-z0-9\-]{2,}(?:\s+[A-Z][A-Za-z0-9\-]{2,})*\b")
_COMPARISON_PATTERN = re.compile(r"\b(?:compare[ds]?|versus|vs|differences?)\b")
_HYPOTHESIS_PATTERN = re.compile(
    r"\b(?:hypothesis|validate[ds]?|support(?:s|ed|ing)?|refute[ds]?)\b"
)
_SYNTHESIS_PATTERN = re.compile(r"\b(?:synthesize[ds]?|summarize[ds]?|literatures?|overviews?)\b")


class QueryAnalyzer:
    """Classify research query intent and extract coarse entities."""

    def analyze(self, query: str) -> QueryObservation:
        """Return a structured observation for a user query."""
        normalized_query = query.strip()
        lowered_query = normalized_query.lower()
        if _COMPARISON_PATTERN.search(lowered_query):
            intent = QueryIntent.COMPARISON
        elif _HYPOTHESIS_PATTERN.search(lowered_query):
            intent = QueryIntent.HYPOTHESIS_VALIDATION
        elif _SYNTHESIS_PATTERN.search(lowered_query):
            intent = QueryIntent.SYNTHESIS
        else:
            intent = QueryIntent.FACTUAL_LOOKUP
        entities = sorted({match.group(0) for match in _ENTITY_PATTERN.finditer(normalized_query)})
        return QueryObservation(original_query=normalized_query, intent=intent, entities=entities)
