"""Deterministic multi-query decomposition for complex research questions."""

import re

_CONJUNCTION_SPLIT = re.compile(
    r"\s+(?:and also|as well as|along with|in addition to|and then|and)\s+|"
    r"\s*;\s*|"
    r"\s*\?\s*",
    re.IGNORECASE,
)


class QueryDecomposer:
    """Split compound queries into distinct retrieval sub-queries.

    Inspired by LlamaIndex and Haystack multi-query patterns, but fully
    deterministic: conjunctions and question marks drive the split without an
    LLM. The original query is always preserved as the first returned item.
    """

    def decompose(self, query: str, max_parts: int | None = None) -> list[str]:
        """Return the original query followed by distinct sub-queries.

        Args:
            query: User question that may contain multiple intents.
            max_parts: Optional inclusive cap on returned parts. When set, it
                must be a positive integer. The original query still comes
                first when any parts are returned.

        Raises:
            ValueError: If ``max_parts`` is provided and is not a positive
                integer.
        """
        if max_parts is not None and (
            not isinstance(max_parts, int) or isinstance(max_parts, bool) or max_parts <= 0
        ):
            raise ValueError("max_parts must be a positive integer")

        normalized = " ".join(query.split())
        if not normalized or not normalized.strip(" ?;."):
            return []

        parts = [normalized]
        for fragment in _CONJUNCTION_SPLIT.split(normalized):
            cleaned = " ".join(fragment.split()).strip(" ?;.")
            if cleaned and cleaned.casefold() not in {part.casefold() for part in parts}:
                parts.append(cleaned)

        if max_parts is None:
            return parts
        return parts[:max_parts]
