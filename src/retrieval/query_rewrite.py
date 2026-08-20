"""Deterministic lexical query rewriting and variant generation."""

from collections.abc import Collection, Mapping, Sequence

from retrieval.sparse import STOPWORDS, tokenize

SynonymValues = str | Sequence[str]


class QueryRewriter:
    """Normalize queries and expand caller-provided scientific synonyms."""

    def __init__(
        self,
        synonyms: Mapping[str, SynonymValues],
        stopwords: Collection[str] | None = None,
    ) -> None:
        """Create a deterministic query rewriter.

        Args:
            synonyms: Mapping from terms or phrases to one or more replacement
                phrases. Matching is case-insensitive.
            stopwords: Optional stopword collection replacing the shared
                retrieval defaults.

        Raises:
            TypeError: If a synonym key or value is not a string.
            ValueError: If a synonym key or value has no content terms after
                normalization.
        """
        words = STOPWORDS if stopwords is None else stopwords
        self._stopwords = {
            token for word in words for token in self._checked_tokens(word, "stopword")
        }
        normalized: dict[tuple[str, ...], tuple[tuple[str, ...], ...]] = {}
        for raw_key, raw_values in synonyms.items():
            key = tuple(self._content_tokens(raw_key, "synonym key"))
            if not key:
                raise ValueError("synonym key must contain a non-stopword term")
            values = [raw_values] if isinstance(raw_values, str) else list(raw_values)
            expansions: list[tuple[str, ...]] = []
            for raw_value in values:
                expansion = tuple(self._content_tokens(raw_value, "synonym value"))
                if not expansion:
                    raise ValueError("synonym value must contain a non-stopword term")
                if expansion != key and expansion not in expansions:
                    expansions.append(expansion)
            normalized[key] = tuple(expansions)
        self._synonyms = normalized
        self._keys = sorted(normalized, key=lambda key: (-len(key), key))

    def rewrite(self, query: str) -> str:
        """Return a normalized query with all matching synonyms appended."""
        tokens = self._content_tokens(query, "query")
        if not tokens:
            return ""

        rewritten: list[str] = []
        index = 0
        while index < len(tokens):
            match = self._match_at(tokens, index)
            if match is None:
                rewritten.append(tokens[index])
                index += 1
                continue
            key, expansions = match
            rewritten.extend(key)
            for expansion in expansions:
                rewritten.extend(expansion)
            index += len(key)
        return " ".join(rewritten)

    def variants(self, query: str, max_variants: int = 5) -> list[str]:
        """Return bounded, distinct query variants suitable for result fusion.

        The normalized query comes first, followed by the all-synonyms expansion
        and then single synonym substitutions in query order.
        """
        if not isinstance(max_variants, int) or isinstance(max_variants, bool):
            raise ValueError("max_variants must be a positive integer")
        if max_variants <= 0:
            raise ValueError("max_variants must be a positive integer")

        tokens = self._content_tokens(query, "query")
        if not tokens:
            return []

        candidates = [" ".join(tokens), self.rewrite(query)]
        for index in range(len(tokens)):
            match = self._match_at(tokens, index)
            if match is None:
                continue
            key, expansions = match
            for expansion in expansions:
                variant = [*tokens[:index], *expansion, *tokens[index + len(key) :]]
                candidates.append(" ".join(variant))

        distinct: list[str] = []
        for candidate in candidates:
            if candidate and candidate not in distinct:
                distinct.append(candidate)
            if len(distinct) == max_variants:
                break
        return distinct

    def _content_tokens(self, text: str, field: str) -> list[str]:
        return [
            token for token in self._checked_tokens(text, field) if token not in self._stopwords
        ]

    @staticmethod
    def _checked_tokens(value: object, field: str) -> list[str]:
        if not isinstance(value, str):
            raise TypeError(f"{field} must be a string")
        return tokenize(value)

    def _match_at(
        self,
        tokens: list[str],
        index: int,
    ) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]] | None:
        for key in self._keys:
            if tuple(tokens[index : index + len(key)]) == key:
                return key, self._synonyms[key]
        return None
