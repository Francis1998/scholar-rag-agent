"""Drop retrieval hits that lack required chunk metadata keys."""

from collections.abc import Sequence

from retrieval.models import SearchResult


class RequiredMetadataGate:
    """Keep only results whose chunk metadata contains required non-empty keys.

    Inspired by Haystack ``MetadataRouter`` / LlamaIndex ``MetadataFilters``.
    Unlike :class:`~retrieval.temporal_freshness_cutoff.TemporalFreshnessCutoff`,
    which interprets date-shaped fields, this gate only checks that each
    required metadata key is present with a non-empty string value. An empty
    ``required_keys`` sequence is a pass-through. Input objects are never
    mutated; survivors keep their original identity, score, and order.
    """

    def __init__(self, required_keys: Sequence[str]) -> None:
        """Create a required-metadata gate.

        Args:
            required_keys: Metadata keys that must be present with a non-empty
                (after strip) string value on every kept chunk. An empty
                sequence disables filtering.
        """
        self._required_keys = tuple(required_keys)

    def filter(self, results: list[SearchResult]) -> list[SearchResult]:
        """Return ``results`` that satisfy every required metadata key.

        A result is dropped when any required key is missing from
        ``chunk.metadata`` or the stored value strips to an empty string.
        When ``required_keys`` is empty, every input result is returned
        unchanged (including empty input → ``[]``).
        """
        if not self._required_keys:
            return list(results)

        kept: list[SearchResult] = []
        for result in results:
            if self._has_required(result.chunk.metadata):
                kept.append(result)
        return kept

    def _has_required(self, metadata: dict[str, str]) -> bool:
        for key in self._required_keys:
            value = metadata.get(key)
            if value is None or not str(value).strip():
                return False
        return True
