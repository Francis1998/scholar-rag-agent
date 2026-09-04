"""Gate that keeps results whose metadata year falls within a range."""

from retrieval.models import SearchResult


class YearRangeGate:
    """Keep results whose ``metadata["year"]`` is inside [min_year, max_year].

    Results missing a parseable year are dropped when a bound is set.

    Inspired by Haystack/LlamaIndex metadata year filters for temporal
    scoping of scholarly corpora.
    Inputs are not mutated.  Local postprocessor for GPT-5.5 /
    Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 pipelines (not a DOI
    connector).
    """

    def __init__(
        self,
        min_year: int | None = None,
        max_year: int | None = None,
    ) -> None:
        """Create a year-range gate.

        Args:
            min_year: Inclusive lower bound, or ``None`` for open.
            max_year: Inclusive upper bound, or ``None`` for open.

        Raises:
            ValueError: If bounds are invalid integers or min > max.
        """
        for label, value in (("min_year", min_year), ("max_year", max_year)):
            if value is not None and not isinstance(value, int):
                raise ValueError(f"{label} must be an int or None")
        if min_year is not None and max_year is not None and min_year > max_year:
            raise ValueError("min_year must be <= max_year")
        self._min_year = min_year
        self._max_year = max_year

    @staticmethod
    def _parse_year(raw: object) -> int | None:
        if raw is None:
            return None
        text = str(raw).strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return None

    def gate(
        self,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Return results whose year is inside the configured range."""
        if not results:
            return []

        kept: list[SearchResult] = []
        limit = len(results) if top_k is None else min(top_k, len(results))
        if limit <= 0:
            return []

        for r in results:
            year = self._parse_year(r.chunk.metadata.get("year"))
            if year is None:
                continue
            if self._min_year is not None and year < self._min_year:
                continue
            if self._max_year is not None and year > self._max_year:
                continue
            kept.append(
                SearchResult(
                    chunk=r.chunk,
                    score=r.score,
                    retriever="year_range_gate",
                    path=[*r.path, r.retriever],
                )
            )
            if len(kept) >= limit:
                break
        return kept
