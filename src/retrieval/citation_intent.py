"""Rule-based citation-intent classification for retrieval queries."""

from enum import StrEnum

from retrieval.models import SearchResult
from retrieval.sparse import tokenize


class CitationIntent(StrEnum):
    """Supported scholarly citation intents."""

    BACKGROUND = "background"
    METHOD = "method"
    RESULT = "result"
    COMPARISON = "comparison"
    UNKNOWN = "unknown"


_RULES: dict[CitationIntent, tuple[str, ...]] = {
    CitationIntent.COMPARISON: (
        "compare",
        "compared",
        "comparing",
        "comparison",
        "versus",
        "vs",
        "contrast",
        "difference",
        "differences",
        "outperform",
        "outperforms",
        "better than",
        "relative to",
    ),
    CitationIntent.METHOD: (
        "method",
        "methods",
        "methodology",
        "approach",
        "approaches",
        "algorithm",
        "algorithms",
        "technique",
        "techniques",
        "protocol",
        "protocols",
        "procedure",
        "procedures",
        "implementation",
        "architecture",
        "framework",
    ),
    CitationIntent.RESULT: (
        "result",
        "results",
        "finding",
        "findings",
        "outcome",
        "outcomes",
        "effect",
        "effects",
        "evidence",
        "efficacy",
        "accuracy",
        "performance",
        "conclude",
        "concludes",
        "concluded",
        "show",
        "shows",
        "showed",
        "demonstrate",
        "demonstrates",
        "demonstrated",
    ),
    CitationIntent.BACKGROUND: (
        "background",
        "overview",
        "introduction",
        "history",
        "review",
        "survey",
        "define",
        "definition",
        "what is",
        "prior work",
        "related work",
        "state of the art",
        "literature landscape",
    ),
}

_PRIORITY = (
    CitationIntent.COMPARISON,
    CitationIntent.METHOD,
    CitationIntent.RESULT,
    CitationIntent.BACKGROUND,
)


class CitationIntentClassifier:
    """Classify citation intent and annotate results for downstream ranking."""

    def __init__(self, metadata_key: str = "citation_intent") -> None:
        """Create a keyword-based intent classifier.

        Args:
            metadata_key: Chunk metadata key populated by :meth:`attach`.

        Raises:
            ValueError: If ``metadata_key`` is blank.
        """
        if not metadata_key.strip():
            raise ValueError("metadata_key must not be blank")
        self._metadata_key = metadata_key.strip()

    def classify(self, query: str) -> CitationIntent:
        """Return the highest-scoring intent under deterministic keyword rules."""
        tokens = tokenize(query)
        if not tokens:
            return CitationIntent.UNKNOWN

        scores = {
            intent: sum(self._contains(tokens, tokenize(rule)) for rule in rules)
            for intent, rules in _RULES.items()
        }
        highest = max(scores.values(), default=0)
        if highest == 0:
            return CitationIntent.UNKNOWN
        return next(intent for intent in _PRIORITY if scores[intent] == highest)

    def attach(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        """Return copied results with classified intent in chunk metadata."""
        intent = self.classify(query).value
        annotated: list[SearchResult] = []
        for result in results:
            metadata = {**result.chunk.metadata, self._metadata_key: intent}
            chunk = result.chunk.model_copy(deep=True, update={"metadata": metadata})
            annotated.append(result.model_copy(deep=True, update={"chunk": chunk}))
        return annotated

    @staticmethod
    def _contains(tokens: list[str], phrase: list[str]) -> bool:
        width = len(phrase)
        return any(
            tokens[index : index + width] == phrase for index in range(len(tokens) - width + 1)
        )
