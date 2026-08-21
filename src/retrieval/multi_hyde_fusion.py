"""Multi-HyDE query expansion and reciprocal-rank fusion."""

from typing import Protocol

from llm.base import BaseLLMAdapter
from llm.schemas import LLMRequest, TaskType
from retrieval.models import SearchResult
from retrieval.rrf import reciprocal_rank_fusion


class QueryRetriever(Protocol):
    """Asynchronous retriever accepted by :class:`MultiHydeFusion`."""

    async def retrieve(self, query: str, limit: int = 10) -> list[SearchResult]:
        """Return ranked results for a query string."""


_PERSPECTIVES = (
    (
        "Background",
        "It defines the central concepts and summarizes the relevant prior literature.",
    ),
    (
        "Methods",
        "It describes representative study designs, datasets, measurements, and analyses.",
    ),
    (
        "Findings",
        "It reports likely outcomes, effect directions, comparisons, and supporting evidence.",
    ),
    (
        "Limitations",
        "It identifies uncertainty, boundary conditions, conflicting evidence, and open questions.",
    ),
)


class MultiHydeFusion:
    """Retrieve against several hypothetical abstracts and fuse their rankings."""

    def __init__(
        self,
        retriever: QueryRetriever,
        num_hypotheses: int = 3,
        llm: BaseLLMAdapter | None = None,
        rank_constant: int = 60,
    ) -> None:
        """Create a deterministic Multi-HyDE retrieval stage.

        Args:
            retriever: Retriever that embeds or otherwise searches each expanded query.
            num_hypotheses: Number of hypothetical abstracts generated per query.
            llm: Optional provider-independent adapter used instead of local templates.
            rank_constant: Positive reciprocal-rank-fusion smoothing constant.

        Raises:
            ValueError: If ``num_hypotheses`` or ``rank_constant`` is not a
                positive integer.
        """
        self._validate_positive_integer(num_hypotheses, "num_hypotheses")
        self._validate_positive_integer(rank_constant, "rank_constant")
        self._retriever = retriever
        self._num_hypotheses = num_hypotheses
        self._llm = llm
        self._rank_constant = rank_constant

    async def hypothetical_abstracts(self, query: str) -> list[str]:
        """Return distinct hypothetical abstracts in deterministic perspective order."""
        normalized_query = " ".join(query.split())
        if not normalized_query:
            return []

        abstracts: list[str] = []
        for index in range(self._num_hypotheses):
            fallback = self._deterministic_abstract(normalized_query, index)
            if self._llm is None:
                abstract = fallback
            else:
                response = await self._llm.generate(
                    LLMRequest(
                        task_type=TaskType.SPEED,
                        prompt=(
                            "Write one concise hypothetical scientific abstract for the "
                            f"research question below. Use the {_PERSPECTIVES[index % 4][0].lower()} "
                            f"perspective and do not include citations.\nQuestion: {normalized_query}"
                        ),
                        context=f"Multi-HyDE variant {index + 1} of {self._num_hypotheses}",
                        citation_chunk_ids=[],
                    )
                )
                abstract = response.text.strip() or fallback
            abstracts.append(abstract)
        return abstracts

    async def expanded_queries(self, query: str) -> list[str]:
        """Return the original query paired with each hypothetical abstract."""
        normalized_query = " ".join(query.split())
        return [
            f"{normalized_query}\nHypothetical abstract: {abstract}"
            for abstract in await self.hypothetical_abstracts(normalized_query)
        ]

    async def retrieve(self, query: str, limit: int = 10) -> list[SearchResult]:
        """Retrieve every expansion and fuse duplicate chunk ids with shared RRF."""
        if limit <= 0:
            return []

        result_sets: list[list[SearchResult]] = []
        for index, expanded_query in enumerate(await self.expanded_queries(query), start=1):
            results = await self._retriever.retrieve(expanded_query, limit=limit)
            result_sets.append(
                [
                    result.model_copy(
                        deep=True,
                        update={
                            "retriever": f"multi_hyde:{index}",
                            "path": [*result.path, result.retriever],
                        },
                    )
                    for result in results
                ]
            )
        if not result_sets:
            return []
        return reciprocal_rank_fusion(
            result_sets,
            limit=limit,
            rank_constant=self._rank_constant,
        )

    @staticmethod
    def _deterministic_abstract(query: str, index: int) -> str:
        label, detail = _PERSPECTIVES[index % len(_PERSPECTIVES)]
        return (
            f"{label} perspective {index + 1}: This hypothetical study addresses {query}. "
            f"{detail} The abstract records methods, evidence, findings, and limitations "
            "relevant to the question."
        )

    @staticmethod
    def _validate_positive_integer(value: int, field: str) -> None:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{field} must be a positive integer")
