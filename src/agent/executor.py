"""Agent executor that retrieves, reasons, validates, and grounds answers."""

from collections.abc import Callable

from agent.evidence import EvidenceSnapshot, GenerationRecord
from agent.models import AgentAnswer, Claim, QueryPlan
from llm.base import BaseLLMAdapter
from retrieval.citations import CitationGrounder
from retrieval.hybrid import HybridRetriever
from retrieval.models import SearchResult
from retrieval.multihop import MultiHopRetriever
from retrieval.rerank import AdaptiveReranker


class Executor:
    """Execute retrieval tools and generation for a planned query."""

    def __init__(
        self,
        retriever: HybridRetriever,
        multihop_retriever: MultiHopRetriever,
        reranker: AdaptiveReranker,
        llm: BaseLLMAdapter,
        grounder: CitationGrounder,
    ) -> None:
        """Create an executor from retrieval, LLM, and grounding components."""
        self._retriever = retriever
        self._multihop_retriever = multihop_retriever
        self._reranker = reranker
        self._llm = llm
        self._grounder = grounder

    async def retrieve(self, plan: QueryPlan, max_results: int = 8) -> list[SearchResult]:
        """Run hybrid and multi-hop retrieval for every planned sub-task."""
        merged_results: dict[str, SearchResult] = {}
        for task in plan.tasks:
            task_results = await self._retriever.retrieve(task.query, limit=max_results)
            graph_results = await self._multihop_retriever.retrieve(
                query=task.query,
                seed_entities=task.target_entities,
                depth=task.max_hops,
                limit=max_results,
            )
            for result in [*task_results, *graph_results]:
                existing = merged_results.get(result.chunk.chunk_id)
                if existing is None or result.score > existing.score:
                    merged_results[result.chunk.chunk_id] = result
        return sorted(merged_results.values(), key=lambda result: result.score, reverse=True)[
            :max_results
        ]

    async def answer(
        self,
        plan: QueryPlan,
        retrieved: list[SearchResult],
        *,
        on_context: Callable[[EvidenceSnapshot], None] | None = None,
        on_generation: Callable[[GenerationRecord], None] | None = None,
    ) -> AgentAnswer:
        """Generate and ground an answer using retrieved chunks."""
        reranked = await self._reranker.rerank(plan.observation.original_query, retrieved)
        snapshot = EvidenceSnapshot.capture(plan.observation.original_query, reranked)
        if on_context is not None:
            on_context(snapshot)
        response = await self._llm.generate(snapshot.request.model_copy(deep=True))
        raw_claims = response.parsed_claims or [response.text]
        claims = [Claim(text=claim, chunk_ids=response.citation_chunk_ids) for claim in raw_claims]
        answer = self._grounder.ground(
            answer_text=response.text,
            claims=claims,
            retrieved_chunks=[source.chunk for source in snapshot.sources],
        )
        if on_generation is not None:
            on_generation(
                GenerationRecord(
                    provider=response.raw_provider,
                    model_name=response.model_name,
                    task_type=snapshot.request.task_type,
                    claim_chunk_ids=[claim.chunk_ids for claim in claims],
                )
            )
        return answer
