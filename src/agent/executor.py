"""Agent executor that retrieves, reasons, validates, and grounds answers."""

from agent.models import AgentAnswer, Claim, QueryPlan
from llm.base import BaseLLMAdapter
from llm.schemas import LLMRequest, TaskType
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

    async def answer(self, plan: QueryPlan, retrieved: list[SearchResult]) -> AgentAnswer:
        """Generate and ground an answer using retrieved chunks."""
        reranked = await self._reranker.rerank(plan.observation.original_query, retrieved)
        context_lines = [
            f"[{result.chunk.chunk_id}] {result.chunk.title}: {result.chunk.text}"
            for result in reranked
        ]
        response = await self._llm.generate(
            LLMRequest(
                task_type=TaskType.REASONING,
                prompt=plan.observation.original_query,
                context="\n".join(context_lines),
                citation_chunk_ids=[result.chunk.chunk_id for result in reranked],
            )
        )
        raw_claims = response.parsed_claims or [response.text]
        claims = [Claim(text=claim, chunk_ids=response.citation_chunk_ids) for claim in raw_claims]
        return self._grounder.ground(
            answer_text=response.text,
            claims=claims,
            retrieved_chunks=[result.chunk for result in reranked],
        )
