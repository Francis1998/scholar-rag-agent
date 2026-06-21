"""Run a deterministic local demo without external API keys."""

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from agent.executor import Executor
from agent.observer import QueryAnalyzer
from agent.planner import Planner
from agent.runner import AgentRunner
from agent.safety import SafetyLimits
from ingestion.chunking import stable_id
from ingestion.pipeline import IngestionPipeline
from llm.fake import FakeLLMAdapter
from retrieval.citations import CitationGrounder
from retrieval.dense import DenseRetriever
from retrieval.graph import GraphRAGBuilder
from retrieval.hybrid import HybridRetriever
from retrieval.hyde import HyDEExpander
from retrieval.models import Document
from retrieval.multihop import MultiHopRetriever
from retrieval.rerank import AdaptiveReranker
from retrieval.sparse import BM25Retriever
from storage.document_store import SQLiteDocumentStore
from storage.event_log import SQLiteEventLog
from storage.graph_store import SQLiteGraphStore

DEMO_TEXT = (
    "GraphRAG connects entities across scientific papers to support multi-hop reasoning. "
    "Hybrid retrieval combines dense semantic similarity with sparse BM25 matching, and "
    "reciprocal rank fusion improves robustness when either signal is noisy. Citation "
    "grounding maps generated claims back to source chunks."
)


async def run_demo() -> None:
    """Execute a deterministic ingestion and query demo."""
    with TemporaryDirectory() as temporary_directory:
        database_path = Path(temporary_directory) / "demo.sqlite3"
        event_log = SQLiteEventLog(database_path)
        document_store = SQLiteDocumentStore(database_path)
        graph_store = SQLiteGraphStore(database_path)
        hybrid = HybridRetriever(DenseRetriever(), BM25Retriever(), HyDEExpander())
        pipeline = IngestionPipeline(document_store, hybrid, GraphRAGBuilder(graph_store))
        document = Document(
            document_id=stable_id("demo-paper", "doc"),
            title="GraphRAG for Scientific Retrieval",
            text=DEMO_TEXT,
            source="fixture",
            metadata={"source_type": "fixture"},
        )
        chunks = pipeline.ingest_documents([document])
        executor = Executor(
            retriever=hybrid,
            multihop_retriever=MultiHopRetriever(graph_store),
            reranker=AdaptiveReranker(),
            llm=FakeLLMAdapter(),
            grounder=CitationGrounder(),
        )
        runner = AgentRunner(
            agent_id="demo-agent",
            event_log=event_log,
            analyzer=QueryAnalyzer(),
            planner=Planner(),
            executor=executor,
            safety_limits=SafetyLimits(),
        )
        result = await runner.run("How does GraphRAG improve scientific literature retrieval?")
        print(f"Ingested chunks: {[chunk.chunk_id for chunk in chunks]}")
        print(f"Run ID: {result.run_id}")
        print(f"State: {result.state}")
        if result.plan is not None:
            print("Planner trace:")
            for rationale in result.plan.rationale_trace:
                print(f"- {rationale}")
        if result.answer is not None:
            print(result.answer.answer)
            for citation in result.answer.citations:
                print(f"Citation {citation.chunk_id}: {citation.title}")


def main() -> None:
    """Run the local demo entrypoint."""
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()
