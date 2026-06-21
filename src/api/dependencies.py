"""Application dependency wiring for local runtime components."""

from pathlib import Path

from agent.executor import Executor
from agent.observer import QueryAnalyzer
from agent.planner import Planner
from agent.runner import AgentRunner
from agent.safety import SafetyLimits
from config import Settings, load_settings
from ingestion.pipeline import IngestionPipeline
from llm.fake import FakeLLMAdapter
from retrieval.citations import CitationGrounder
from retrieval.dense import DenseRetriever
from retrieval.graph import GraphRAGBuilder
from retrieval.hybrid import HybridRetriever
from retrieval.hyde import HyDEExpander
from retrieval.multihop import MultiHopRetriever
from retrieval.rerank import AdaptiveReranker
from retrieval.sparse import BM25Retriever
from storage.document_store import SQLiteDocumentStore
from storage.event_log import SQLiteEventLog
from storage.graph_store import SQLiteGraphStore


class AppContainer:
    """Container for API runtime dependencies."""

    def __init__(self, settings: Settings) -> None:
        """Create local-first runtime dependencies."""
        database_path = Path(settings.database_path)
        self.event_log = SQLiteEventLog(database_path)
        self.document_store = SQLiteDocumentStore(database_path)
        self.graph_store = SQLiteGraphStore(database_path)
        self.llm = FakeLLMAdapter()
        self.hybrid_retriever = HybridRetriever(
            dense_retriever=DenseRetriever(),
            sparse_retriever=BM25Retriever(),
            hyde_expander=HyDEExpander(),
        )
        existing_chunks = self.document_store.list_chunks()
        if existing_chunks:
            self.hybrid_retriever.add_chunks(existing_chunks)
        self.graph_builder = GraphRAGBuilder(self.graph_store)
        self.ingestion_pipeline = IngestionPipeline(
            document_store=self.document_store,
            hybrid_retriever=self.hybrid_retriever,
            graph_builder=self.graph_builder,
        )
        executor = Executor(
            retriever=self.hybrid_retriever,
            multihop_retriever=MultiHopRetriever(self.graph_store),
            reranker=AdaptiveReranker(),
            llm=self.llm,
            grounder=CitationGrounder(),
        )
        self.runner = AgentRunner(
            agent_id=settings.agent_id,
            event_log=self.event_log,
            analyzer=QueryAnalyzer(),
            planner=Planner(),
            executor=executor,
            safety_limits=SafetyLimits(
                retrieval_timeout_seconds=settings.retrieval_timeout_seconds,
                reasoning_timeout_seconds=settings.reasoning_timeout_seconds,
                max_source_docs=settings.max_source_docs,
                max_hops=settings.max_hops,
            ),
        )


def create_container(settings: Settings | None = None) -> AppContainer:
    """Create a dependency container from settings."""
    return AppContainer(settings or load_settings())
