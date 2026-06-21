"""Pydantic models and enums used by the agent runtime."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class AgentState(StrEnum):
    """Explicit persisted states for an agent run."""

    IDLE = "IDLE"
    PLANNING = "PLANNING"
    RETRIEVING = "RETRIEVING"
    REASONING = "REASONING"
    ANSWERING = "ANSWERING"
    DONE = "DONE"
    ERROR = "ERROR"


class QueryIntent(StrEnum):
    """Research task categories supported by the query analyzer."""

    FACTUAL_LOOKUP = "factual_lookup"
    SYNTHESIS = "synthesis"
    COMPARISON = "comparison"
    HYPOTHESIS_VALIDATION = "hypothesis_validation"


class RetrievalTask(BaseModel):
    """A retrieval sub-task planned from a user query."""

    task_id: str
    query: str
    rationale: str
    target_entities: list[str] = Field(default_factory=list)
    max_hops: int = Field(default=3, ge=0, le=5)


class QueryObservation(BaseModel):
    """Observed query metadata produced before planning."""

    original_query: str
    intent: QueryIntent
    entities: list[str] = Field(default_factory=list)
    constraints: dict[str, Any] = Field(default_factory=dict)


class QueryPlan(BaseModel):
    """Structured plan with rationale trace for retrieval execution."""

    run_id: str
    observation: QueryObservation
    tasks: list[RetrievalTask]
    rationale_trace: list[str]


class StateTransition(BaseModel):
    """Persistable state transition payload."""

    agent_id: str
    run_id: str
    from_state: AgentState
    to_state: AgentState
    payload: dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    """A grounded citation pointing to a retrieved source chunk."""

    chunk_id: str
    document_id: str
    title: str
    snippet: str


class Claim(BaseModel):
    """A generated claim and its supporting chunk IDs."""

    text: str
    chunk_ids: list[str] = Field(default_factory=list)
    grounded: bool = False


class AgentAnswer(BaseModel):
    """Final answer contract returned by the executor and API."""

    answer: str
    citations: list[Citation]
    claims: list[Claim]
    ungrounded: bool = False
    warnings: list[str] = Field(default_factory=list)


class AgentRunResult(BaseModel):
    """Complete run result returned to API and CLI callers."""

    run_id: str
    state: AgentState
    observation: QueryObservation | None = None
    plan: QueryPlan | None = None
    answer: AgentAnswer | None = None
    error: str | None = None
