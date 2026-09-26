"""Top-level Observe-Decide-Act agent runner."""

import asyncio
from dataclasses import replace
from inspect import signature
from uuid import uuid4

from agent.evidence import EvidenceSnapshot, GenerationRecord, RunConfiguration
from agent.executor import Executor
from agent.models import AgentRunResult, AgentState, QueryObservation, QueryPlan, StateTransition
from agent.observer import QueryAnalyzer
from agent.planner import Planner
from agent.retrieval_preview import RetrievalPreview, RetrievalPreviewError
from agent.safety import CancellationToken, SafetyLimits, with_timeout
from agent.state_machine import AgentStateMachine
from retrieval.evidence_policy import (
    EvidencePolicy,
    ensure_evidence_policy,
    normalize_evidence_policy,
    policy_arguments,
)
from retrieval.scope import (
    DocumentIdsInput,
    ensure_document_scope,
    normalize_document_ids,
    scope_arguments,
)
from storage.event_log import SQLiteEventLog


class AgentRunner:
    """Run one grounded scientific RAG query through the explicit state machine."""

    def __init__(
        self,
        agent_id: str,
        event_log: SQLiteEventLog,
        analyzer: QueryAnalyzer,
        planner: Planner,
        executor: Executor,
        safety_limits: SafetyLimits,
    ) -> None:
        """Create an agent runner with durable event persistence."""
        self._agent_id = agent_id
        self._event_log = event_log
        self._analyzer = analyzer
        self._planner = planner
        self._executor = executor
        self._safety_limits = safety_limits
        self._state_machine = AgentStateMachine()

    async def preview(
        self,
        query: str,
        *,
        document_ids: DocumentIdsInput | None = None,
        max_chunks_per_document: int | None = None,
    ) -> RetrievalPreview:
        """Inspect the query's prepared context without generation, grounding, or event writes.

        Invalid input raises ValueError. Operational failures raise RetrievalPreviewError
        with a diagnostic cause; external task cancellation propagates without journaling.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a nonempty string.")
        scope = normalize_document_ids(document_ids)
        policy = normalize_evidence_policy(max_chunks_per_document)
        phase = "planning"
        try:
            limits = replace(self._safety_limits)
            configuration = self._configuration(limits)
            if self._executor.retrieval_uses_llm:
                raise RetrievalPreviewError(
                    "generative_retrieval",
                    "Preview requires non-generative retrieval; LLM-backed HyDE is configured.",
                    409,
                )
            observation = self._observe(query, scope, policy)
            plan = self._planner.plan(str(uuid4()), observation)
            plan = self._clamp_plan(plan, limits)
            self._validate_plan_request(plan, scope, policy)

            phase = "retrieval"
            retrieved = await with_timeout(
                self._executor.retrieve(
                    plan, configuration.max_source_docs, **scope_arguments(scope)
                ),
                configuration.retrieval_timeout_seconds,
                "retrieval",
            )
            ensure_document_scope(scope, (result.chunk.document_id for result in retrieved))
            if len(retrieved) > configuration.max_source_docs:
                raise ValueError("Retrieved evidence exceeds the effective source limit.")

            phase = "context_preparation"
            snapshot = await with_timeout(
                self._executor.prepare_context(plan, retrieved, **policy_arguments(policy)),
                configuration.reasoning_timeout_seconds,
                "context preparation",
            )
            self._validate_plan_request(plan, scope, policy)
            ensure_document_scope(scope, (source.chunk.document_id for source in snapshot.sources))
            ensure_evidence_policy(policy, snapshot.sources)
            return RetrievalPreview.from_preparation(plan, configuration, snapshot)
        except RetrievalPreviewError:
            raise
        except TimeoutError as exc:
            raise RetrievalPreviewError(
                f"{phase}_timeout", f"Retrieval preview timed out during {phase}.", 504
            ) from exc
        except Exception as exc:
            raise RetrievalPreviewError(
                f"{phase}_failed",
                f"Retrieval preview failed during {phase}; no partial evidence was returned.",
            ) from exc

    async def run(
        self,
        query: str,
        token: CancellationToken | None = None,
        *,
        document_ids: DocumentIdsInput | None = None,
        max_chunks_per_document: int | None = None,
    ) -> AgentRunResult:
        """Execute an Observe-Decide-Act query and return the final result."""
        scope = normalize_document_ids(document_ids)
        policy = normalize_evidence_policy(max_chunks_per_document)
        run_id = str(uuid4())
        cancellation_token = token or CancellationToken()
        state = AgentState.IDLE
        observation: QueryObservation | None = None
        plan: QueryPlan | None = None
        context_captured = False

        def record_context(snapshot: EvidenceSnapshot) -> None:
            nonlocal context_captured
            if policy is not None:
                if plan is None:
                    raise ValueError("Evidence capture requires the requested plan.")
                self._validate_plan_request(plan, scope, policy)
                ensure_document_scope(
                    scope, (source.chunk.document_id for source in snapshot.sources)
                )
                ensure_evidence_policy(policy, snapshot.sources)
            self._event_log.append_event(
                agent_id=self._agent_id,
                run_id=run_id,
                event_type="evidence_snapshot",
                payload=snapshot.model_dump(mode="json"),
            )
            context_captured = True

        def record_generation(generation: GenerationRecord) -> None:
            self._event_log.append_event(
                agent_id=self._agent_id,
                run_id=run_id,
                event_type="generation_record",
                payload=generation.model_dump(mode="json"),
            )

        try:
            cancellation_token.raise_if_cancelled()
            limits = replace(self._safety_limits)
            configuration = self._configuration(limits)
            state = self._transition(
                run_id,
                state,
                AgentState.PLANNING,
                {
                    "query": query,
                    "configuration": configuration.model_dump(mode="json"),
                    "document_ids": scope,
                    "evidence_policy": policy.model_dump(mode="json") if policy else None,
                },
            )
            observation = self._observe(query, scope, policy)
            plan = self._planner.plan(run_id, observation)
            plan = self._clamp_plan(plan, limits)
            self._validate_plan_request(plan, scope, policy)
            self._event_log.append_event(
                agent_id=self._agent_id,
                run_id=run_id,
                event_type="decision_log",
                payload=plan.model_dump(mode="json"),
            )

            cancellation_token.raise_if_cancelled()
            state = self._transition(
                run_id, state, AgentState.RETRIEVING, plan.model_dump(mode="json")
            )
            retrieved = await with_timeout(
                self._executor.retrieve(
                    plan,
                    configuration.max_source_docs,
                    **scope_arguments(scope),
                ),
                configuration.retrieval_timeout_seconds,
                "retrieval",
            )
            ensure_document_scope(scope, (result.chunk.document_id for result in retrieved))
            self._validate_plan_request(plan, scope, policy)
            if policy is not None and len(retrieved) > configuration.max_source_docs:
                raise ValueError("Retrieved evidence exceeds the effective source limit.")

            cancellation_token.raise_if_cancelled()
            state = self._transition(
                run_id,
                state,
                AgentState.REASONING,
                {"chunk_ids": [result.chunk.chunk_id for result in retrieved]},
            )
            answer_method = self._executor.answer
            # Bind before calling: retrying a generation TypeError could run the model twice.
            try:
                signature(answer_method).bind(
                    plan,
                    retrieved,
                    on_context=record_context,
                    on_generation=record_generation,
                    **policy_arguments(policy),
                )
            except TypeError as exc:
                if policy is not None:
                    raise TypeError(
                        "Executor.answer does not support max_chunks_per_document "
                        "with evidence_policy and evidence capture callbacks."
                    ) from exc
                answer_call = answer_method(plan, retrieved)
            else:
                answer_call = answer_method(
                    plan,
                    retrieved,
                    on_context=record_context,
                    on_generation=record_generation,
                    **policy_arguments(policy),
                )
            answer = await with_timeout(
                answer_call,
                configuration.reasoning_timeout_seconds,
                "reasoning",
            )
            if policy is not None and not context_captured:
                raise ValueError(
                    "Executor.answer did not capture evidence for max_chunks_per_document."
                )
            ensure_document_scope(scope, (citation.document_id for citation in answer.citations))

            cancellation_token.raise_if_cancelled()
            state = self._transition(
                run_id, state, AgentState.ANSWERING, answer.model_dump(mode="json")
            )
            state = self._transition(
                run_id, state, AgentState.DONE, {"ungrounded": answer.ungrounded}
            )
            return AgentRunResult(
                run_id=run_id,
                state=state,
                observation=observation,
                plan=plan,
                answer=answer,
            )
        except asyncio.CancelledError as exc:
            if state not in {AgentState.DONE, AgentState.ERROR}:
                reason = "agent run was cancelled"
                if str(exc):
                    reason += f": {exc}"
                self._transition(run_id, state, AgentState.ERROR, {"error": reason})
            raise
        except Exception as exc:
            if state not in {AgentState.DONE, AgentState.ERROR}:
                state = self._transition(run_id, state, AgentState.ERROR, {"error": str(exc)})
            return AgentRunResult(
                run_id=run_id,
                state=state,
                observation=observation,
                plan=plan,
                error=str(exc),
            )

    def _observe(
        self, query: str, scope: tuple[str, ...] | None, policy: EvidencePolicy | None
    ) -> QueryObservation:
        """Detach the analyzed query and the request's immutable scope and evidence policy."""
        return self._analyzer.analyze(query).model_copy(
            deep=True, update={"document_ids": scope, "evidence_policy": policy}
        )

    @staticmethod
    def _validate_plan_request(
        plan: QueryPlan, scope: tuple[str, ...] | None, policy: EvidencePolicy | None
    ) -> None:
        if plan.observation.document_ids != scope:
            raise ValueError("Planned document_ids do not match the requested scope.")
        if plan.observation.evidence_policy != policy:
            raise ValueError("Planned evidence_policy does not match the requested quota.")

    @staticmethod
    def _configuration(limits: SafetyLimits) -> RunConfiguration:
        """Use the same effective source, hop, and phase limits in both entrypoints."""
        return RunConfiguration(
            max_source_docs=limits.clamp_sources(limits.max_source_docs),
            max_hops=limits.clamp_hops(limits.max_hops),
            retrieval_timeout_seconds=limits.retrieval_timeout_seconds,
            reasoning_timeout_seconds=limits.reasoning_timeout_seconds,
        )

    def _transition(
        self,
        run_id: str,
        from_state: AgentState,
        to_state: AgentState,
        payload: dict[str, object],
    ) -> AgentState:
        """Validate and persist a transition before returning the new state."""
        self._state_machine.validate(from_state, to_state)
        transition = StateTransition(
            agent_id=self._agent_id,
            run_id=run_id,
            from_state=from_state,
            to_state=to_state,
            payload=payload,
        )
        self._event_log.append_transition(transition)
        return to_state

    def _clamp_plan(self, plan: QueryPlan, limits: SafetyLimits) -> QueryPlan:
        """Apply configured safety limits to planned retrieval tasks."""
        clamped_tasks = [
            task.model_copy(update={"max_hops": limits.clamp_hops(task.max_hops)})
            for task in plan.tasks
        ]
        return plan.model_copy(update={"tasks": clamped_tasks})
