"""Top-level Observe-Decide-Act agent runner."""

from uuid import uuid4

from agent.executor import Executor
from agent.models import AgentRunResult, AgentState, QueryObservation, QueryPlan, StateTransition
from agent.observer import QueryAnalyzer
from agent.planner import Planner
from agent.safety import CancellationToken, SafetyLimits, with_timeout
from agent.state_machine import AgentStateMachine
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

    async def run(self, query: str, token: CancellationToken | None = None) -> AgentRunResult:
        """Execute an Observe-Decide-Act query and return the final result."""
        run_id = str(uuid4())
        cancellation_token = token or CancellationToken()
        state = AgentState.IDLE
        observation: QueryObservation | None = None
        plan: QueryPlan | None = None
        try:
            cancellation_token.raise_if_cancelled()
            state = self._transition(run_id, state, AgentState.PLANNING, {"query": query})
            observation = self._analyzer.analyze(query)
            plan = self._planner.plan(run_id, observation)
            plan = self._clamp_plan(plan)
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
                self._executor.retrieve(plan, self._safety_limits.clamp_sources(8)),
                self._safety_limits.retrieval_timeout_seconds,
                "retrieval",
            )

            cancellation_token.raise_if_cancelled()
            state = self._transition(
                run_id,
                state,
                AgentState.REASONING,
                {"chunk_ids": [result.chunk.chunk_id for result in retrieved]},
            )
            answer = await with_timeout(
                self._executor.answer(plan, retrieved),
                self._safety_limits.reasoning_timeout_seconds,
                "reasoning",
            )

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

    def _clamp_plan(self, plan: QueryPlan) -> QueryPlan:
        """Apply configured safety limits to planned retrieval tasks."""
        clamped_tasks = [
            task.model_copy(update={"max_hops": self._safety_limits.clamp_hops(task.max_hops)})
            for task in plan.tasks
        ]
        return plan.model_copy(update={"tasks": clamped_tasks})
