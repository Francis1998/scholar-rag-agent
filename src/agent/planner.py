"""Planner that decomposes observed queries into retrieval sub-tasks."""

from agent.models import QueryIntent, QueryObservation, QueryPlan, RetrievalTask


class Planner:
    """Create retrieval plans with JSON-serializable rationale traces."""

    def plan(self, run_id: str, observation: QueryObservation) -> QueryPlan:
        """Build a retrieval plan for an observed query."""
        base_query = observation.original_query
        if observation.intent == QueryIntent.COMPARISON:
            tasks = [
                RetrievalTask(
                    task_id="comparison-evidence",
                    query=base_query,
                    rationale="Retrieve evidence for each compared concept before synthesis.",
                    target_entities=observation.entities,
                    max_hops=3,
                ),
                RetrievalTask(
                    task_id="contrast-findings",
                    query=f"contrasting results for {base_query}",
                    rationale="Find disagreements, limitations, and methodological differences.",
                    target_entities=observation.entities,
                    max_hops=2,
                ),
            ]
        elif observation.intent == QueryIntent.HYPOTHESIS_VALIDATION:
            tasks = [
                RetrievalTask(
                    task_id="supporting-evidence",
                    query=f"evidence supporting {base_query}",
                    rationale="Collect papers that support the hypothesis.",
                    target_entities=observation.entities,
                    max_hops=3,
                ),
                RetrievalTask(
                    task_id="counter-evidence",
                    query=f"evidence refuting {base_query}",
                    rationale="Collect papers that challenge the hypothesis.",
                    target_entities=observation.entities,
                    max_hops=3,
                ),
            ]
        elif observation.intent == QueryIntent.SYNTHESIS:
            tasks = [
                RetrievalTask(
                    task_id="synthesis-corpus",
                    query=base_query,
                    rationale="Retrieve broad context for a literature synthesis.",
                    target_entities=observation.entities,
                    max_hops=3,
                )
            ]
        else:
            tasks = [
                RetrievalTask(
                    task_id="factual-lookup",
                    query=base_query,
                    rationale="Retrieve the most direct evidence for the factual question.",
                    target_entities=observation.entities,
                    max_hops=1,
                )
            ]
        rationale_trace = [f"{task.task_id}: {task.rationale}" for task in tasks]
        return QueryPlan(
            run_id=run_id, observation=observation, tasks=tasks, rationale_trace=rationale_trace
        )
