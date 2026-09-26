"""Read completed evidence bundles from the existing append-only event log."""

from json import JSONDecodeError

from pydantic import BaseModel, JsonValue, ValidationError

from agent.evidence import (
    CitationEvidenceLink,
    ClaimEvidenceLink,
    EvidenceBundle,
    EvidenceSnapshot,
    ExportEvent,
    GenerationRecord,
    RunConfiguration,
)
from agent.models import AgentAnswer, AgentState, QueryPlan
from retrieval.evidence_policy import EvidencePolicy, ensure_evidence_policy
from retrieval.scope import DocumentIds, documents_within_scope
from storage.event_log import SQLiteEventLog

EXPECTED_EVENTS = [
    "state_transition",
    "decision_log",
    "state_transition",
    "state_transition",
    "evidence_snapshot",
    "generation_record",
    "state_transition",
    "state_transition",
]
EXPECTED_TRANSITIONS = [
    (AgentState.IDLE, AgentState.PLANNING),
    (AgentState.PLANNING, AgentState.RETRIEVING),
    (AgentState.RETRIEVING, AgentState.REASONING),
    (AgentState.REASONING, AgentState.ANSWERING),
    (AgentState.ANSWERING, AgentState.DONE),
]
EXPORT_WARNINGS = [
    "Review before sharing: this artifact contains original queries, full source text, "
    "metadata, and generated answers, which may be sensitive.",
    "Citation grounding uses token overlap, not semantic entailment or proof.",
    "Frozen evidence does not promise identical future model output or a signed, "
    "tamper-proof record.",
]


class EvidenceExportError(ValueError):
    """An explicit, non-success export outcome safe to expose through the API."""

    def __init__(self, code: str, message: str, status_code: int = 409) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


class _Transition(BaseModel):
    from_state: AgentState
    to_state: AgentState
    payload: dict[str, JsonValue]


class _Started(BaseModel):
    query: str
    configuration: RunConfiguration
    document_ids: DocumentIds | None = None
    evidence_policy: EvidencePolicy | None = None


class _Retrieved(BaseModel):
    chunk_ids: list[str]


class _Completed(BaseModel):
    ungrounded: bool


def _invalid_record() -> EvidenceExportError:
    return EvidenceExportError(
        "invalid_run_record", "Saved evidence is inconsistent, invalid, or an unsupported version."
    )


class EvidenceExporter:
    """Export saved records without access to a model, retriever, or document store."""

    def __init__(self, event_log: SQLiteEventLog) -> None:
        self._event_log = event_log

    def export(self, run_id: str) -> EvidenceBundle:
        """Freeze the trace at its first terminal event; reject legacy/failed runs."""
        try:
            raw_events = self._event_log.list_events(run_id)
        except JSONDecodeError as exc:
            raise _invalid_record() from exc
        if not raw_events:
            raise EvidenceExportError("run_not_found", "No events exist for this run.", 404)
        terminal_index: int | None = None
        for index, event in enumerate(raw_events):
            payload = event["payload"]
            if event["event_type"] != "state_transition" or not isinstance(payload, dict):
                continue
            if payload.get("to_state") == AgentState.ERROR:
                raise EvidenceExportError("run_failed", "Failed runs cannot be exported.")
            if payload.get("to_state") == AgentState.DONE:
                terminal_index = index
                break
        if terminal_index is None:
            raise EvidenceExportError("run_incomplete", "This run has no terminal DONE event.")
        completed_events = raw_events[: terminal_index + 1]
        if not any(event["event_type"] == "evidence_snapshot" for event in completed_events):
            raise EvidenceExportError(
                "snapshot_unavailable",
                "This run predates evidence capture or has no saved snapshot; it cannot "
                "be reconstructed from the current corpus.",
            )
        try:
            return self._build(run_id, [ExportEvent.model_validate(e) for e in completed_events])
        except ValidationError as exc:
            raise _invalid_record() from exc

    def _build(self, run_id: str, events: list[ExportEvent]) -> EvidenceBundle:
        known = [event for event in events if event.event_type in set(EXPECTED_EVENTS)]
        if [event.event_type for event in known] != EXPECTED_EVENTS:
            raise _invalid_record()
        if any(event.run_id != run_id or event.agent_id != events[0].agent_id for event in events):
            raise _invalid_record()
        transitions = [
            _Transition.model_validate(event.payload)
            for event in known
            if event.event_type == "state_transition"
        ]
        if [(t.from_state, t.to_state) for t in transitions] != EXPECTED_TRANSITIONS:
            raise _invalid_record()
        started = _Started.model_validate(transitions[0].payload)
        plan = QueryPlan.model_validate(known[1].payload)
        retrieval_plan = QueryPlan.model_validate(transitions[1].payload)
        retrieved = _Retrieved.model_validate(transitions[2].payload)
        snapshot = EvidenceSnapshot.model_validate(known[4].payload)
        generation = GenerationRecord.model_validate(known[5].payload)
        answer = AgentAnswer.model_validate(transitions[3].payload)
        completed = _Completed.model_validate(transitions[4].payload)
        if (
            plan != retrieval_plan
            or plan.run_id != run_id
            or started.document_ids != plan.observation.document_ids
            or started.evidence_policy != plan.observation.evidence_policy
            or started.query.strip() != plan.observation.original_query
            or snapshot.request.prompt != plan.observation.original_query
            or generation.task_type != snapshot.request.task_type
            or len(generation.claim_chunk_ids) != len(answer.claims)
            or completed.ungrounded != answer.ungrounded
        ):
            raise _invalid_record()
        if not documents_within_scope(
            started.document_ids,
            [
                *(source.chunk.document_id for source in snapshot.sources),
                *(citation.document_id for citation in answer.citations),
            ],
        ):
            raise _invalid_record()
        try:
            ensure_evidence_policy(started.evidence_policy, snapshot.sources)
        except ValueError as exc:
            raise _invalid_record() from exc

        warnings = list(EXPORT_WARNINGS)
        rank_by_id = {source.chunk.chunk_id: source.rank for source in snapshot.sources}
        claim_links = []
        for number, (claim, proposed) in enumerate(
            zip(answer.claims, generation.claim_chunk_ids, strict=True), start=1
        ):
            referenced = list(dict.fromkeys([*proposed, *claim.chunk_ids]))
            missing = [chunk_id for chunk_id in referenced if chunk_id not in rank_by_id]
            claim_links.append(
                ClaimEvidenceLink(
                    claim_number=number,
                    proposed_chunk_ids=proposed,
                    grounded_chunk_ids=claim.chunk_ids,
                    evidence_ranks=[rank_by_id[c] for c in referenced if c in rank_by_id],
                    missing_chunk_ids=missing,
                )
            )
            if missing:
                warnings.append(
                    f"Claim {number} references missing evidence: {', '.join(missing)}."
                )
        citation_links = [
            CitationEvidenceLink(
                citation_number=number,
                chunk_id=citation.chunk_id,
                evidence_rank=rank_by_id.get(citation.chunk_id),
            )
            for number, citation in enumerate(answer.citations, start=1)
        ]
        for link in citation_links:
            if link.evidence_rank is None:
                warnings.append(
                    f"Citation {link.citation_number} references missing evidence: {link.chunk_id}."
                )

        payloads: list[BaseModel] = [started, retrieval_plan, retrieved, answer, completed]
        safe_transitions = {
            event.id: _Transition(
                from_state=transition.from_state,
                to_state=transition.to_state,
                payload=payload.model_dump(mode="json"),
            ).model_dump(mode="json")
            for event, transition, payload in zip(
                (event for event in known if event.event_type == "state_transition"),
                transitions,
                payloads,
                strict=True,
            )
        }
        safe_events = []
        for event in events:
            if event.event_type == "state_transition":
                safe_events.append(event.model_copy(update={"payload": safe_transitions[event.id]}))
            elif event.event_type == "decision_log":
                safe_events.append(
                    event.model_copy(update={"payload": plan.model_dump(mode="json")})
                )
            elif event.event_type in {"evidence_snapshot", "generation_record"}:
                field = "snapshot" if event.event_type == "evidence_snapshot" else "generation"
                safe_events.append(event.model_copy(update={"payload": None, "payload_ref": field}))
            else:
                safe_events.append(
                    event.model_copy(update={"payload": None, "payload_omitted": True})
                )
                warnings.append(f"Event {event.id} payload omitted: not an allowlisted event type.")

        return EvidenceBundle(
            run_id=run_id,
            agent_id=events[0].agent_id,
            query=started.query,
            completed_at=events[-1].timestamp,
            configuration=started.configuration,
            plan=plan,
            answer=answer,
            snapshot=snapshot,
            generation=generation,
            claim_evidence=claim_links,
            citation_evidence=citation_links,
            events=safe_events,
            warnings=warnings,
        )
