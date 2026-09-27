"""Read-only retrieval inspection contracts, without invented run or generation records."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, SerializerFunctionWrapHandler, model_serializer

from agent.evidence import SHA256, CaptureLimits, EvidenceSnapshot, EvidenceSource, RunConfiguration
from agent.models import QueryObservation, QueryPlan, RetrievalTask
from retrieval.evidence_policy import EvidenceAssessment, assess_evidence


class RetrievalPreviewError(RuntimeError):
    """An explicit preview failure with a safe public message and chained diagnostic cause."""

    def __init__(self, code: str, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


class RetrievalPreviewPlan(BaseModel):
    """The actual clamped plan, without a durable run identifier."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    observation: QueryObservation
    tasks: list[RetrievalTask]
    rationale_trace: list[str]


class RetrievalPreview(BaseModel):
    """Exact prepared evidence, not an answer, claim assessment, or saved run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    plan: RetrievalPreviewPlan
    configuration: RunConfiguration
    sources: list[EvidenceSource] = Field(max_length=50)
    context: str
    context_sha256: SHA256
    context_format: Literal["chunk-id-title-text-v1"] = "chunk-id-title-text-v1"
    capture_limits: CaptureLimits
    evidence_assessment: EvidenceAssessment | None = None

    @model_serializer(mode="wrap")
    def serialize_assessment(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        """Omission preserves the original preview shape when no minimum was requested."""
        fields: dict[str, object] = handler(self)
        if self.evidence_assessment is None:
            fields.pop("evidence_assessment", None)
        return fields

    @classmethod
    def from_preparation(
        cls, plan: QueryPlan, configuration: RunConfiguration, snapshot: EvidenceSnapshot
    ) -> Self:
        """Project shared context capture into an inspection-only response."""
        if len(snapshot.sources) > configuration.max_source_docs:
            raise ValueError("Prepared evidence exceeds the effective source limit.")
        return cls(
            plan=RetrievalPreviewPlan.model_validate(plan.model_dump(exclude={"run_id"})),
            configuration=configuration,
            sources=snapshot.sources,
            context=snapshot.request.context,
            context_sha256=snapshot.context_sha256,
            context_format=snapshot.context_format,
            capture_limits=snapshot.limits,
            evidence_assessment=assess_evidence(plan.observation.evidence_policy, snapshot.sources),
        )
