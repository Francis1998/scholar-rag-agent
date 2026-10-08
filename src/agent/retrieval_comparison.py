"""Bounded, current-corpus policy inspection through two real generation-free previews."""

import asyncio
import json
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)
from pydantic.json_schema import SkipJsonSchema

from agent.comparison_models import EvidenceComparison, IdentityOverlap, SourceChanges
from agent.evidence import text_digest
from agent.markdown import literal_block
from agent.retrieval_preview import RetrievalPreview, RetrievalPreviewError
from agent.run_comparison import compare_sources
from agent.runner import AgentRunner
from retrieval.evidence_policy import (
    EvidencePolicy,
    assess_evidence,
    ensure_evidence_policy,
    evidence_limit_arguments,
)
from retrieval.scope import DocumentIds, ensure_document_scope, scope_arguments
from storage.paper_collections import (
    CollectionError,
    CollectionId,
    Revision,
    SQLitePaperCollections,
)

MAX_QUERY_CHARACTERS = 500
MAX_RESPONSE_BYTES = 262144
COMPARISON_TIMEOUT_SECONDS = 30
Query = Annotated[
    str, StringConstraints(strict=True, min_length=1, max_length=MAX_QUERY_CHARACTERS)
]
VariantName = Literal["baseline", "candidate"]
_Count = Annotated[int, Field(ge=0, le=50)]
_NOTICES = (
    "Two sequential previews inspect the current corpus, not a frozen corpus or a controlled "
    "experiment. Only requested scope membership and policies are copied before the first await; "
    "concurrent corpus, index, or configuration changes may affect either preview.",
    "Exact chunk/document identities, ranks, scores, provenance, and digests describe differences, "
    "not scientific support, retrieval quality, or an improvement. There is no ground truth here. "
    "Empty/empty identity overlap is null, not perfect agreement.",
    "no_passages means no passages were returned, not an absence of scientific evidence. "
    "An unmet minimum retains the preview and its count assessment; it is not answerability "
    "or a measure of independent evidence.",
    "No fake or live answer generation, saved run, event writes, or corpus writes occur. "
    "Digests identify exact UTF-8 content; they are not signatures.",
    "Review before sharing: queries, full passages, source metadata, paths, and identifiers may "
    "be sensitive. The local API has no authentication or tenant isolation. Save the download "
    "yourself; a later comparison may see different corpus contents.",
)


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class RetrievalVariant(_Model):
    """An empty object uses ordinary preview defaults; only existing policy fields are allowed."""

    evidence_policy: EvidencePolicy | SkipJsonSchema[None] = Field(
        default=None,
        description=(
            "Omit for unchanged preview defaults. Otherwise supply one or more existing "
            "EvidencePolicy fields; an empty policy or explicit null field is invalid."
        ),
    )

    @field_validator("evidence_policy", mode="before")
    @classmethod
    def nonnull_policy(cls, value: object) -> object:
        if value is None or (isinstance(value, dict) and any(v is None for v in value.values())):
            raise ValueError("Omit unconfigured policy fields instead of supplying null.")
        return value


class RetrievalComparisonRequest(_Model):
    """Exactly two named variants share one bounded query and optional, exclusive scope."""

    query: Query
    baseline: RetrievalVariant
    candidate: RetrievalVariant
    document_ids: DocumentIds | SkipJsonSchema[None] = Field(
        default=None,
        description=(
            "1-100 supplied IDs, each 1-128 characters after trimming; deduplicate in input "
            "order. Unknown IDs match no chunks, as in /retrieve. Omit both scope fields "
            "for the full current corpus. Empty or null selections are invalid."
        ),
    )
    collection_id: CollectionId | SkipJsonSchema[None] = Field(
        default=None,
        description=(
            "Mutually exclusive with document_ids. Resolve membership, identity, and revision "
            "in one read transaction before either preview; never reread the collection."
        ),
    )

    @field_validator("query")
    @classmethod
    def readable_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank.")
        value.encode("utf-8")
        return value

    @field_validator("document_ids", "collection_id", mode="before")
    @classmethod
    def nonnull_scope(cls, value: object) -> object:
        if value is None:
            raise ValueError("Scope cannot be null; omit it to search the full corpus.")
        return value

    @model_validator(mode="after")
    def exclusive_scope(self) -> Self:
        if self.document_ids is not None and self.collection_id is not None:
            raise ValueError("Supply either collection_id or document_ids, not both.")
        for identifier in self.document_ids or ():
            identifier.encode("utf-8")
        return self


class RetrievalComparisonError(RuntimeError):
    """Explicit all-or-nothing failure; only the chained cause contains private diagnostics."""

    def __init__(
        self, code: str, message: str, status_code: int = 500, *, variant: VariantName | None = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.variant = variant


class ComparisonLimits(_Model):
    max_query_characters: Literal[500] = 500
    max_document_ids: Literal[100] = 100
    variants: Literal[2] = 2
    max_sources_per_variant: Literal[50] = 50
    max_serialized_bytes: Literal[262144] = 262144
    overall_timeout_seconds: Literal[30] = 30


class CollectionSnapshot(_Model):
    """Identity and revision from the same transaction as the shared document selection."""

    collection_id: CollectionId
    revision: Revision


class ComparedRetrieval(_Model):
    """The complete actual preview, with exact counts and context UTF-8 size."""

    status: Literal["passages_returned", "no_passages"]
    preview: RetrievalPreview
    source_count: _Count
    document_count: _Count
    context_utf8_bytes: int = Field(ge=0, le=262144)


class DocumentComparison(_Model):
    """Distinct document identity counts, never paper independence or relevance."""

    statistics: IdentityOverlap
    added: tuple[str, ...] = Field(max_length=50)
    removed: tuple[str, ...] = Field(max_length=50)
    shared: tuple[str, ...] = Field(max_length=50)


class SourceChangeCounts(_Model):
    """Counts among shared chunk/document pairs; per-pair values remain in evidence.shared."""

    rank_changed: _Count
    text_changed: _Count
    title_changed: _Count
    source_changed: _Count
    metadata_changed: _Count
    score_changed: _Count
    retriever_changed: _Count
    path_changed: _Count
    record_changed: _Count


class RetrievalChanges(_Model):
    evidence_policy_changed: bool
    configuration_changed: bool
    plan_changed: bool
    context_changed: bool
    source_membership_changed: bool
    source_order_changed: bool
    evidence_changed: bool
    evidence_assessment_changed: bool


def _bounded(content: str) -> str:
    if len(content.encode("utf-8")) > MAX_RESPONSE_BYTES:
        raise RetrievalComparisonError(
            "comparison_too_large",
            "Comparison exceeds the serialized download limit; reduce the source scope or "
            "configured source limit. No partial comparison was returned.",
            413,
        )
    return content


def _json_block(fields: dict[str, object]) -> str:
    return literal_block(
        json.dumps(fields, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2), "json"
    )


class RetrievalComparison(_Model):
    """Portable current-corpus inspection, with no invented completion or generation identity."""

    schema_version: Literal["1.0"] = "1.0"
    kind: Literal["retrieval_policy_comparison"] = "retrieval_policy_comparison"
    query: Query
    document_ids: DocumentIds | None
    collection: CollectionSnapshot | None
    baseline: ComparedRetrieval
    candidate: ComparedRetrieval
    evidence: EvidenceComparison
    documents: DocumentComparison
    source_changes: SourceChangeCounts
    changes: RetrievalChanges
    any_changes: bool
    context_bytes_delta: int = Field(ge=-262144, le=262144)
    limits: ComparisonLimits = Field(default_factory=ComparisonLimits)
    notices: tuple[str, ...] = _NOTICES

    def to_json(self) -> str:
        """Bound the complete serialized UTF-8 output, including metadata and notices."""
        return _bounded(
            json.dumps(
                self.model_dump(mode="json"),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )

    def to_markdown(self) -> str:
        """Keep every untrusted string in literal blocks, not headings, links, or tables."""
        sections = [
            "# Retrieval policy comparison\n",
            "Current-corpus inspection before generation; not a quality evaluation.\n",
            "## Interpretation and privacy\n",
            literal_block("\n".join(self.notices)),
            "## Shared query\n",
            literal_block(self.query),
            "## Baseline\n",
            _json_block(self.baseline.model_dump(mode="json")),
            "## Candidate\n",
            _json_block(self.candidate.model_dump(mode="json")),
            "## Exact differences, shared scope, and limits\n",
            _json_block(
                self.model_dump(mode="json", exclude={"query", "baseline", "candidate", "notices"}),
            ),
        ]
        return _bounded("\n".join(sections))


def _inspection(
    raw: RetrievalPreview, query: str, scope: tuple[str, ...] | None, policy: EvidencePolicy | None
) -> ComparedRetrieval:
    preview = RetrievalPreview.model_validate(raw.model_dump(mode="json"))
    sources = preview.sources
    ensure_document_scope(scope, (source.chunk.document_id for source in sources))
    ensure_evidence_policy(policy, sources)
    if (
        preview.plan.observation.original_query != query.strip()
        or preview.plan.observation.document_ids != scope
        or preview.plan.observation.evidence_policy != policy
        or preview.evidence_assessment != assess_evidence(policy, sources)
        or len(sources) > preview.configuration.max_source_docs
        or [source.rank for source in sources] != list(range(1, len(sources) + 1))
        or len({source.chunk.chunk_id for source in sources}) != len(sources)
        or any(text_digest(source.chunk.text) != source.text_sha256 for source in sources)
        or preview.context
        != "\n".join(
            f"[{source.chunk.chunk_id}] {source.chunk.title}: {source.chunk.text}"
            for source in sources
        )
        or preview.context_sha256 != text_digest(preview.context)
    ):
        raise ValueError(
            "Preview does not match the requested scope, policy, or captured evidence."
        )
    return ComparedRetrieval(
        status="passages_returned" if sources else "no_passages",
        preview=preview,
        source_count=len(sources),
        document_count=len({source.chunk.document_id for source in sources}),
        context_utf8_bytes=len(preview.context.encode("utf-8")),
    )


def _documents(baseline: RetrievalPreview, candidate: RetrievalPreview) -> DocumentComparison:
    left = dict.fromkeys(source.chunk.document_id for source in baseline.sources)
    right = dict.fromkeys(source.chunk.document_id for source in candidate.sources)
    shared, union = left.keys() & right.keys(), left.keys() | right.keys()
    return DocumentComparison(
        statistics=IdentityOverlap(
            baseline_count=len(left),
            candidate_count=len(right),
            shared_count=len(shared),
            added_count=len(right.keys() - left.keys()),
            removed_count=len(left.keys() - right.keys()),
            union_count=len(union),
            identity_jaccard=len(shared) / len(union) if union else None,
        ),
        added=tuple(key for key in right if key not in left),
        removed=tuple(key for key in left if key not in right),
        shared=tuple(key for key in left if key in right),
    )


class RetrievalComparator:
    """Compose preview without changing the runner, its policies, or its shared components."""

    def __init__(self, runner: AgentRunner, collections: SQLitePaperCollections) -> None:
        self._runner = runner
        self._collections = collections

    async def compare(self, request: RetrievalComparisonRequest) -> RetrievalComparison:
        """Copy inputs before awaiting, then return both inspections or one sanitized failure."""
        request = RetrievalComparisonRequest.model_validate(request.model_dump(exclude_none=True))
        loop = asyncio.get_running_loop()
        deadline = loop.time() + COMPARISON_TIMEOUT_SECONDS
        variant: VariantName | None = None

        def check_deadline() -> None:
            if loop.time() >= deadline:
                raise TimeoutError("Retrieval comparison deadline exceeded.")

        try:
            async with asyncio.timeout_at(deadline):
                scope = request.document_ids
                collection = None
                if request.collection_id is not None:
                    with self._collections.resolved_collection(request.collection_id) as (_, saved):
                        scope = saved.document_ids
                        collection = CollectionSnapshot(
                            collection_id=saved.collection_id, revision=saved.revision
                        )
                inspections = []
                variants: tuple[tuple[VariantName, RetrievalVariant], ...] = (
                    ("baseline", request.baseline),
                    ("candidate", request.candidate),
                )
                for name, options in variants:
                    variant = name
                    check_deadline()
                    policy = options.evidence_policy
                    arguments = (
                        evidence_limit_arguments(
                            policy.max_chunks_per_document,
                            policy.min_evidence_documents,
                            policy.near_duplicate_threshold,
                        )
                        if policy is not None
                        else {}
                    )
                    raw = await self._runner.preview(
                        request.query, **scope_arguments(scope), **arguments
                    )
                    check_deadline()
                    inspections.append(_inspection(raw, request.query, scope, policy))
                baseline, candidate = inspections
                left, right = baseline.preview, candidate.preview
                evidence = compare_sources(left.sources, right.sources)
                counts = SourceChangeCounts(
                    **{
                        field: sum(getattr(pair.changes, field) for pair in evidence.shared)
                        for field in SourceChanges.model_fields
                    }
                )
                membership_changed = bool(evidence.added or evidence.removed)
                changes = RetrievalChanges(
                    evidence_policy_changed=(
                        left.plan.observation.evidence_policy
                        != right.plan.observation.evidence_policy
                    ),
                    configuration_changed=left.configuration != right.configuration,
                    plan_changed=left.plan != right.plan,
                    context_changed=left.context != right.context,
                    source_membership_changed=membership_changed,
                    source_order_changed=membership_changed or counts.rank_changed > 0,
                    evidence_changed=membership_changed or counts.record_changed > 0,
                    evidence_assessment_changed=left.evidence_assessment
                    != right.evidence_assessment,
                )
                result = RetrievalComparison(
                    query=request.query,
                    document_ids=scope,
                    collection=collection,
                    baseline=baseline,
                    candidate=candidate,
                    evidence=evidence,
                    documents=_documents(left, right),
                    source_changes=counts,
                    changes=changes,
                    any_changes=any(changes.model_dump().values()),
                    context_bytes_delta=candidate.context_utf8_bytes - baseline.context_utf8_bytes,
                )
                result.to_json()
                result.to_markdown()
                check_deadline()
                return result
        except (RetrievalComparisonError, CollectionError):
            raise
        except RetrievalPreviewError as exc:
            raise RetrievalComparisonError(
                exc.code,
                "Retrieval preview failed; no partial comparison was returned.",
                exc.status_code,
                variant=variant,
            ) from exc
        except TimeoutError as exc:
            raise RetrievalComparisonError(
                "comparison_timeout",
                "Retrieval comparison exceeded its overall deadline; no partial comparison "
                "was returned.",
                504,
                variant=variant,
            ) from exc
        except Exception as exc:
            raise RetrievalComparisonError(
                "comparison_failed",
                "Retrieval comparison failed; no partial comparison was returned.",
                variant=variant,
            ) from exc
