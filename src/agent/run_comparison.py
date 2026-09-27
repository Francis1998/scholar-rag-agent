"""Pure exact comparisons of validated, frozen evidence bundles."""

import json
from urllib.parse import quote

from pydantic import BaseModel, JsonValue

from agent.comparison_models import (
    PREVIEW_CHARACTERS,
    AnswerSummary,
    ComparedRun,
    EvidenceComparison,
    IdentityOverlap,
    RunChanges,
    RunComparison,
    SavedGenerationIdentity,
    SharedSourceComparison,
    SourceChanges,
    SourceSummary,
    TextSummary,
)
from agent.evidence import EvidenceBundle, EvidenceSource, text_digest


def _canonical(value: BaseModel | JsonValue) -> str:
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _digest(value: BaseModel | JsonValue) -> str:
    return text_digest(_canonical(value))


def _text(text: str) -> TextSummary:
    return TextSummary(
        preview=text[:PREVIEW_CHARACTERS],
        truncated=len(text) > PREVIEW_CHARACTERS,
        characters=len(text),
        utf8_bytes=len(text.encode("utf-8")),
        sha256=text_digest(text),
    )


def _associations(bundle: EvidenceBundle) -> dict[str, JsonValue]:
    return {
        "proposed": [list(ids) for ids in bundle.generation.claim_chunk_ids],
        "claims": [link.model_dump(mode="json") for link in bundle.claim_evidence],
        "citations": [link.model_dump(mode="json") for link in bundle.citation_evidence],
        "citation_owners": [citation.document_id for citation in bundle.answer.citations],
    }


def _warnings(bundle: EvidenceBundle) -> dict[str, JsonValue]:
    return {"answer": list(bundle.answer.warnings), "bundle": list(bundle.warnings)}


def _run(bundle: EvidenceBundle) -> ComparedRun:
    addressable = "/" not in bundle.run_id and bundle.run_id not in {"", ".", ".."}
    policy = bundle.plan.observation.evidence_policy
    return ComparedRun(
        run_id=bundle.run_id,
        agent_id=bundle.agent_id,
        completed_at=bundle.completed_at,
        export_url=f"/runs/{quote(bundle.run_id, safe='')}/export" if addressable else None,
        query=_text(bundle.query),
        document_ids=bundle.plan.observation.document_ids,
        evidence_policy=policy.model_copy(deep=True) if policy is not None else None,
        configuration=bundle.configuration.model_copy(deep=True),
        generation=SavedGenerationIdentity(
            provider=bundle.generation.provider,
            model_name=bundle.generation.model_name,
            task_type=bundle.generation.task_type,
        ),
        context_sha256=bundle.snapshot.context_sha256,
        request_sha256=_digest(bundle.snapshot.request),
        answer=AnswerSummary(
            text=_text(bundle.answer.answer),
            record_sha256=_digest(bundle.answer),
            claims_sha256=_digest(
                [claim.model_dump(mode="json") for claim in bundle.answer.claims]
            ),
            citations_sha256=_digest(
                [citation.model_dump(mode="json") for citation in bundle.answer.citations]
            ),
            citation_associations_sha256=_digest(_associations(bundle)),
            warnings_sha256=_digest(_warnings(bundle)),
            ungrounded=bundle.answer.ungrounded,
            claim_count=len(bundle.answer.claims),
            grounded_claim_count=sum(claim.grounded for claim in bundle.answer.claims),
            citation_count=len(bundle.answer.citations),
            answer_warning_count=len(bundle.answer.warnings),
            bundle_warning_count=len(bundle.warnings),
        ),
    )


def _source(source: EvidenceSource) -> SourceSummary:
    return SourceSummary(
        chunk_id=source.chunk.chunk_id,
        document_id=source.chunk.document_id,
        rank=source.rank,
        title=_text(source.chunk.title),
        text_sha256=source.text_sha256,
        source_sha256=text_digest(source.chunk.source),
        metadata_sha256=_digest(dict(source.chunk.metadata)),
        path_sha256=_digest(list(source.path)),
        retriever=_text(source.retriever),
        score=source.score,
        record_sha256=_digest(source),
    )


def _identity(source: EvidenceSource) -> tuple[str, str]:
    return source.chunk.chunk_id, source.chunk.document_id


def _shared(baseline: EvidenceSource, candidate: EvidenceSource) -> SharedSourceComparison:
    return SharedSourceComparison(
        baseline=_source(baseline),
        candidate=_source(candidate),
        rank_delta=candidate.rank - baseline.rank,
        changes=SourceChanges(
            rank_changed=baseline.rank != candidate.rank,
            text_changed=baseline.chunk.text != candidate.chunk.text,
            title_changed=baseline.chunk.title != candidate.chunk.title,
            source_changed=baseline.chunk.source != candidate.chunk.source,
            metadata_changed=baseline.chunk.metadata != candidate.chunk.metadata,
            score_changed=_canonical(baseline.score) != _canonical(candidate.score),
            retriever_changed=baseline.retriever != candidate.retriever,
            path_changed=baseline.path != candidate.path,
            record_changed=_canonical(baseline) != _canonical(candidate),
        ),
    )


def _evidence(baseline: EvidenceBundle, candidate: EvidenceBundle) -> EvidenceComparison:
    left = {_identity(source): source for source in baseline.snapshot.sources}
    right = {_identity(source): source for source in candidate.snapshot.sources}
    shared = left.keys() & right.keys()
    union = left.keys() | right.keys()
    left_owners = {source.chunk.chunk_id: source.chunk.document_id for source in left.values()}
    right_owners = {source.chunk.chunk_id: source.chunk.document_id for source in right.values()}
    return EvidenceComparison(
        statistics=IdentityOverlap(
            baseline_count=len(left),
            candidate_count=len(right),
            shared_count=len(shared),
            added_count=len(right.keys() - left.keys()),
            removed_count=len(left.keys() - right.keys()),
            union_count=len(union),
            identity_jaccard=len(shared) / len(union) if union else None,
        ),
        added=[_source(source) for key, source in right.items() if key not in left],
        removed=[_source(source) for key, source in left.items() if key not in right],
        shared=[_shared(source, right[key]) for key, source in left.items() if key in right],
        reassigned_chunk_ids=sorted(
            chunk_id
            for chunk_id in left_owners.keys() & right_owners.keys()
            if left_owners[chunk_id] != right_owners[chunk_id]
        ),
    )


def compare_bundles(baseline: EvidenceBundle, candidate: EvidenceBundle) -> RunComparison:
    """Compare validated saved objects without I/O, mutation, or semantic evaluation."""
    left_scope = baseline.plan.observation.document_ids
    right_scope = candidate.plan.observation.document_ids
    left_sources = baseline.snapshot.sources
    right_sources = candidate.snapshot.sources
    left_ids = [_identity(source) for source in left_sources]
    right_ids = [_identity(source) for source in right_sources]
    changes = RunChanges(
        query_changed=baseline.query != candidate.query,
        document_scope_changed=left_scope != right_scope,
        document_scope_membership_changed=(
            (set(left_scope) if left_scope is not None else None)
            != (set(right_scope) if right_scope is not None else None)
        ),
        evidence_policy_changed=(
            baseline.plan.observation.evidence_policy != candidate.plan.observation.evidence_policy
        ),
        configuration_changed=baseline.configuration != candidate.configuration,
        provider_changed=baseline.generation.provider != candidate.generation.provider,
        model_changed=baseline.generation.model_name != candidate.generation.model_name,
        task_type_changed=baseline.generation.task_type != candidate.generation.task_type,
        request_changed=baseline.snapshot.request != candidate.snapshot.request,
        context_changed=baseline.snapshot.request.context != candidate.snapshot.request.context,
        source_membership_changed=set(left_ids) != set(right_ids),
        source_order_changed=left_ids != right_ids,
        evidence_changed=(
            [_canonical(source) for source in left_sources]
            != [_canonical(source) for source in right_sources]
        ),
        answer_changed=baseline.answer != candidate.answer,
        answer_text_changed=baseline.answer.answer != candidate.answer.answer,
        claim_text_changed=(
            [claim.text for claim in baseline.answer.claims]
            != [claim.text for claim in candidate.answer.claims]
        ),
        grounding_changed=(
            baseline.answer.ungrounded != candidate.answer.ungrounded
            or [claim.grounded for claim in baseline.answer.claims]
            != [claim.grounded for claim in candidate.answer.claims]
        ),
        citations_changed=baseline.answer.citations != candidate.answer.citations,
        citation_associations_changed=_associations(baseline) != _associations(candidate),
        warnings_changed=_warnings(baseline) != _warnings(candidate),
    )
    notices = [
        "Exact saved-data inspection, not model-quality scoring, semantic validation, "
        "a scientific finding, or a controlled A/B test.",
        "Identity overlap counts chunk/document pairs, not evidence support or retrieval quality. "
        "Empty/empty overlap is null.",
        "Grounding flags record token overlap, not entailment. "
        "Matching digests are not signatures.",
        "Review before sharing: query/answer/title previews and identifiers may be sensitive. "
        "Full passages, metadata, citations, and warnings remain in each evidence export.",
    ]
    if changes.query_changed or changes.document_scope_changed:
        notices.append(
            "The saved queries or document scopes differ; do not attribute differences "
            "to the model alone or interpret this as a fair model A/B test."
        )
    if changes.evidence_policy_changed:
        notices.append(
            "The saved per-paper evidence policies differ, even if the contexts match; "
            "do not attribute differences to the model alone or treat this as a controlled "
            "A/B test."
        )
    if "fake" in (baseline.generation.provider, candidate.generation.provider):
        notices.append(
            "At least one saved run used the fake provider; its output is a plumbing "
            "demonstration, not a scientific finding."
        )
    return RunComparison(
        baseline=_run(baseline),
        candidate=_run(candidate),
        changes=changes,
        any_changes=any(changes.model_dump().values()),
        evidence=_evidence(baseline, candidate),
        notices=notices,
    )
