"""Safe JSON, literal-Markdown and offline HTML downloads for completed evidence bundles."""

import json
from typing import Literal

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

from agent.evidence import EvidenceBundle, text_digest
from agent.evidence_html import HTML_CSP, render_html
from agent.markdown import literal_block as _literal
from api.dependencies import AppContainer
from retrieval.evidence_policy import assess_evidence
from storage.evidence_export import EvidenceExportError

router = APIRouter()


def _json_block(value: BaseModel) -> str:
    return _literal(
        json.dumps(value.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, indent=2),
        "json",
    )


def _policy_description(bundle: EvidenceBundle) -> str:
    policy = bundle.plan.observation.evidence_policy
    description = (
        "No per-document quota requested. Older records may omit evidence_policy."
        if policy is None or policy.max_chunks_per_document is None
        else f"max_chunks_per_document={policy.max_chunks_per_document}\n"
        "Applied after reranking within the bounded candidate pool; no extra retrieval.\n"
        "Fewer passages may remain; distinct-paper coverage is not guaranteed."
    )
    if policy is not None and policy.near_duplicate_threshold is not None:
        description += (
            f"\nnear_duplicate_threshold={policy.near_duplicate_threshold}\n"
            "Lexical Jaccard collapse after reranking, before any per-document quota and "
            "minimum assessment; no extra retrieval. Survivors retain exact chunks and scores.\n"
            "Provenance: near_duplicate_collapse, then diversity_cap_gate if requested.\n"
            "Similar passages may contain important differences. Lexical overlap is not "
            "semantic equivalence, paper identity, evidence independence, or scientific truth."
        )
    assessment = assess_evidence(policy, bundle.snapshot.sources)
    if assessment is not None:
        description += (
            f"\nmin_evidence_documents={assessment.required_documents}\n"
            f"Final captured context: observed_documents={assessment.observed_documents}; "
            f"passed={assessment.passed}.\n"
            "Document counts do not establish relevance, independence, or scientific answerability."
        )
    return description


def render_markdown(bundle: EvidenceBundle) -> str:
    """Render the saved bundle; dynamic content is never a link, heading, or HTML."""
    sections = [
        "# Research evidence bundle\n",
        "Version 1.0. A saved operational record, not a scientific endorsement.\n",
        "## Review before sharing\n",
        _literal("\n".join(bundle.warnings)),
        "## Run\n",
        _literal(
            f"Run: {bundle.run_id}\nAgent: {bundle.agent_id}\nStatus: {bundle.status}\n"
            f"Completed (UTC): {bundle.completed_at}\n"
            f"Context SHA-256: {bundle.snapshot.context_sha256}"
        ),
        "## Original query\n",
        _literal(bundle.query),
        "## Plan and operational rationale\n",
        _json_block(bundle.plan),
        "## Document scope\n",
        _literal(
            "All documents (unscoped). Older records may omit document_ids."
            if bundle.plan.observation.document_ids is None
            else "Selected document IDs (unknown IDs may match no chunks):\n"
            + "\n".join(bundle.plan.observation.document_ids)
        ),
        "## Per-paper evidence policy\n",
        _literal(_policy_description(bundle)),
        "## Effective runtime configuration\n",
        _json_block(bundle.configuration),
        "## Answer\n",
        _literal(bundle.answer.answer),
        "### Answer warnings\n",
        _literal("\n".join(bundle.answer.warnings) or "No answer warnings were recorded."),
        "## Provider and model provenance\n",
        _json_block(bundle.generation),
        "## Claims and evidence associations\n",
        "The grounded flag records token overlap only; resolved IDs do not prove a claim.\n",
    ]
    for claim, claim_link in zip(bundle.answer.claims, bundle.claim_evidence, strict=True):
        sections.extend(
            [f"### Claim {claim_link.claim_number}\n", _json_block(claim), _json_block(claim_link)]
        )
    sections.append("## Citations\n")
    for citation, citation_link in zip(
        bundle.answer.citations, bundle.citation_evidence, strict=True
    ):
        sections.extend(
            [
                f"### Citation {citation_link.citation_number}\n",
                _json_block(citation),
                _json_block(citation_link),
            ]
        )
    sections.append("## Exact final-context evidence\n")
    for source in bundle.snapshot.sources:
        metadata = source.model_dump(mode="json")
        metadata["chunk"].pop("text")
        sections.extend(
            [
                f"### Evidence {source.rank}\n",
                _literal(
                    json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2), "json"
                ),
                _literal(source.chunk.text),
            ]
        )
    if not bundle.snapshot.sources:
        sections.append("The saved model context contained no source passages.\n")
    sections.extend(
        [
            "## Exact generation request\n",
            _json_block(bundle.snapshot.request),
            "### Capture limits\n",
            _json_block(bundle.snapshot.limits),
            "## Ordered operational events\n",
            "Snapshot and generation payload references point to the sections above. "
            "Unknown event payloads are explicitly omitted for privacy.\n",
        ]
    )
    sections.extend(_json_block(event) for event in bundle.events)
    return "\n".join(sections)


@router.get(
    "/runs/{run_id}/export",
    response_model=EvidenceBundle,
    responses={
        200: {
            "content": {
                "text/markdown": {"schema": {"type": "string"}},
                "text/html": {"schema": {"type": "string"}},
            }
        },
        404: {"description": "Unknown run"},
        409: {"description": "Unexportable saved evidence or unrepresentable HTML text"},
        413: {"description": "Complete HTML exceeds the rendered UTF-8 byte limit"},
    },
)
def export_run(
    request: Request, run_id: str, format: Literal["json", "markdown", "html"] = "json"
) -> Response:
    """Download a completed run without retrieval, generation, or live provider calls."""
    container: AppContainer = request.app.state.container
    try:
        bundle = container.evidence_exporter.export(run_id)
        if format == "html":
            content = render_html(bundle)
            extension = "html"
            media_type = "text/html"
        elif format == "markdown":
            content = render_markdown(bundle)
            extension = "md"
            media_type = "text/markdown"
        else:
            content = (
                json.dumps(
                    bundle.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, indent=2
                )
                + "\n"
            )
            extension = "json"
            media_type = "application/json"
    except EvidenceExportError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": str(exc)},
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        ) from exc
    filename = f"evidence-{text_digest(run_id)[:16]}.{extension}"
    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
        "Cache-Control": "no-store",
        "X-Content-Type-Options": "nosniff",
    }
    if format == "html":
        headers["Content-Security-Policy"] = HTML_CSP + "; frame-ancestors 'none'"
    return Response(
        content=content,
        media_type=media_type,
        headers=headers,
    )
