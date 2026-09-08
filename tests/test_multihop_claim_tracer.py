"""Tests for MultiHopClaimTracer."""

import pytest

from retrieval.models import Chunk, SearchResult
from retrieval.multihop_claim_tracer import MultiHopClaimTracer


def _chunk(
    chunk_id: str,
    document_id: str,
    title: str,
    text: str,
    *,
    doi: str = "",
) -> Chunk:
    metadata: dict[str, str] = {}
    if doi:
        metadata["doi"] = doi
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        title=title,
        text=text,
        source="test",
        metadata=metadata,
    )


def test_rejects_invalid_threshold() -> None:
    with pytest.raises(ValueError, match="support_threshold"):
        MultiHopClaimTracer(support_threshold=-0.1)


def test_rejects_invalid_max_span_hops() -> None:
    with pytest.raises(ValueError, match="max_span_hops"):
        MultiHopClaimTracer(max_span_hops=0)


def test_empty_claim_is_ungrounded() -> None:
    trace = MultiHopClaimTracer().trace("   ", [])
    assert trace.claim == ""
    assert trace.hops == ()
    assert trace.grounded is False


def test_builds_claim_span_document_hops() -> None:
    evidence = [
        SearchResult(
            chunk=_chunk(
                "c1",
                "d1",
                "GraphRAG Survey",
                "Graph retrieval improves multi-hop reasoning over papers. See [1].",
                doi="10.1000/graph",
            ),
            score=0.9,
            retriever="hybrid",
        )
    ]
    claim = "Graph retrieval improves multi-hop reasoning."
    trace = MultiHopClaimTracer(support_threshold=0.3).trace(claim, evidence)
    types = [hop.hop_type for hop in trace.hops]
    assert types[0] == "claim"
    assert "supporting_span" in types
    assert "cited_document" in types
    assert trace.grounded is True
    assert trace.support_score > 0.0
    docs = [hop for hop in trace.hops if hop.hop_type == "cited_document"]
    assert any(hop.document_id == "d1" for hop in docs)
    assert any("citation 1" in hop.label for hop in docs)


def test_ungrounded_when_evidence_mismatches() -> None:
    evidence = [_chunk("c1", "d1", "Optics", "Laser cavity modes are stable under cooling.")]
    trace = MultiHopClaimTracer(support_threshold=0.5).trace(
        "Transformer attention scales quadratically with sequence length.",
        evidence,
    )
    assert trace.grounded is False
    assert any(hop.hop_type == "claim" for hop in trace.hops)


def test_trace_many_and_markdown_mentions_gap() -> None:
    evidence = [
        _chunk(
            "c1",
            "d1",
            "RAG",
            "Dense retrieval retrieves relevant passages for generation.",
        )
    ]
    traces = MultiHopClaimTracer().trace_many(
        ["Dense retrieval retrieves relevant passages.", "Unrelated claim about cats."],
        evidence,
    )
    assert len(traces) == 2
    markdown = traces[0].to_markdown()
    assert "PaperQA claim-verification UI gap" in markdown
    assert "GPT-5.5" in markdown
    assert "Claude Sonnet 4.6" in markdown
    assert "Gemini 3.x" in markdown
    assert "Kimi K2" in markdown


def test_docstring_distinguishes_siblings_and_models() -> None:
    doc = MultiHopClaimTracer.__doc__ or ""
    assert "GPT-5.5" in doc
    assert "Claude Sonnet 4.6" in doc
    assert "Gemini 3.x" in doc
    assert "Kimi K2" in doc
    assert "gate" in doc.lower()
    assert "PaperQA" in doc
