"""Tests for deterministic evidence-span alignment."""

from retrieval.evidence_span_align import EvidenceSpan, EvidenceSpanAligner
from retrieval.models import Chunk, SearchResult


def _result(chunk_id: str, text: str) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id=f"doc-{chunk_id}",
            title=f"Paper {chunk_id}",
            text=text,
            source="test",
        ),
        score=0.8,
        retriever="rrf",
    )


def test_aligns_case_insensitively_with_exact_half_open_offsets() -> None:
    text = "Neural retrieval improves neural ranking."
    aligner = EvidenceSpanAligner()

    spans = aligner.align_text("NEURAL retrieval", text)

    assert spans == (
        EvidenceSpan(start=0, end=6, term="neural"),
        EvidenceSpan(start=7, end=16, term="retrieval"),
        EvidenceSpan(start=26, end=32, term="neural"),
    )
    assert [text[span.start : span.end] for span in spans] == [
        "Neural",
        "retrieval",
        "neural",
    ]


def test_matches_whole_terms_and_excludes_stopwords() -> None:
    aligner = EvidenceSpanAligner()
    text = "The RAG result should not match ragged fragments in the text."

    spans = aligner.align_text("the rag in", text)

    assert spans == (EvidenceSpan(start=4, end=7, term="rag"),)


def test_preserves_result_order_and_includes_empty_alignments() -> None:
    results = [
        _result("match", "Graph-based retrieval is reproducible."),
        _result("miss", "A clinical outcome was measured."),
    ]

    alignments = EvidenceSpanAligner().align("graph-based retrieval", results)

    assert [alignment.result.chunk.chunk_id for alignment in alignments] == ["match", "miss"]
    assert alignments[0].spans == (
        EvidenceSpan(start=0, end=11, term="graph-based"),
        EvidenceSpan(start=12, end=21, term="retrieval"),
    )
    assert alignments[1].spans == ()
    assert alignments[0].result is results[0]


def test_supports_unicode_terms_and_custom_stopwords() -> None:
    text = "β-catenin and Café biomarkers; café replication."
    aligner = EvidenceSpanAligner(stopwords={"biomarkers"})

    spans = aligner.align_text("β-catenin café biomarkers", text)

    assert [span.term for span in spans] == ["β-catenin", "café", "café"]
    assert [text[span.start : span.end] for span in spans] == ["β-catenin", "Café", "café"]


def test_empty_inputs_produce_stable_empty_outputs() -> None:
    aligner = EvidenceSpanAligner()
    result = _result("paper", "Graph retrieval.")

    assert aligner.align_text("", result.chunk.text) == ()
    assert aligner.align_text("the and", result.chunk.text) == ()
    assert aligner.align("graph", []) == []
