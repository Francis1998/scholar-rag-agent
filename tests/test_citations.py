"""Tests for citation grounding and hallucination guard."""

from agent.models import Claim
from retrieval.citations import CitationGrounder
from retrieval.models import Chunk


def test_grounder_flags_unsupported_claims() -> None:
    """Unsupported claims are marked as ungrounded."""
    chunk = Chunk(
        chunk_id="c1",
        document_id="d1",
        title="Evidence",
        text="Hybrid retrieval improves grounded scientific answers.",
        source="fixture",
    )
    answer = CitationGrounder().ground(
        answer_text="Quantum teleportation is solved.",
        claims=[Claim(text="Quantum teleportation is solved", chunk_ids=["c1"])],
        retrieved_chunks=[chunk],
    )
    assert answer.ungrounded is True
    assert answer.answer.startswith("[UNGROUNDED]")


def test_grounder_flags_empty_token_claim_as_ungrounded() -> None:
    """A claim that tokenizes to nothing must not be auto-grounded by an attached chunk id."""
    chunk = Chunk(
        chunk_id="c1",
        document_id="d1",
        title="Evidence",
        text="Hybrid retrieval improves grounded scientific answers.",
        source="fixture",
    )
    answer = CitationGrounder().ground(
        answer_text="   ",
        claims=[Claim(text="   ", chunk_ids=["c1"])],
        retrieved_chunks=[chunk],
    )
    assert answer.ungrounded is True
    assert answer.answer.startswith("[UNGROUNDED]")
    assert answer.citations == []


def test_grounder_flags_stopword_only_overlap_as_ungrounded() -> None:
    """Stopword-only overlap must not count as citation grounding."""
    chunk = Chunk(
        chunk_id="c1",
        document_id="d1",
        title="Evidence",
        text="Hybrid retrieval is the baseline approach.",
        source="fixture",
    )
    answer = CitationGrounder().ground(
        answer_text="It is the case.",
        claims=[Claim(text="It is the case", chunk_ids=["c1"])],
        retrieved_chunks=[chunk],
    )
    assert answer.ungrounded is True
    assert answer.citations == []
