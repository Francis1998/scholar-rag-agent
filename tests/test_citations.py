"""Tests for citation grounding and hallucination guard."""

from agent.models import Claim
from retrieval.citations import CitationGrounder
from retrieval.models import Chunk
from retrieval.sparse import tokenize


def test_tokenize_drops_punctuation_only_tokens() -> None:
    """Punctuation-only tokens must not survive as empty-string terms."""
    assert tokenize("alpha ( ) beta") == ["alpha", "beta"]
    assert "" not in tokenize("gamma - delta")


def test_grounder_ignores_punctuation_only_token_overlap() -> None:
    """A claim sharing only spaced punctuation with a chunk must stay ungrounded.

    Punctuation-only tokens previously collapsed to an empty-string term that
    any two texts containing punctuation shared, silently grounding otherwise
    unsupported claims and bypassing the hallucination guard.
    """
    chunk = Chunk(
        chunk_id="c1",
        document_id="d1",
        title="Evidence",
        text="Hybrid retrieval ( RRF ) improves answers.",
        source="fixture",
    )
    answer = CitationGrounder().ground(
        answer_text="Cold fusion was confirmed.",
        claims=[Claim(text="Cold fusion ( ) confirmed", chunk_ids=["c1"])],
        retrieved_chunks=[chunk],
    )
    assert answer.ungrounded is True
    assert answer.citations == []


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
