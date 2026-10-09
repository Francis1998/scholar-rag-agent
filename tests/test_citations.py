"""Tests for citation grounding and hallucination guard."""

import string

import pytest

from agent.models import Claim
from retrieval.citations import CitationGrounder
from retrieval.models import Chunk
from retrieval.sparse import meaningful_terms, tokenize


@pytest.mark.parametrize(
    "answer_text",
    ["Graph retrieval connects passages.", "", " \t\n"],
    ids=["nonempty-answer", "empty-answer", "whitespace-answer"],
)
@pytest.mark.parametrize(
    "chunk_text",
    [None, "Ocean warming persists.", "Graph retrieval connects passages."],
    ids=["no-sources", "irrelevant-source", "overlapping-source"],
)
def test_grounder_flags_missing_claims(answer_text: str, chunk_text: str | None) -> None:
    claims: list[Claim] = []
    chunks = (
        [
            Chunk(
                chunk_id="c1",
                document_id="d1",
                title="Evidence",
                text=chunk_text,
                source="fixture",
                metadata={"kind": "test"},
            )
        ]
        if chunk_text is not None
        else []
    )
    original_chunks = [chunk.model_copy(deep=True) for chunk in chunks]

    answer = CitationGrounder().ground(answer_text, claims, chunks)

    assert claims == []
    assert chunks == original_chunks
    assert answer.claims == []
    assert answer.citations == []
    assert answer.ungrounded is True
    assert answer.answer == f"[UNGROUNDED] {answer_text}"
    assert answer.warnings == ["No claims were available for grounding."]


def test_tokenize_drops_tokens_emptied_by_punctuation_stripping() -> None:
    """Limited punctuation stripping drops empty terms, not all symbol tokens."""
    assert tokenize("alpha ( ) beta") == ["alpha", "beta"]
    assert tokenize("gamma - delta") == ["gamma", "-", "delta"]


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(" ".join(string.punctuation), id="separate-ascii-punctuation"),
        pytest.param(string.punctuation, id="combined-ascii-punctuation"),
        pytest.param("\u2014 \u2026 \u201c\u201d \u2212 \u00d7 \u00b1", id="unicode-symbols"),
        pytest.param("\U0001f52c \u0301 \u200d", id="emoji-and-nonalphanumeric-marks"),
        pytest.param("the - and / is \u2014", id="stopwords-and-symbols"),
        pytest.param(" \t\n", id="whitespace"),
    ],
)
def test_meaningful_terms_excludes_symbol_only_tokens(text: str) -> None:
    assert meaningful_terms(text) == set()


@pytest.mark.parametrize(
    "symbol",
    [
        pytest.param("-", id="hyphen"),
        pytest.param("/", id="slash"),
        pytest.param("+", id="plus"),
        pytest.param("_", id="underscore"),
        pytest.param("\u2014", id="em-dash"),
        pytest.param("\u2026", id="unicode-ellipsis"),
        pytest.param("\u221e", id="infinity"),
        pytest.param("\u201c\u201d", id="curly-quotes"),
        pytest.param("\U0001f52c", id="emoji"),
    ],
)
@pytest.mark.parametrize(
    "claim_template",
    ["Cold fusion {symbol} confirmed.", "{symbol}", "the {symbol} and"],
    ids=["unrelated-claim", "symbol-only-claim", "stopwords-and-symbols"],
)
def test_grounder_ignores_symbol_only_overlap(symbol: str, claim_template: str) -> None:
    chunk = Chunk(
        chunk_id="c1",
        document_id="d1",
        title="Evidence",
        text=f"Graph retrieval {symbol} connects passages.",
        source="fixture",
    )
    claim_text = claim_template.format(symbol=symbol)

    answer = CitationGrounder().ground(
        answer_text=claim_text,
        claims=[Claim(text=claim_text, chunk_ids=["c1"])],
        retrieved_chunks=[chunk],
    )

    assert answer.citations == []
    assert answer.claims == [Claim(text=claim_text, chunk_ids=[], grounded=False)]
    assert answer.ungrounded is True
    assert answer.answer == f"[UNGROUNDED] {claim_text}"
    assert answer.warnings == ["One or more claims lacked retrieved chunk support."]


@pytest.mark.parametrize(
    "term",
    [
        "C++",
        "P53",
        "IL-6",
        "p<0.05",
        "42",
        "0",
        "-1.5",
        "1e-3",
        pytest.param("\u03b2", id="greek-letter"),
        pytest.param("TNF-\u03b1", id="greek-scientific-term"),
        pytest.param("\u7814\u7a76", id="non-latin-text"),
        pytest.param("\u0661\u0662", id="unicode-digits"),
        pytest.param("e\u0301", id="letter-with-combining-mark"),
    ],
)
def test_grounder_preserves_scientific_lexical_terms(term: str) -> None:
    assert meaningful_terms(f"The ({term}), and {term}!") == {term.lower()}
    chunk = Chunk(
        chunk_id="c1",
        document_id="d1",
        title="Evidence",
        text=f"Measurements concern {term.lower()}.",
        source="fixture",
    )
    claim_text = f"{term} matters."

    answer = CitationGrounder().ground(
        answer_text=claim_text,
        claims=[Claim(text=claim_text, chunk_ids=["c1"])],
        retrieved_chunks=[chunk],
    )

    assert answer.claims == [Claim(text=claim_text, chunk_ids=["c1"], grounded=True)]
    assert [citation.chunk_id for citation in answer.citations] == ["c1"]
    assert answer.ungrounded is False
    assert answer.answer == claim_text
    assert answer.warnings == []


def test_grounder_keeps_only_lexically_supported_citations() -> None:
    chunks = [
        Chunk(
            chunk_id="unrelated",
            document_id="d1",
            title="Other evidence",
            text="Ocean - warming persists.",
            source="fixture",
        ),
        Chunk(
            chunk_id="supported",
            document_id="d2",
            title="Graph evidence",
            text="Graph retrieval - connects passages.",
            source="fixture",
        ),
        Chunk(
            chunk_id="also-supported",
            document_id="d3",
            title="More graph evidence",
            text="Graph retrieval follows entities.",
            source="fixture",
        ),
    ]
    claims = [
        Claim(text="Cold fusion - confirmed.", chunk_ids=["unrelated", "supported"], grounded=True),
        Claim(
            text="Graph retrieval works.",
            chunk_ids=["also-supported", "unrelated", "supported"],
        ),
    ]
    original_claims = [claim.model_copy(deep=True) for claim in claims]
    original_chunks = [chunk.model_copy(deep=True) for chunk in chunks]

    answer = CitationGrounder().ground("Draft answer.", claims, chunks)

    assert answer.claims == [
        Claim(text=claims[0].text, chunk_ids=[], grounded=False),
        Claim(text=claims[1].text, chunk_ids=["also-supported", "supported"], grounded=True),
    ]
    assert [citation.chunk_id for citation in answer.citations] == ["also-supported", "supported"]
    assert answer.ungrounded is True
    assert answer.answer == "[UNGROUNDED] Draft answer."
    assert answer.warnings == ["One or more claims lacked retrieved chunk support."]
    assert claims == original_claims
    assert chunks == original_chunks


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


@pytest.mark.parametrize("claim_text", ["", "   ", " \t\n"], ids=["empty", "spaces", "whitespace"])
def test_grounder_flags_empty_token_claim_as_ungrounded(claim_text: str) -> None:
    """A claim that tokenizes to nothing must not be auto-grounded by an attached chunk id."""
    chunk = Chunk(
        chunk_id="c1",
        document_id="d1",
        title="Evidence",
        text="Hybrid retrieval improves grounded scientific answers.",
        source="fixture",
    )
    answer = CitationGrounder().ground(
        answer_text=claim_text,
        claims=[Claim(text=claim_text, chunk_ids=["c1"])],
        retrieved_chunks=[chunk],
    )
    assert answer.ungrounded is True
    assert answer.answer == f"[UNGROUNDED] {claim_text}"
    assert answer.claims == [Claim(text=claim_text, chunk_ids=[], grounded=False)]
    assert answer.citations == []
    assert answer.warnings == ["One or more claims lacked retrieved chunk support."]


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
