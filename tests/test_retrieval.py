"""Tests for hybrid retrieval components."""

import json
import math

import pytest

from ingestion.chunking import TextChunker, stable_id
from retrieval.dense import DenseRetriever
from retrieval.embeddings import HashEmbeddingModel, cosine_similarity
from retrieval.hybrid import HybridRetriever
from retrieval.hyde import HyDEExpander
from retrieval.models import Chunk, Document, SearchResult
from retrieval.rrf import reciprocal_rank_fusion
from retrieval.sparse import BM25Retriever


@pytest.mark.parametrize("chunk_size", [0, -1, True, False, 1.5, 4.0, "private-source", None, []])
def test_text_chunker_rejects_invalid_size_at_construction(chunk_size: object) -> None:
    with pytest.raises(ValueError, match=r"^chunk_size must be a positive integer$"):
        TextChunker(chunk_size=chunk_size, overlap=0)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "overlap", [-1, -2, 4, 5, True, False, 1.5, 0.0, "private-source", None, []]
)
def test_text_chunker_rejects_invalid_overlap_at_construction(overlap: object) -> None:
    with pytest.raises(
        ValueError, match=r"^overlap must be an integer satisfying 0 <= overlap < chunk_size$"
    ):
        TextChunker(chunk_size=4, overlap=overlap)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("chunk_size", "overlap", "expected_texts", "expected_ids"),
    [
        pytest.param(
            1,
            0,
            ["a", " ", "b", " ", "c"],
            [
                "chunk-62d8443997134731",
                "chunk-6fad16ca81845a19",
                "chunk-b1b2ed11fff7750f",
                "chunk-a89df47d7c8bc0d0",
                "chunk-fa0df9b1279fce25",
            ],
            id="one-character",
        ),
        pytest.param(
            3,
            0,
            ["a b", " c"],
            ["chunk-f0a788a8261fde09", "chunk-b37de94e9e466120"],
            id="no-overlap",
        ),
        pytest.param(
            3,
            2,
            ["a b", " b ", "b c"],
            ["chunk-f0a788a8261fde09", "chunk-d53b65ebab119c1f", "chunk-0d5222e7795eb44f"],
            id="maximum-valid-overlap",
        ),
        pytest.param(
            4,
            1,
            ["a b ", " c"],
            ["chunk-ce1286623d0d9cd0", "chunk-b37de94e9e466120"],
            id="partial-final-window",
        ),
        pytest.param(5, 0, ["a b c"], ["chunk-726e000199f295af"], id="exact-length"),
        pytest.param(10**30, 0, ["a b c"], ["chunk-726e000199f295af"], id="no-upper-cap"),
    ],
)
def test_text_chunker_preserves_exact_geometry_and_provenance(
    chunk_size: int, overlap: int, expected_texts: list[str], expected_ids: list[str]
) -> None:
    document = Document(
        document_id="d1",
        title="Paper",
        text=" a\t b\nc ",
        source="fixture",
        metadata={"source_type": "fixture", "chunk_index": "original"},
    )
    chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)
    chunks = chunker.chunk(document)

    assert chunks == [
        Chunk(
            chunk_id=chunk_id,
            document_id="d1",
            title="Paper",
            text=text,
            source="fixture",
            metadata={"source_type": "fixture", "chunk_index": str(index)},
        )
        for index, (text, chunk_id) in enumerate(zip(expected_texts, expected_ids, strict=True))
    ]
    assert chunks[0].text + "".join(chunk.text[overlap:] for chunk in chunks[1:]) == "a b c"
    assert chunker.chunk(document) == chunks
    assert document.text == " a\t b\nc "
    assert document.metadata == {"source_type": "fixture", "chunk_index": "original"}


def test_text_chunker_preserves_default_windows_and_ids() -> None:
    text = "".join(f"{number:04d}" for number in range(400))
    document = Document(document_id="d1", title="Paper", text=text, source="fixture")

    chunks = TextChunker().chunk(document)

    assert chunks == TextChunker(chunk_size=800, overlap=120).chunk(document)
    assert [chunk.text for chunk in chunks] == [text[:800], text[680:1480], text[1360:]]
    assert [chunk.chunk_id for chunk in chunks] == [
        "chunk-79cca48eb7178516",
        "chunk-2b8c0edfe9058d2a",
        "chunk-c13d04c8bdbf0d0a",
    ]
    assert chunks[0].text + "".join(chunk.text[120:] for chunk in chunks[1:]) == text


@pytest.mark.parametrize(
    "document_id", ["paper", "paper:section", 'paper:["\\', "paper:\u7814\u7a76\U0001f52c"]
)
def test_text_chunker_keeps_delimiter_boundaries_distinct(document_id: str) -> None:
    chunker = TextChunker(chunk_size=5, overlap=0)
    first = Document(document_id=f"{document_id}:1", title="First", text="abc", source="fixture")
    second = Document(document_id=document_id, title="Second", text="zzzzz0:abc", source="fixture")
    first_chunks = chunker.chunk(first)
    second_chunks = chunker.chunk(second)

    assert [chunk.text for chunk in first_chunks] == ["abc"]
    assert [chunk.text for chunk in second_chunks] == ["zzzzz", "0:abc"]
    assert {chunk.chunk_id for chunk in first_chunks}.isdisjoint(
        chunk.chunk_id for chunk in second_chunks
    )
    assert chunker.chunk(first) == first_chunks
    assert chunker.chunk(second) == second_chunks


@pytest.mark.parametrize(
    "document_id",
    [
        "paper",
        "doc-api-v2-0123456789abcdef",
        "doc-0123456789abcdef",
        "paper:Note",
        "paper:note",
        " paper:\t",
        "paper:\u00e9",
        "paper:e\u0301",
    ],
)
def test_text_chunker_preserves_exact_identity_fields_and_legacy_ids(document_id: str) -> None:
    document = Document(
        document_id=document_id,
        title="Paper",
        text=' \u7814\u7a76:\u00e9  "\\value"\n',
        source="fixture",
    )
    chunks = TextChunker(chunk_size=5, overlap=0).chunk(document)

    assert "".join(chunk.text for chunk in chunks) == '\u7814\u7a76:\u00e9 "\\value"'
    for index, chunk in enumerate(chunks):
        assert chunk.document_id == document_id
        assert chunk.metadata["chunk_index"] == str(index)
        if ":" in document_id:
            identity = json.dumps(
                [document_id, index, chunk.text], ensure_ascii=False, separators=(",", ":")
            )
            assert chunk.chunk_id == stable_id(identity, "chunk-v2")
            assert chunk.chunk_id != stable_id(identity, "chunk")
        else:
            assert chunk.chunk_id == stable_id(f"{document_id}:{index}:{chunk.text}", "chunk")
    assert TextChunker(chunk_size=5, overlap=0).chunk(document) == chunks


@pytest.mark.parametrize("text", ["", " \t\n "])
def test_text_chunker_preserves_empty_document_behavior(text: str) -> None:
    document = Document(document_id="d1", title="Paper", text=text, source="fixture")
    assert TextChunker().chunk(document) == []
    assert TextChunker(chunk_size=1, overlap=0).chunk(document) == []


@pytest.fixture(
    params=[
        pytest.param(0, id="zero"),
        pytest.param(-1, id="negative"),
        pytest.param(True, id="true"),
        pytest.param(False, id="false"),
        pytest.param(1.0, id="integral-float"),
        pytest.param(1.5, id="fractional-float"),
        pytest.param(math.nan, id="nan"),
        pytest.param(math.inf, id="infinity"),
        pytest.param(-math.inf, id="negative-infinity"),
        pytest.param("64", id="numeric-string"),
        pytest.param("private-dimension-value", id="private-string"),
        pytest.param(None, id="none"),
        pytest.param([], id="list"),
        pytest.param({}, id="dict"),
        pytest.param(object(), id="object"),
        pytest.param(1 + 0j, id="complex"),
    ]
)
def invalid_embedding_dimensions(request: pytest.FixtureRequest) -> object:
    return request.param


def test_hash_embedding_rejects_invalid_dimensions_at_construction(
    invalid_embedding_dimensions: object,
) -> None:
    with pytest.raises(ValueError, match=r"^dimensions must be a positive integer$"):
        HashEmbeddingModel(dimensions=invalid_embedding_dimensions)  # type: ignore[arg-type]


async def test_hash_embedding_rejects_invalid_dimension_reassignment(
    invalid_embedding_dimensions: object,
) -> None:
    embedder = HashEmbeddingModel(dimensions=8)
    retriever = DenseRetriever(embedder)
    chunk = Chunk(
        chunk_id="matching",
        document_id="paper",
        title="Evidence",
        text="Retrieval.",
        source="fixture",
    )
    retriever.add_chunks([chunk])

    with pytest.raises(ValueError, match=r"^dimensions must be a positive integer$"):
        embedder.dimensions = invalid_embedding_dimensions  # type: ignore[assignment]

    assert embedder.dimensions == 8
    assert embedder.embed("retrieval") == [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0]
    assert embedder.embed("") == [0.0] * 8
    assert await retriever.retrieve("retrieval!") == [
        SearchResult(chunk=chunk, score=1.0, retriever="dense")
    ]


@pytest.mark.parametrize("dimensions", [1, 7, 64, 129, 10**30])
def test_hash_embedding_accepts_positive_dimensions_without_an_upper_cap(dimensions: int) -> None:
    embedder = HashEmbeddingModel(dimensions=dimensions)
    assert embedder.dimensions == dimensions
    reassigned = HashEmbeddingModel()
    reassigned.dimensions = dimensions
    assert reassigned.dimensions == dimensions


def test_hash_embedding_preserves_exact_default_vectors() -> None:
    embedder = HashEmbeddingModel()
    expected = [0.0] * 64
    expected[0] = -0.4472135954999579
    expected[53] = -0.8944271909999159

    assert embedder.dimensions == 64
    assert embedder.embed("machine machine learning") == expected
    assert embedder.embed("(Machine), MACHINE! 'learning'.") == expected
    assert embedder.embed("machine machine learning") == expected
    assert HashEmbeddingModel(dimensions=64).embed("machine machine learning") == expected
    assert sum(value**2 for value in expected) == pytest.approx(1.0)


@pytest.mark.parametrize("dimensions", [1, 7, 64, 129])
@pytest.mark.parametrize("text", ["", " \t\n ", ".,;:()[]{}!?\"'"])
def test_hash_embedding_preserves_empty_token_vectors(dimensions: int, text: str) -> None:
    assert HashEmbeddingModel(dimensions=dimensions).embed(text) == [0.0] * dimensions


def test_hash_embedding_preserves_cancelling_collision_vectors() -> None:
    assert HashEmbeddingModel(dimensions=8).embed("alpha gamma") == [0.0] * 8


def test_hash_embedding_preserves_positive_dimension_reassignment() -> None:
    embedder = HashEmbeddingModel(dimensions=1)
    assert embedder.embed("retrieval") == [-1.0]

    embedder.dimensions = 8

    assert embedder.dimensions == 8
    assert embedder.embed("retrieval.") == [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0]
    assert embedder.embed(" \t\n ") == [0.0] * 8
    assert HashEmbeddingModel().dimensions == 64


@pytest.mark.parametrize("dimensions", [None, 1, 7, 64, 129])
async def test_hash_embedding_preserves_dense_retrieval(dimensions: int | None) -> None:
    embedder = HashEmbeddingModel()
    if dimensions is not None:
        embedder.dimensions = dimensions
    retriever = DenseRetriever(embedder)
    chunks = [
        Chunk(
            chunk_id=chunk_id,
            document_id=chunk_id,
            title="Evidence",
            text=text,
            source="fixture",
        )
        for chunk_id, text in [("other", "alpha"), ("matching", "retrieval."), ("blank", "")]
    ]
    retriever.add_chunks(chunks)

    results = await retriever.retrieve("Retrieval!")

    assert [result.chunk.chunk_id for result in results] == ["matching", "blank", "other"]
    assert [result.score for result in results] == [1.0, 0.0, -1.0 if dimensions == 1 else 0.0]
    assert all(result.retriever == "dense" for result in results)
    assert [result.score for result in await retriever.retrieve("")] == [0.0, 0.0, 0.0]


def test_hash_embedding_is_invariant_to_attached_punctuation() -> None:
    """Trailing punctuation must not change a token's embedding dimension.

    The dense embedder previously split on raw whitespace, so ``retrieval.``
    hashed to a different dimension than ``retrieval`` and the same word was a
    hit in BM25 sparse retrieval but a miss in the dense vector the two are
    fused with. Tokenizing with the shared sparse tokenizer makes the embedding
    of a phrase identical whether or not its terms carry attached punctuation.
    """
    embedder = HashEmbeddingModel()

    plain = embedder.embed("machine learning")
    punctuated = embedder.embed("machine, learning.")

    assert cosine_similarity(plain, punctuated) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "query",
    [
        "retrieval",
        "C++",
        "p53",
        "IL-6",
        "p<0.05",
        "42",
        "-",
        "/",
        "+",
        pytest.param("\u2014", id="em-dash"),
        pytest.param("\u03b2", id="greek-letter"),
        pytest.param("\u7814\u7a76", id="non-latin-text"),
    ],
)
async def test_sparse_and_dense_preserve_existing_tokenization(query: str) -> None:
    chunks = [
        Chunk(
            chunk_id=chunk_id,
            document_id=chunk_id,
            title="Evidence",
            text=text,
            source="fixture",
        )
        for chunk_id, text in [("unrelated", "unrelated"), ("matching", query)]
    ]
    retriever = BM25Retriever()
    retriever.add_chunks(chunks)

    results = await retriever.retrieve(query)

    assert [result.chunk.chunk_id for result in results] == ["matching", "unrelated"]
    assert results[0].score == pytest.approx(math.log(2.0))
    assert results[1].score == 0.0
    vector = HashEmbeddingModel().embed(query)
    assert sum(value**2 for value in vector) == pytest.approx(1.0)


def build_chunks() -> list:
    """Build deterministic fixture chunks."""
    document = Document(
        document_id=stable_id("rag", "doc"),
        title="RAG Methods",
        text=(
            "Hybrid retrieval combines dense embeddings and BM25 sparse search for scientific RAG."
        ),
        source="fixture",
    )
    return TextChunker(chunk_size=240, overlap=0).chunk(document)


async def test_hybrid_retriever_returns_rrf_results() -> None:
    """Hybrid retrieval returns fused RRF results."""
    chunks = build_chunks()
    retriever = HybridRetriever(DenseRetriever(), BM25Retriever(), HyDEExpander())
    retriever.add_chunks(chunks)
    results = await retriever.retrieve("dense BM25 scientific retrieval", limit=3)
    assert results
    assert results[0].retriever == "rrf"


def test_rrf_promotes_shared_results() -> None:
    """RRF gives a shared chunk a positive fused score."""
    chunk = build_chunks()[0]
    fused = reciprocal_rank_fusion(
        [
            [SearchResult(chunk=chunk, score=0.9, retriever="dense")],
            [SearchResult(chunk=chunk, score=2.0, retriever="bm25")],
        ]
    )
    assert fused[0].score > 0
    assert "dense" in fused[0].path
