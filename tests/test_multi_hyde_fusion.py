"""Tests for deterministic Multi-HyDE retrieval fusion."""

from copy import deepcopy

import pytest

from llm.base import BaseLLMAdapter
from llm.schemas import LLMRequest, LLMResponse
from retrieval.models import Chunk, SearchResult
from retrieval.multi_hyde_fusion import MultiHydeFusion


def _result(chunk_id: str, score: float = 1.0) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id=f"doc-{chunk_id}",
            title=f"Paper {chunk_id}",
            text="Scientific evidence.",
            source="test",
        ),
        score=score,
        retriever="dense",
        path=["embedding"],
    )


class RecordingRetriever:
    """Return controlled rankings while recording expanded queries."""

    def __init__(self, rankings: list[list[SearchResult]]) -> None:
        self.queries: list[str] = []
        self._rankings = rankings

    async def retrieve(self, query: str, limit: int = 10) -> list[SearchResult]:
        self.queries.append(query)
        return self._rankings[len(self.queries) - 1][:limit]


class StubLLM(BaseLLMAdapter):
    """Return one configured completion per request."""

    provider_name = "stub"

    def __init__(self, texts: list[str]) -> None:
        self.requests: list[LLMRequest] = []
        self._texts = iter(texts)

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(text=next(self._texts), raw_provider=self.provider_name)


async def test_generates_distinct_reproducible_local_hypotheses() -> None:
    retriever = RecordingRetriever([])
    fusion = MultiHydeFusion(retriever, num_hypotheses=4)

    first = await fusion.hypothetical_abstracts(" graph   neural retrieval ")
    second = await fusion.hypothetical_abstracts(" graph neural retrieval ")

    assert first == second
    assert len(first) == len(set(first)) == 4
    assert first[0].startswith("Background perspective 1:")
    assert first[1].startswith("Methods perspective 2:")
    assert all("graph neural retrieval" in abstract for abstract in first)


async def test_retrieves_each_expansion_and_promotes_shared_results_with_rrf() -> None:
    retriever = RecordingRetriever(
        [
            [_result("shared"), _result("background")],
            [_result("methods"), _result("shared")],
            [_result("shared"), _result("findings")],
        ]
    )
    fusion = MultiHydeFusion(retriever, num_hypotheses=3, rank_constant=10)

    fused = await fusion.retrieve("retrieval evaluation", limit=2)

    assert len(retriever.queries) == 3
    assert all(
        query.startswith("retrieval evaluation\nHypothetical abstract:")
        for query in retriever.queries
    )
    assert [result.chunk.chunk_id for result in fused] == ["shared", "methods"]
    assert fused[0].retriever == "rrf"
    assert fused[0].score == pytest.approx(1 / 11 + 1 / 12 + 1 / 11)
    assert fused[0].path == ["multi_hyde:1", "multi_hyde:2", "multi_hyde:3"]


async def test_duplicate_returning_retriever_cannot_inflate_hypothesis_support() -> None:
    repeated = _result("repeated", score=100)
    first_shared = _result("shared", score=0.1)
    first_shared.chunk.metadata["origin"] = "first-hypothesis"
    later_shared = _result("shared", score=99)
    later_shared.chunk.metadata["origin"] = "later-hypothesis"
    rankings = [
        [repeated, repeated, repeated, first_shared, _result("background-tail")],
        [_result("methods"), later_shared, later_shared],
    ]
    before = deepcopy(rankings)
    retriever = RecordingRetriever(rankings)
    fusion = MultiHydeFusion(retriever, num_hypotheses=2)

    fused = await fusion.retrieve("retrieval evaluation", limit=5)

    assert rankings == before
    assert len(retriever.queries) == 2
    assert all(
        query.startswith("retrieval evaluation\nHypothetical abstract:")
        for query in retriever.queries
    )
    assert [result.chunk.chunk_id for result in fused] == [
        "shared",
        "repeated",
        "methods",
        "background-tail",
    ]
    assert [result.score for result in fused] == pytest.approx(
        [1 / 64 + 1 / 62, 1 / 61, 1 / 61, 1 / 65]
    )
    assert [result.path for result in fused] == [
        ["multi_hyde:1", "multi_hyde:2"],
        ["multi_hyde:1"],
        ["multi_hyde:2"],
        ["multi_hyde:1"],
    ]
    assert all(result.retriever == "rrf" for result in fused)
    assert fused[0].chunk == first_shared.chunk
    assert fused[0].chunk is not first_shared.chunk


async def test_uses_optional_llm_and_falls_back_when_a_completion_is_blank() -> None:
    llm = StubLLM(["A generated abstract.", "  "])
    retriever = RecordingRetriever([])
    fusion = MultiHydeFusion(retriever, num_hypotheses=2, llm=llm)

    abstracts = await fusion.hypothetical_abstracts("protein folding")

    assert abstracts[0] == "A generated abstract."
    assert abstracts[1].startswith("Methods perspective 2:")
    assert [request.context for request in llm.requests] == [
        "Multi-HyDE variant 1 of 2",
        "Multi-HyDE variant 2 of 2",
    ]


async def test_blank_queries_and_non_positive_limits_do_not_call_retriever() -> None:
    retriever = RecordingRetriever([])
    fusion = MultiHydeFusion(retriever)

    assert await fusion.retrieve("   ") == []
    assert await fusion.retrieve("valid query", limit=0) == []
    assert retriever.queries == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("num_hypotheses", 0),
        ("num_hypotheses", True),
        ("rank_constant", -1),
        ("rank_constant", 2.5),
    ],
)
def test_rejects_invalid_positive_integer_configuration(field: str, value: object) -> None:
    kwargs = {field: value}
    with pytest.raises(ValueError, match="positive integer"):
        MultiHydeFusion(RecordingRetriever([]), **kwargs)  # type: ignore[arg-type]
