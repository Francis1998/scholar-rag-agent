"""Offline regression coverage for adaptive reranking and empty evidence previews."""

import builtins
import hashlib
import sys
from pathlib import Path
from types import ModuleType
from typing import NoReturn

import httpx
import pytest
from scripts.demo_evidence_export import offline_settings

from api.dependencies import AppContainer
from retrieval.evidence_policy import EvidenceAssessment
from retrieval.models import Chunk, Document, SearchResult
from retrieval.rerank import AdaptiveReranker


class PredictorStub:
    def __init__(self) -> None:
        self.calls: list[list[tuple[str, str]]] = []
        self.scores = [0.75, 0.25, 0.75]
        self.error: Exception | None = None

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.calls.append(pairs)
        if not pairs:
            raise ValueError("cross-encoder received an empty batch")
        if self.error is not None:
            raise self.error
        return self.scores


@pytest.fixture
def predictor(monkeypatch: pytest.MonkeyPatch) -> PredictorStub:
    predictor = PredictorStub()
    module = ModuleType("sentence_transformers")

    def cross_encoder(model_name: str) -> PredictorStub:
        assert model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        return predictor

    monkeypatch.setattr(module, "CrossEncoder", cross_encoder, raising=False)
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return predictor


@pytest.fixture
def search_results() -> list[SearchResult]:
    return [
        SearchResult(
            chunk=Chunk(
                chunk_id=f"chunk-{index}",
                document_id=f"paper-{index}",
                title="Synthetic paper",
                text=text,
                source="synthetic:rerank",
                metadata={"section": "abstract"},
            ),
            score=score,
            retriever=retriever,
            path=["seed"],
        )
        for index, (text, score, retriever) in enumerate(
            [
                ("Unrelated data.", 0.5, "bm25"),
                ("GraphRAG retrieval.", 0.25, "rrf"),
                ("GraphRAG retrieval methods.", 0.25, "multihop"),
            ]
        )
    ]


def test_cross_encoder_missing_dependency_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)

    with pytest.raises(ModuleNotFoundError, match="sentence_transformers") as caught:
        AdaptiveReranker(use_cross_encoder=True)

    assert caught.value.name == "sentence_transformers"


@pytest.mark.parametrize(
    "error",
    [OSError("model weights unavailable"), ValueError("invalid model configuration")],
    ids=["weights-unavailable", "invalid-model"],
)
def test_cross_encoder_constructor_errors_propagate(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    module = ModuleType("sentence_transformers")
    calls: list[str] = []

    def cross_encoder(model_name: str) -> NoReturn:
        calls.append(model_name)
        raise error

    monkeypatch.setattr(module, "CrossEncoder", cross_encoder, raising=False)
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    with pytest.raises(type(error), match=str(error)) as caught:
        AdaptiveReranker(use_cross_encoder=True, model_name="synthetic:broken-model")

    assert caught.value is error
    assert calls == ["synthetic:broken-model"]


@pytest.mark.parametrize("use_default", [True, False], ids=["default", "explicit-false"])
async def test_lexical_reranking_does_not_import_cross_encoder(
    monkeypatch: pytest.MonkeyPatch, search_results: list[SearchResult], use_default: bool
) -> None:
    original_import = builtins.__import__
    import_attempts: list[str] = []

    def guarded_import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> ModuleType:
        if name == "sentence_transformers" or name.startswith("sentence_transformers."):
            import_attempts.append(name)
            raise AssertionError("Lexical reranking must not import cross-encoder dependencies")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    reranker = AdaptiveReranker() if use_default else AdaptiveReranker(use_cross_encoder=False)

    reranked = await reranker.rerank("GraphRAG retrieval", search_results)

    assert reranker._model is None
    assert [result.score for result in reranked] == [1.25, 1.25, 0.5]
    assert all(result.retriever == "lexical_rerank" for result in reranked)
    assert import_attempts == []


@pytest.mark.parametrize("use_cross_encoder", [False, True], ids=["lexical", "cross-encoder"])
async def test_empty_results_skip_prediction(
    predictor: PredictorStub, use_cross_encoder: bool
) -> None:
    reranker = AdaptiveReranker(use_cross_encoder=use_cross_encoder)
    assert reranker._model is (predictor if use_cross_encoder else None)

    assert await reranker.rerank("no matching paper", []) == []
    assert predictor.calls == []


@pytest.mark.parametrize(
    ("use_cross_encoder", "order", "scores", "retriever"),
    [
        (False, [1, 2, 0], [1.25, 1.25, 0.5], "lexical_rerank"),
        (True, [0, 2, 1], [0.75, 0.75, 0.25], "cross_encoder"),
    ],
    ids=["default-lexical", "cross-encoder"],
)
async def test_nonempty_results_preserve_scores_stable_order_and_provenance(
    predictor: PredictorStub,
    search_results: list[SearchResult],
    use_cross_encoder: bool,
    order: list[int],
    scores: list[float],
    retriever: str,
) -> None:
    originals = [result.model_copy(deep=True) for result in search_results]
    reranker = AdaptiveReranker(use_cross_encoder=True) if use_cross_encoder else AdaptiveReranker()
    assert reranker._model is (predictor if use_cross_encoder else None)

    reranked = await reranker.rerank("GraphRAG retrieval", search_results)

    assert reranked == [
        SearchResult(
            chunk=search_results[index].chunk,
            score=score,
            retriever=retriever,
            path=[*search_results[index].path, search_results[index].retriever],
        )
        for index, score in zip(order, scores, strict=True)
    ]
    assert search_results == originals
    expected_pairs = [("GraphRAG retrieval", result.chunk.text) for result in search_results]
    assert predictor.calls == ([expected_pairs] if use_cross_encoder else [])


async def test_nonempty_prediction_errors_propagate(
    predictor: PredictorStub, search_results: list[SearchResult]
) -> None:
    error = ValueError("cross-encoder prediction failed")
    predictor.error = error
    reranker = AdaptiveReranker(use_cross_encoder=True)
    assert reranker._model is predictor

    with pytest.raises(ValueError, match="cross-encoder prediction failed") as caught:
        await reranker.rerank("GraphRAG retrieval", search_results)

    assert caught.value is error
    assert predictor.calls == [
        [("GraphRAG retrieval", result.chunk.text) for result in search_results]
    ]


def denied(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("Empty previews must not generate, write events, or use HTTP.")


@pytest.mark.parametrize(
    "document_ids", [None, ("not-ingested",)], ids=["empty-corpus", "unknown-scope"]
)
@pytest.mark.parametrize("minimum", [None, 1], ids=["no-minimum", "minimum-one"])
async def test_cross_encoder_empty_preview_without_generation_or_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    predictor: PredictorStub,
    document_ids: tuple[str, ...] | None,
    minimum: int | None,
) -> None:
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", denied)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", denied)
    container = AppContainer(offline_settings(tmp_path / "rerank.sqlite3"))
    if document_ids is not None:
        container.ingestion_pipeline.ingest_documents(
            [
                Document(
                    document_id="ingested-paper",
                    title="Synthetic paper",
                    text="GraphRAG retrieval connects local evidence.",
                    source="synthetic:rerank",
                )
            ]
        )
    reranker = AdaptiveReranker(use_cross_encoder=True)
    assert reranker._model is predictor
    monkeypatch.setattr(container.runner._executor, "_reranker", reranker)
    monkeypatch.setattr(container.llm, "generate", denied)
    monkeypatch.setattr(container.runner._executor, "answer", denied)
    monkeypatch.setattr(container.event_log, "append_event", denied)
    monkeypatch.setattr(container.event_log, "append_transition", denied)

    preview = await container.runner.preview(
        "GraphRAG retrieval", document_ids=document_ids, min_evidence_documents=minimum
    )

    assert preview.sources == []
    assert preview.context == ""
    assert preview.context_sha256 == hashlib.sha256(b"").hexdigest()
    assert preview.plan.observation.document_ids == document_ids
    if minimum is None:
        assert preview.evidence_assessment is None
    else:
        assert preview.evidence_assessment == EvidenceAssessment(
            required_documents=minimum, observed_documents=0, passed=False
        )
    assert predictor.calls == []
    assert container.event_log.list_events() == []
