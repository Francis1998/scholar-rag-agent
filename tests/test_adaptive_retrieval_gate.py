"""Tests for deterministic Adaptive-RAG retrieve-or-skip gating."""

from retrieval.adaptive_retrieval_gate import (
    AdaptiveRetrievalAction,
    AdaptiveRetrievalGate,
)


def test_skips_chitchat_and_meta_system_queries() -> None:
    gate = AdaptiveRetrievalGate()

    hello = gate.decide("Hello!")
    thanks = gate.decide("thank you")
    meta = gate.decide("What can you do?")
    identity = gate.decide("Who are you?")

    assert hello.action is AdaptiveRetrievalAction.SKIP
    assert "hello" in hello.matched_cues
    assert thanks.action is AdaptiveRetrievalAction.SKIP
    assert meta.action is AdaptiveRetrievalAction.SKIP
    assert "what can you do" in meta.matched_cues
    assert identity.action is AdaptiveRetrievalAction.SKIP


def test_retrieves_knowledge_seeking_scholarly_queries() -> None:
    gate = AdaptiveRetrievalGate()

    papers = gate.decide("Which papers compare graph neural networks for molecules?")
    methods = gate.decide("What methods improve sparse retrieval ranking?")
    evidence = gate.decide("Is there evidence that MMR reduces redundancy?")

    assert papers.action is AdaptiveRetrievalAction.RETRIEVE
    assert "papers" in papers.matched_cues
    assert methods.action is AdaptiveRetrievalAction.RETRIEVE
    assert "methods" in methods.matched_cues
    assert evidence.action is AdaptiveRetrievalAction.RETRIEVE
    assert "evidence" in evidence.matched_cues


def test_content_terms_without_cues_still_retrieve() -> None:
    decision = AdaptiveRetrievalGate().decide("transformer latency tradeoffs")

    assert decision.action is AdaptiveRetrievalAction.RETRIEVE
    assert decision.matched_cues
    assert "content terms" in decision.reason


def test_empty_or_stopword_only_queries_skip() -> None:
    gate = AdaptiveRetrievalGate()

    empty = gate.decide("   ")
    stopwords = gate.decide("the and of")

    assert empty.action is AdaptiveRetrievalAction.SKIP
    assert empty.matched_cues == ()
    assert stopwords.action is AdaptiveRetrievalAction.SKIP
    assert "no content terms" in stopwords.reason


def test_knowledge_cues_override_meta_phrasing() -> None:
    decision = AdaptiveRetrievalGate().decide(
        "How does this work for PubMed literature review of diabetes studies?"
    )

    assert decision.action is AdaptiveRetrievalAction.RETRIEVE
    assert "literature" in decision.matched_cues or "studies" in decision.matched_cues
