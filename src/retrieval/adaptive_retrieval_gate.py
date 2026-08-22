"""Deterministic pre-retrieval Adaptive-RAG retrieve-or-skip gate."""

from dataclasses import dataclass
from enum import StrEnum

from retrieval.sparse import meaningful_terms, tokenize

_SKIP_EXACT = frozenset(
    {
        "hi",
        "hello",
        "hey",
        "thanks",
        "thank you",
        "thx",
        "ok",
        "okay",
        "bye",
        "goodbye",
        "good morning",
        "good evening",
        "how are you",
        "who are you",
        "what are you",
        "what's your name",
        "what is your name",
        "help",
        "help me",
    }
)

_SKIP_PHRASES = (
    "who are you",
    "what can you do",
    "what do you do",
    "how do you work",
    "how does this work",
    "what is this system",
    "what is this tool",
    "tell me about yourself",
    "introduce yourself",
    "your capabilities",
    "your name",
)

_KNOWLEDGE_CUES = frozenset(
    {
        "paper",
        "papers",
        "study",
        "studies",
        "research",
        "literature",
        "evidence",
        "finding",
        "findings",
        "method",
        "methods",
        "dataset",
        "datasets",
        "compare",
        "comparison",
        "versus",
        "vs",
        "cite",
        "citation",
        "doi",
        "pubmed",
        "arxiv",
        "hypothesis",
        "effect",
        "effects",
        "outcome",
        "outcomes",
        "review",
        "survey",
        "meta-analysis",
    }
)


class AdaptiveRetrievalAction(StrEnum):
    """Whether corpus retrieval should run for the current user query."""

    RETRIEVE = "RETRIEVE"
    SKIP = "SKIP"


@dataclass(frozen=True)
class AdaptiveRetrievalDecision:
    """Inspectable retrieve-or-skip decision with matched lexical cues."""

    action: AdaptiveRetrievalAction
    reason: str
    matched_cues: tuple[str, ...]


class AdaptiveRetrievalGate:
    """Decide RETRIEVE vs SKIP before any corpus lookup.

    Inspired by Adaptive RAG and Self-RAG retrieval decisions, this gate runs
    *before* retrieval. It is distinct from :class:`SelfRagReflectionGate`,
    which evaluates evidence quality after documents are fetched.
    """

    def decide(self, query: str) -> AdaptiveRetrievalDecision:
        """Return RETRIEVE for knowledge-seeking queries, otherwise SKIP."""
        normalized = " ".join(query.lower().split()).strip(" ?!.")
        if not normalized:
            return AdaptiveRetrievalDecision(
                action=AdaptiveRetrievalAction.SKIP,
                reason="Empty query does not require corpus retrieval.",
                matched_cues=(),
            )

        if normalized in _SKIP_EXACT:
            return AdaptiveRetrievalDecision(
                action=AdaptiveRetrievalAction.SKIP,
                reason="Query matches a chitchat or meta greeting cue.",
                matched_cues=(normalized,),
            )

        phrase_hits = tuple(phrase for phrase in _SKIP_PHRASES if phrase in normalized)
        content_terms = meaningful_terms(normalized)
        knowledge_hits = tuple(sorted(term for term in content_terms if term in _KNOWLEDGE_CUES))

        if phrase_hits and not knowledge_hits:
            return AdaptiveRetrievalDecision(
                action=AdaptiveRetrievalAction.SKIP,
                reason="Query asks about the system rather than literature evidence.",
                matched_cues=phrase_hits,
            )

        if knowledge_hits:
            return AdaptiveRetrievalDecision(
                action=AdaptiveRetrievalAction.RETRIEVE,
                reason="Query contains knowledge-seeking scholarly cues.",
                matched_cues=knowledge_hits,
            )

        if content_terms:
            return AdaptiveRetrievalDecision(
                action=AdaptiveRetrievalAction.RETRIEVE,
                reason="Query contains content terms that may need corpus evidence.",
                matched_cues=tuple(sorted(content_terms)[:5]),
            )

        tokens = tokenize(normalized)
        return AdaptiveRetrievalDecision(
            action=AdaptiveRetrievalAction.SKIP,
            reason="Query has no content terms for retrieval.",
            matched_cues=tuple(tokens[:3]),
        )
