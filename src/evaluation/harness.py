"""Offline RAG evaluation harness for scholarly retrieval pipelines.

Inspired by RAGAS / PaperQA evaluation loops and BEIR-style retrieval metrics.
Computes deterministic hit-rate@k, recall@k, generated-answer faithfulness,
and independent reference coverage without network calls. The text metrics
are lexical proxies, not truth or entailment checks.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from retrieval.models import SearchResult

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "was",
        "with",
    }
)


def _tokens(text: str) -> set[str]:
    return {token for token in _TOKEN_PATTERN.findall(text.lower()) if token not in _STOPWORDS}


def _lexical_coverage(text: str, results: Sequence[SearchResult]) -> float:
    needed = _tokens(text)
    if not needed:
        return 0.0
    evidence = _tokens(" ".join(f"{item.chunk.title} {item.chunk.text}" for item in results))
    covered = needed & evidence
    score = len(covered) / len(needed)
    return float(score) if math.isfinite(score) else 0.0


@dataclass(frozen=True)
class EvalCase:
    """One case with gold chunk ids and an optional reference answer.

    ``gold_answer`` is used only for reference coverage, never faithfulness.
    """

    case_id: str
    query: str
    relevant_chunk_ids: frozenset[str]
    gold_answer: str | None = None


@dataclass(frozen=True)
class CaseScore:
    """Per-case retrieval, generated-answer faithfulness, and reference coverage."""

    case_id: str
    hit_at_k: float
    recall_at_k: float
    faithfulness: float | None
    retrieved_chunk_ids: tuple[str, ...]
    reference_coverage: float | None = None


@dataclass(frozen=True)
class HarnessReport:
    """Aggregate report with independent means for the optional text metrics.

    Missing values are excluded; supplied empty answers contribute zero.
    """

    cases: tuple[CaseScore, ...]
    mean_hit_at_k: float
    mean_recall_at_k: float
    mean_faithfulness: float | None
    k: int
    mean_reference_coverage: float | None = None


RetrieveFn = Callable[[str, int], list[SearchResult]]


class EvaluationHarness:
    """Run offline retrieval (+ optional faithfulness) evaluation cases.

    Faithfulness scores generated-answer content tokens against retrieved
    titles and text; reference coverage separately scores ``gold_answer``.
    Both are deterministic lexical proxies, not truth or entailment checks.
    No LLM or network call is required.
    """

    def __init__(self, cases: Sequence[EvalCase]) -> None:
        """Create a harness from evaluation cases.

        Raises:
            ValueError: If ``cases`` is empty or a case has no relevant ids.
        """
        if not cases:
            raise ValueError("cases must be a non-empty sequence")
        normalized: list[EvalCase] = []
        for case in cases:
            if not case.case_id.strip():
                raise ValueError("case_id must be a non-empty string")
            if not case.query.strip():
                raise ValueError("query must be a non-empty string")
            if not case.relevant_chunk_ids:
                raise ValueError(f"case {case.case_id!r} needs relevant_chunk_ids")
            normalized.append(case)
        self._cases = tuple(normalized)

    def evaluate(
        self,
        retrieve_fn: RetrieveFn,
        *,
        k: int = 5,
        answers: dict[str, str] | None = None,
    ) -> HarnessReport:
        """Evaluate ``retrieve_fn`` on all cases using only the first ``k`` results.

        Reference coverage is computed whenever a case has ``gold_answer``,
        independently of whether a generated answer is supplied.

        Args:
            retrieve_fn: Synchronous callable ``(query, k) -> list[SearchResult]``.
            k: Cutoff for hit-rate@k and recall@k.
            answers: Optional map of ``case_id -> generated answer`` for
                faithfulness scoring against retrieved titles and text.
                Missing entries are unscored (``None``); supplied answers
                with no content tokens score zero.

        Raises:
            ValueError: If ``k`` is not a positive integer.
        """
        if k <= 0:
            raise ValueError("k must be a positive integer")
        scores: list[CaseScore] = []
        for case in self._cases:
            results = retrieve_fn(case.query, k)[:k]
            retrieved_ids = tuple(result.chunk.chunk_id for result in results)
            retrieved_set = set(retrieved_ids)
            overlap = retrieved_set & set(case.relevant_chunk_ids)
            hit = 1.0 if overlap else 0.0
            recall = len(overlap) / len(case.relevant_chunk_ids)
            faithfulness: float | None = None
            if answers is not None and case.case_id in answers:
                faithfulness = self.faithfulness(answers[case.case_id], results)
            reference_coverage = (
                _lexical_coverage(case.gold_answer, results)
                if case.gold_answer is not None
                else None
            )
            scores.append(
                CaseScore(
                    case_id=case.case_id,
                    hit_at_k=hit,
                    recall_at_k=recall,
                    faithfulness=faithfulness,
                    retrieved_chunk_ids=retrieved_ids,
                    reference_coverage=reference_coverage,
                )
            )
        mean_hit = sum(item.hit_at_k for item in scores) / len(scores)
        mean_recall = sum(item.recall_at_k for item in scores) / len(scores)
        faith_values = [item.faithfulness for item in scores if item.faithfulness is not None]
        mean_faith = sum(faith_values) / len(faith_values) if faith_values else None
        reference_values = [
            item.reference_coverage for item in scores if item.reference_coverage is not None
        ]
        mean_reference = sum(reference_values) / len(reference_values) if reference_values else None
        return HarnessReport(
            cases=tuple(scores),
            mean_hit_at_k=mean_hit,
            mean_recall_at_k=mean_recall,
            mean_faithfulness=mean_faith,
            k=k,
            mean_reference_coverage=mean_reference,
        )

    @staticmethod
    def faithfulness(
        answer: str,
        results: Sequence[SearchResult],
        gold_answer: str | None = None,
    ) -> float:
        """Return lexical faithfulness of ``answer`` vs retrieved evidence.

        Score is covered unique content tokens / all unique content tokens
        from ``answer`` (0.0 when empty). Evidence includes titles and text.
        This lexical proxy does not measure truth or entailment.

        ``gold_answer`` is accepted only for calling compatibility and ignored.
        Use ``CaseScore.reference_coverage`` for a case's reference-answer score.
        """
        return _lexical_coverage(answer, results)
