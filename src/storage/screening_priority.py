"""Order papers still needing a human decision with a local TF-IDF + Naive Bayes ranker.

The ranker learns only from current human include (relevant) and exclude (irrelevant)
decisions. It never records, changes or suggests a decision: it orders the queue a person
still screens. Everything is deterministic, offline and pure Python (no NumPy/scikit-learn).
"""

import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from retrieval.scope import MAX_DOCUMENT_IDS
from storage.document_catalog import (
    MAX_TITLE_CHARACTERS,
    DocumentCatalogError,
    Identity,
    _prefix,
)
from storage.paper_collections import (
    CollectionError,
    CollectionId,
    CollectionName,
    PaperCollection,
    Revision,
)
from storage.paper_screening import (
    ScreeningCounts,
    ScreeningQueue,
    ScreeningStatus,
    SQLitePaperScreening,
)

MAX_FEATURE_TEXT_CHARACTERS = 4000
MIN_TERM_CHARACTERS = 2
MAX_TERM_CHARACTERS = 64
TOP_TERMS = 5
NAIVE_BAYES_ALPHA = 1.0
MAX_PRIORITY_RESPONSE_BYTES = 524_288
# Equal log-odds up to this many decimal places count as ties, broken by document ID.
TIE_DECIMALS = 12
_DISPLAY_DECIMALS = 6
ModelStatus = Literal["ready", "needs_include_and_exclude", "no_training_terms"]
PriorityOrdering = Literal["most_likely_relevant_first", "collection_order"]
PendingStatus = Literal["unscreened", "stale", "unsure"]
_PENDING: tuple[ScreeningStatus, ...] = ("unscreened", "stale")
_STOP_WORD_TEXT = """
    about above after again against all also am an and any are as at be because been before
    being below between both but by can could did do does doing down during each either else
    etc ever every few for from further had has have having he her here hers herself him
    himself his how however if in into is it its itself just may me might more most must my
    myself neither no nor not now of off on once only or other our ours ourselves out over own
    per same shall she should since so some such than that the their theirs them themselves
    then there therefore these they this those though through thus to too under until up upon
    us very via was we were what when where whether which while who whom whose why will with
    within without would yet you your yours yourself yourselves
    """
STOP_WORDS = frozenset(_STOP_WORD_TEXT.split())
_TOKEN = re.compile(r"[^\W_]+")
_NOTES = (
    "Ordering only: no decision is recorded, changed or suggested. Every paper still needs "
    "your own explicit include, exclude or unsure decision; a score is not a label.",
    "Training labels are current include (relevant) and current exclude (irrelevant) "
    "decisions only. Unsure, stale and unscreened members are never training labels.",
    "Scores order papers within this collection. They are not calibrated probabilities, "
    "recall estimates, quality ratings or evidence of scientific relevance.",
    "No stopping signal is reported: decisions keep per-paper revisions and server wall-clock "
    "times, not a reliable global decision sequence, so consecutive excludes since the last "
    "include cannot be counted honestly. When to stop screening remains your judgment.",
    "One read snapshot of membership, decisions, titles and text prefixes; nothing is cached "
    "or saved. A new decision changes the next ordering, so reread it before choosing.",
)


def tokenize(text: str) -> list[str]:
    """NFKC-casefold, split on non-letter/digit runs, drop short/long/numeric/stop words."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return [
        token
        for token in _TOKEN.findall(normalized)
        if MIN_TERM_CHARACTERS <= len(token) <= MAX_TERM_CHARACTERS
        and not token.isdigit()
        and token not in STOP_WORDS
    ]


def feature_text(title: str, text: str) -> str:
    """The bounded title plus stored-text prefix that every member contributes."""
    return title[:MAX_TITLE_CHARACTERS] + "\n" + text[:MAX_FEATURE_TEXT_CHARACTERS]


def tfidf_vectors(documents: Sequence[Sequence[str]]) -> list[dict[str, float]]:
    """Sublinear tf, smoothed idf over these documents, then an L2 norm per document.

    tf = 1 + ln(count); idf = ln((1 + N) / (1 + df)) + 1; weight = tf * idf / ||tf * idf||.
    """
    counts = [Counter(tokens) for tokens in documents]
    frequency = Counter(term for count in counts for term in count)
    total = len(counts)
    idf = {term: math.log((1 + total) / (1 + df)) + 1.0 for term, df in frequency.items()}
    vectors = []
    for count in counts:
        weights = {term: (1.0 + math.log(n)) * idf[term] for term, n in sorted(count.items())}
        norm = math.sqrt(math.fsum(value * value for value in weights.values()))
        vectors.append({term: value / norm for term, value in weights.items()} if norm else {})
    return vectors


def train_naive_bayes(
    relevant: Sequence[Mapping[str, float]],
    irrelevant: Sequence[Mapping[str, float]],
    *,
    alpha: float = NAIVE_BAYES_ALPHA,
) -> dict[str, float]:
    """Return w_t = ln(theta_rel,t) - ln(theta_irr,t) for terms seen in any labeled paper.

    Each class's term mass is the mean TF-IDF weight over its labeled papers (class-balanced),
    theta_c,t = (mass_c,t + alpha) / (sum_t' mass_c,t' + alpha * |V|).
    """
    if not relevant or not irrelevant:
        raise ValueError("Naive Bayes needs at least one relevant and one irrelevant paper.")
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("Smoothing alpha must be a positive finite number.")
    vocabulary = sorted({term for vector in (*relevant, *irrelevant) for term in vector})

    def log_theta(vectors: Sequence[Mapping[str, float]]) -> dict[str, float]:
        parts: defaultdict[str, list[float]] = defaultdict(list)
        for vector in vectors:
            for term, value in vector.items():
                parts[term].append(value)
        mass = {term: math.fsum(parts[term]) / len(vectors) for term in vocabulary}
        denominator = math.fsum(mass.values()) + alpha * len(vocabulary)
        return {term: math.log((mass[term] + alpha) / denominator) for term in vocabulary}

    positive, negative = log_theta(relevant), log_theta(irrelevant)
    return {term: positive[term] - negative[term] for term in vocabulary}


def explain(
    vector: Mapping[str, float], weights: Mapping[str, float]
) -> tuple[float, tuple[tuple[str, float], ...]]:
    """Log-odds under uniform class priors, plus every term's x_t * w_t contribution."""
    contributions = tuple(
        sorted(
            ((term, value * weights[term]) for term, value in vector.items() if term in weights),
            key=lambda pair: (-pair[1], pair[0]),
        )
    )
    return math.fsum(value for _, value in contributions), contributions


def relevance_probability(log_odds: float) -> float:
    """Numerically stable logistic function."""
    if log_odds >= 0:
        return 1.0 / (1.0 + math.exp(-log_odds))
    exponential = math.exp(log_odds)
    return exponential / (1.0 + exponential)


class TermContribution(BaseModel):
    """One term's positive share x_t * w_t of a paper's log-odds, rounded for display."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    term: str = Field(min_length=MIN_TERM_CHARACTERS, max_length=MAX_TERM_CHARACTERS)
    contribution: float = Field(gt=0.0, allow_inf_nan=False)


class PriorityItem(BaseModel):
    """A paper still needing a human decision; never a suggested include or exclude."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rank: int = Field(strict=True, ge=1, le=MAX_DOCUMENT_IDS)
    document_id: Identity
    status: PendingStatus
    title: str = Field(max_length=MAX_TITLE_CHARACTERS)
    title_truncated: bool
    score: float | None = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    log_odds: float | None = Field(allow_inf_nan=False)
    top_terms: tuple[TermContribution, ...] = Field(max_length=TOP_TERMS)


class PriorityMethod(BaseModel):
    """Exact, versioned description of how the ordering was computed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: Literal["tfidf-multinomial-naive-bayes-v1"] = "tfidf-multinomial-naive-bayes-v1"
    features: str = (
        f"Title (first {MAX_TITLE_CHARACTERS} characters) plus the first "
        f"{MAX_FEATURE_TEXT_CHARACTERS} characters of stored text. NFKC-casefolded letter/digit "
        f"runs of {MIN_TERM_CHARACTERS}-{MAX_TERM_CHARACTERS} characters; all-digit tokens and "
        "a fixed English stop-word list are dropped."
    )
    weighting: str = (
        "tf = 1 + ln(count); idf = ln((1 + N) / (1 + df)) + 1 over all N current members; "
        "each paper's tf * idf vector is L2-normalized."
    )
    classifier: str = (
        "Multinomial Naive Bayes on TF-IDF weights of terms that occur in at least one labeled "
        "paper. Class term mass is the mean weight over that class's labeled papers "
        "(class-balanced); theta = (mass + alpha) / (total mass + alpha * |V|)."
    )
    class_prior: str = (
        "Uniform (1/2 each). The prior adds the same constant to every paper's log-odds, so it "
        "never changes the ranking."
    )
    training_labels: str = (
        "Current include = relevant; current exclude = irrelevant. Unsure, stale and "
        "unscreened members are never labels."
    )
    score: str = (
        "log_odds = sum over terms of x_t * (ln theta_rel,t - ln theta_irr,t); "
        "score = P(relevant) = 1 / (1 + exp(-log_odds))."
    )
    tie_break: str = (
        f"Equal log-odds (to {TIE_DECIMALS} decimal places) order by ascending document ID."
    )
    explanation: str = (
        f"Up to {TOP_TERMS} terms with the largest positive contribution x_t * w_t; "
        f"values rounded to {_DISPLAY_DECIMALS} decimal places."
    )
    alpha: float = NAIVE_BAYES_ALPHA
    max_text_characters: int = MAX_FEATURE_TEXT_CHARACTERS
    top_terms: int = TOP_TERMS


class ScreeningPriority(BaseModel):
    """An auditable ordering of papers still needing a decision; never a decision itself."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    collection_id: CollectionId
    collection_name: CollectionName
    collection_revision: Revision
    model_status: ModelStatus
    ordering: PriorityOrdering
    total_documents: int = Field(strict=True, ge=1, le=MAX_DOCUMENT_IDS)
    counts: ScreeningCounts
    training_documents: int = Field(strict=True, ge=0, le=MAX_DOCUMENT_IDS)
    training_relevant: int = Field(strict=True, ge=0, le=MAX_DOCUMENT_IDS)
    training_irrelevant: int = Field(strict=True, ge=0, le=MAX_DOCUMENT_IDS)
    feature_terms: int = Field(strict=True, ge=0)
    items: tuple[PriorityItem, ...] = Field(max_length=MAX_DOCUMENT_IDS)
    unsure_items: tuple[PriorityItem, ...] = Field(max_length=MAX_DOCUMENT_IDS)
    method: PriorityMethod = PriorityMethod()
    notes: tuple[str, ...] = _NOTES

    @model_validator(mode="after")
    def consistent(self) -> Self:
        counts = self.counts
        ready = self.model_status == "ready"
        scored = [item.score is not None and item.log_odds is not None for item in self.everything]
        if (
            (self.ordering == "most_likely_relevant_first") != ready
            or (ready and not all(scored))
            or (not ready and (any(scored) or any(item.top_terms for item in self.everything)))
            or (ready and (not self.training_relevant or not self.training_irrelevant))
            or (ready != (self.feature_terms > 0))
            or self.training_relevant != counts.include
            or self.training_irrelevant != counts.exclude
            or self.training_documents != counts.include + counts.exclude
            or self.total_documents != sum(counts.model_dump().values())
            or len(self.items) != counts.unscreened + counts.stale
            or len(self.unsure_items) != counts.unsure
            or any(item.status == "unsure" for item in self.items)
            or any(item.status != "unsure" for item in self.unsure_items)
            or [item.rank for item in self.items] != list(range(1, len(self.items) + 1))
            or [item.rank for item in self.unsure_items]
            != list(range(1, len(self.unsure_items) + 1))
            or len({item.document_id for item in self.everything}) != len(self.everything)
        ):
            raise ValueError("Priority fields must agree with one complete screening snapshot.")
        return self

    @property
    def everything(self) -> tuple[PriorityItem, ...]:
        return self.items + self.unsure_items


@dataclass(frozen=True)
class PriorityMember:
    """One member's exact identity, current status and bounded stored text."""

    document_id: str
    status: ScreeningStatus
    title: str
    title_truncated: bool
    text: str


@dataclass(frozen=True)
class _Ranking:
    model_status: ModelStatus
    feature_terms: int
    items: tuple[PriorityItem, ...]
    unsure_items: tuple[PriorityItem, ...]


def _model(
    vectors: Sequence[Mapping[str, float]], statuses: Sequence[ScreeningStatus]
) -> tuple[ModelStatus, dict[str, float]]:
    relevant = [
        vector for vector, status in zip(vectors, statuses, strict=True) if status == "include"
    ]
    irrelevant = [
        vector for vector, status in zip(vectors, statuses, strict=True) if status == "exclude"
    ]
    if not relevant or not irrelevant:
        return "needs_include_and_exclude", {}
    weights = train_naive_bayes(relevant, irrelevant)
    return ("ready", weights) if weights else ("no_training_terms", {})


def _ordered(
    indices: Sequence[int], log_odds: Sequence[float], identifiers: Sequence[str]
) -> list[int]:
    return sorted(
        indices, key=lambda index: (-round(log_odds[index], TIE_DECIMALS), identifiers[index])
    )


def _display(value: float) -> float:
    return round(value, _DISPLAY_DECIMALS) + 0.0


def rank_members(members: Sequence[PriorityMember]) -> _Ranking:
    """Pure ranking of one complete membership snapshot; no I/O, clock or randomness."""
    identifiers = [member.document_id for member in members]
    statuses = [member.status for member in members]
    vectors = tfidf_vectors([tokenize(feature_text(m.title, m.text)) for m in members])
    model_status, weights = _model(vectors, statuses)
    explained = [explain(vector, weights) for vector in vectors]
    log_odds = [value for value, _ in explained]

    ready = model_status == "ready"

    def items(wanted: tuple[ScreeningStatus, ...]) -> tuple[PriorityItem, ...]:
        selected = [index for index, status in enumerate(statuses) if status in wanted]
        if ready:
            selected = _ordered(selected, log_odds, identifiers)
        result = []
        for rank, index in enumerate(selected, 1):
            member = members[index]
            top_terms = tuple(
                TermContribution(term=term, contribution=_display(value))
                for term, value in explained[index][1]
                if ready and _display(value) > 0
            )[:TOP_TERMS]
            result.append(
                PriorityItem(
                    rank=rank,
                    document_id=member.document_id,
                    status=cast(PendingStatus, member.status),
                    title=member.title,
                    title_truncated=member.title_truncated,
                    score=_display(relevance_probability(log_odds[index])) if ready else None,
                    log_odds=_display(log_odds[index]) if ready else None,
                    top_terms=top_terms,
                )
            )
        return tuple(result)

    return _Ranking(
        model_status=model_status,
        feature_terms=len(weights),
        items=items(_PENDING),
        unsure_items=items(("unsure",)),
    )


def build_priority(
    collection: PaperCollection, queue: ScreeningQueue, members: Sequence[PriorityMember]
) -> ScreeningPriority:
    """Combine one snapshot's metadata and its pure ranking into the public response."""
    if [member.document_id for member in members] != [
        item.document_id for item in queue.items
    ] or any(
        member.status != item.status for member, item in zip(members, queue.items, strict=True)
    ):
        raise _invalid_priority()
    try:
        ranking = rank_members(members)
        return ScreeningPriority(
            collection_id=collection.collection_id,
            collection_name=collection.name,
            collection_revision=collection.revision,
            model_status=ranking.model_status,
            ordering="most_likely_relevant_first"
            if ranking.model_status == "ready"
            else "collection_order",
            total_documents=queue.total_documents,
            counts=queue.counts,
            training_documents=queue.counts.include + queue.counts.exclude,
            training_relevant=queue.counts.include,
            training_irrelevant=queue.counts.exclude,
            feature_terms=ranking.feature_terms,
            items=ranking.items,
            unsure_items=ranking.unsure_items,
        )
    except ValidationError as exc:
        raise _invalid_priority() from exc


def _invalid_priority() -> CollectionError:
    return CollectionError(
        "invalid_screening_priority",
        "The screening snapshot could not be ordered consistently; nothing was saved.",
        409,
    )


class _PriorityOptions(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    collection_id: CollectionId
    collection_revision: Revision


_MEMBERS_SQL = """
SELECT document_id,
       CASE WHEN title = '' THEN X''
           ELSE substr(CAST(title AS BLOB), 1, :title_bytes) END AS title,
       CASE WHEN text = '' THEN X''
           ELSE substr(CAST(text AS BLOB), 1, :text_bytes) END AS text,
       typeof(title) != 'text' OR typeof(text) != 'text'
           OR instr(title, char(0)) > 0 AS invalid
FROM documents
WHERE document_id IN (SELECT value FROM json_each(:document_ids))
LIMIT :row_limit
"""


def prioritize_screening(
    screening: SQLitePaperScreening,
    collection_id: str,
    *,
    collection_revision: int,
) -> ScreeningPriority:
    """Read one snapshot of an exact collection revision and order its pending members.

    Read-only: no decision, history, collection, corpus or event write occurs.
    """
    options = _PriorityOptions(collection_id=collection_id, collection_revision=collection_revision)
    with screening.collections.resolved_collection(
        options.collection_id, expected_revision=options.collection_revision
    ) as (connection, collection):
        queue = screening._snapshot(connection, collection)
        encoding = connection.execute("PRAGMA encoding").fetchone()[0]
        rows = connection.execute(
            _MEMBERS_SQL,
            {
                "title_bytes": 4 * (MAX_TITLE_CHARACTERS + 1),
                "text_bytes": 4 * (MAX_FEATURE_TEXT_CHARACTERS + 1),
                "document_ids": json.dumps(collection.document_ids),
                "row_limit": MAX_DOCUMENT_IDS + 1,
            },
        ).fetchall()
        stored = {row["document_id"]: row for row in rows}
        try:
            if (
                len(stored) != len(rows)
                or set(stored) != set(collection.document_ids)
                or any(row["invalid"] for row in rows)
            ):
                raise DocumentCatalogError()
            members = []
            for item in queue.items:
                row = stored[item.document_id]
                title = _prefix(row["title"], encoding, MAX_TITLE_CHARACTERS)
                text = _prefix(row["text"], encoding, MAX_FEATURE_TEXT_CHARACTERS)
                members.append(
                    PriorityMember(
                        document_id=item.document_id,
                        status=item.status,
                        title=title[:MAX_TITLE_CHARACTERS],
                        title_truncated=len(title) > MAX_TITLE_CHARACTERS,
                        text=text[:MAX_FEATURE_TEXT_CHARACTERS],
                    )
                )
        except DocumentCatalogError as exc:
            error = DocumentCatalogError()
            raise CollectionError(error.code, str(error), 409) from exc
    result = build_priority(collection, queue, members)
    if len(result.model_dump_json().encode("utf-8")) > MAX_PRIORITY_RESPONSE_BYTES:
        raise CollectionError(
            "screening_priority_too_large",
            "The complete ordering exceeds its UTF-8 byte limit; no partial ordering.",
            413,
        )
    return result


class SimulatedPaper(BaseModel):
    """A paper with a known label, for offline replay only; never stored or shown as a decision."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: Identity
    title: str
    text: str
    relevant: bool


class ScreeningSimulation(BaseModel):
    """One replay of a reviewer who screens every paper in the order a strategy proposes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy: PriorityOrdering
    total_documents: int = Field(strict=True, ge=2, le=MAX_DOCUMENT_IDS)
    relevant_documents: int = Field(strict=True, ge=1, le=MAX_DOCUMENT_IDS)
    recall_target: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)
    relevant_target: int = Field(strict=True, ge=1, le=MAX_DOCUMENT_IDS)
    screened_document_ids: tuple[str, ...] = Field(max_length=MAX_DOCUMENT_IDS)
    relevant_found: tuple[int, ...] = Field(max_length=MAX_DOCUMENT_IDS)
    papers_to_recall_target: int = Field(strict=True, ge=1, le=MAX_DOCUMENT_IDS)
    papers_to_find_all_relevant: int = Field(strict=True, ge=1, le=MAX_DOCUMENT_IDS)
    work_saved_over_sampling: float = Field(allow_inf_nan=False)
    longest_irrelevant_run_before_last_relevant: int = Field(strict=True, ge=0)


def simulate_screening(
    papers: Sequence[SimulatedPaper],
    *,
    strategy: PriorityOrdering,
    recall_target: float = 0.95,
) -> ScreeningSimulation:
    """Replay screening with known labels, retraining after every simulated decision.

    "collection_order" screens in the given order. "most_likely_relevant_first" uses exactly
    the API ranking: until one relevant and one irrelevant paper are labeled it falls back to
    the given order, then always screens the current top-ranked remaining paper.
    WSS = (N - n) / N - (1 - recall_target), where n papers were screened to reach the
    target recall (Cohen et al., 2006).
    """
    if strategy not in ("most_likely_relevant_first", "collection_order"):
        raise ValueError("Unknown simulation strategy.")
    count = len(papers)
    identifiers = [paper.document_id for paper in papers]
    relevant_total = sum(paper.relevant for paper in papers)
    if not 2 <= count <= MAX_DOCUMENT_IDS or len(set(identifiers)) != count:
        raise ValueError("Simulate 2-100 papers with unique document IDs.")
    if not relevant_total:
        raise ValueError("Recall needs at least one relevant paper.")
    if not math.isfinite(recall_target) or not 0 < recall_target <= 1:
        raise ValueError("The recall target must be in (0, 1].")
    vectors = tfidf_vectors([tokenize(feature_text(p.title, p.text)) for p in papers])
    statuses: list[ScreeningStatus] = ["unscreened"] * count
    screened: list[str] = []
    labels: list[bool] = []
    found: list[int] = []
    for _ in range(count):
        remaining = [index for index, status in enumerate(statuses) if status == "unscreened"]
        choice = remaining[0]
        if strategy == "most_likely_relevant_first":
            model_status, weights = _model(vectors, statuses)
            if model_status == "ready":
                log_odds = [
                    explain(vector, weights)[0] if status == "unscreened" else 0.0
                    for vector, status in zip(vectors, statuses, strict=True)
                ]
                choice = _ordered(remaining, log_odds, identifiers)[0]
        relevant = papers[choice].relevant
        statuses[choice] = "include" if relevant else "exclude"
        screened.append(identifiers[choice])
        labels.append(relevant)
        found.append((found[-1] if found else 0) + int(relevant))
    target = max(1, math.ceil(round(recall_target * relevant_total, 9)))
    reached = next(index for index, value in enumerate(found, 1) if value >= target)
    last = found.index(relevant_total) + 1
    run = longest = 0
    for relevant in labels[:last]:
        run = 0 if relevant else run + 1
        longest = max(longest, run)
    return ScreeningSimulation(
        strategy=strategy,
        total_documents=count,
        relevant_documents=relevant_total,
        recall_target=recall_target,
        relevant_target=target,
        screened_document_ids=tuple(screened),
        relevant_found=tuple(found),
        papers_to_recall_target=reached,
        papers_to_find_all_relevant=last,
        work_saved_over_sampling=_display((count - reached) / count - (1 - recall_target)),
        longest_irrelevant_run_before_last_relevant=longest,
    )
