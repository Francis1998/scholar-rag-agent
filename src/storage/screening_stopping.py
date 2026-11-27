"""Advisory, deterministic stopping rules for prioritized human title/abstract screening.

One read snapshot of a collection's recorded human decisions is evaluated against three
published ideas: ASReview's "N consecutive irrelevant records" heuristic, a SAFE-style
phase-2 checkpoint (Boetje & van de Schoot, 2024) and buscar's hypergeometric test of
"recall < target" on a seeded random sample of the unscreened remainder (Callaghan &
Mueller-Hansen, 2020). Nothing here records, excludes, stops or changes a decision.
Arithmetic is exact (integers and fractions) and offline; the only randomness is a
user-chosen seed hashed with SHA-256.
"""

import hashlib
import json
import math
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from fractions import Fraction
from typing import Annotated, Literal, Self

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

from retrieval.scope import MAX_DOCUMENT_IDS
from storage.document_catalog import Identity
from storage.paper_collections import (
    MAX_REVISION,
    CollectionError,
    CollectionId,
    CollectionName,
    PaperCollection,
    Revision,
)
from storage.paper_screening import (
    Decision,
    ScreeningCounts,
    ScreeningQueue,
    ScreeningStatus,
    SQLitePaperScreening,
    _history_records,
    _invalid_history,
)
from storage.source_context import ContextDocumentIdentity

MAX_STOPPING_RESPONSE_BYTES = 524_288
MAX_HISTORY_ROWS = 10_000
MAX_SEED = 2**63 - 1
MAX_DECIMAL_PLACES = 6
DEFAULT_TARGET_RECALL = 0.95
DEFAULT_CONFIDENCE = 0.95
DEFAULT_IRRELEVANT_RUN_PERCENT = 10
DEFAULT_MIN_SCREENED_PERCENT = 10
SAMPLE_ORDER_VERSION = "scholar-rag-stopping-sample-v1"
_HISTORY_PAGE = 100
_PENDING: tuple[ScreeningStatus, ...] = ("unscreened", "stale")
RuleStatus = Literal["met", "not_met", "not_evaluable"]
CheckpointStatus = Literal["met", "not_met", "incomplete"]
RecallTestStatus = Literal["not_started", "met", "not_met", "not_evaluable"]
SafeCriterionName = Literal[
    "key_papers_included",
    "screened_twice_estimated_relevant",
    "minimum_screened_percent",
    "consecutive_irrelevant",
]


def _check_hypergeometric(population: int, successes: int, draws: int) -> None:
    if any(type(value) is not int for value in (population, successes, draws)):
        raise ValueError("Hypergeometric arguments must be integers.")
    if not 0 <= successes <= population or not 0 <= draws <= population:
        raise ValueError("Need 0 <= successes <= population and 0 <= draws <= population.")


def _support(population: int, successes: int, draws: int) -> tuple[int, int]:
    return max(0, draws - (population - successes)), min(draws, successes)


def hypergeometric_pmf(k: int, population: int, successes: int, draws: int) -> Fraction:
    """Exact P(X = k) for ``draws`` taken without replacement from ``population`` items,
    ``successes`` of which are relevant: C(K, k) C(N - K, n - k) / C(N, n)."""
    _check_hypergeometric(population, successes, draws)
    if type(k) is not int:
        raise ValueError("k must be an integer.")
    low, high = _support(population, successes, draws)
    if not low <= k <= high:
        return Fraction(0)
    return Fraction(
        math.comb(successes, k) * math.comb(population - successes, draws - k),
        math.comb(population, draws),
    )


def hypergeometric_cdf(k: int, population: int, successes: int, draws: int) -> Fraction:
    """Exact lower tail P(X <= k), the quantity buscar compares with 1 - confidence."""
    _check_hypergeometric(population, successes, draws)
    if type(k) is not int:
        raise ValueError("k must be an integer.")
    low, high = _support(population, successes, draws)
    if k < low:
        return Fraction(0)
    if k >= high:
        return Fraction(1)
    return Fraction(
        sum(
            math.comb(successes, x) * math.comb(population - successes, draws - x)
            for x in range(low, k + 1)
        ),
        math.comb(population, draws),
    )


def min_relevant_for_missed_target(
    relevant_seen: int, relevant_before_sample: int, target_recall: Fraction
) -> int:
    """buscar's K_hat: the fewest relevant urn records that would leave recall < target.

    Final recall is relevant_seen / (relevant_before_sample + K) for K relevant records in
    the urn, so recall < target exactly when K > relevant_seen / target - relevant_before.
    """
    if not 0 < target_recall <= 1:
        raise ValueError("The target recall must be in (0, 1].")
    if not 0 <= relevant_before_sample <= relevant_seen:
        raise ValueError("Relevant records before the sample cannot exceed those seen.")
    return math.floor(Fraction(relevant_seen) / target_recall - relevant_before_sample) + 1


@dataclass(frozen=True)
class UrnObservation:
    """What is known about the urn: members not yet screened when the sample was drawn.

    The observed sample is a simple random sample of the urn. Members of the urn decided
    outside the sample (for example from the prioritized queue) count as found records but
    never as sample observations.
    """

    urn_size: int
    sample_size: int
    sample_relevant: int
    relevant_before: int
    decided_outside: int = 0
    relevant_outside: int = 0

    def __post_init__(self) -> None:
        values = (
            self.urn_size,
            self.sample_size,
            self.sample_relevant,
            self.relevant_before,
            self.decided_outside,
            self.relevant_outside,
        )
        if any(type(value) is not int or value < 0 for value in values) or not (
            self.sample_relevant <= self.sample_size
            and self.relevant_outside <= self.decided_outside
            and self.sample_size + self.decided_outside <= self.urn_size
        ):
            raise ValueError("Urn counts must be consistent non-negative integers.")

    @property
    def relevant_seen(self) -> int:
        return self.relevant_before + self.sample_relevant + self.relevant_outside

    @property
    def capacity(self) -> int:
        """The most relevant records the urn can hold given every known label."""
        irrelevant = (self.sample_size - self.sample_relevant) + (
            self.decided_outside - self.relevant_outside
        )
        return self.urn_size - irrelevant

    @property
    def unscreened(self) -> int:
        return self.urn_size - self.sample_size - self.decided_outside


def recall_p_value(observation: UrnObservation, target_recall: Fraction) -> tuple[int, Fraction]:
    """Return (K_hat, p) for H0 "recall < target"; p = P(X <= k | N_urn, K_hat, n).

    When fewer than K_hat urn members can still be relevant, H0 is impossible and p = 0.
    """
    k_hat = min_relevant_for_missed_target(
        observation.relevant_seen, observation.relevant_before, target_recall
    )
    if k_hat > observation.capacity:
        return k_hat, Fraction(0)
    return k_hat, hypergeometric_cdf(
        observation.sample_relevant, observation.urn_size, k_hat, observation.sample_size
    )


def recall_lower_bound(observation: UrnObservation, alpha: Fraction) -> Fraction | None:
    """The smallest recall not rejected at level alpha: the inverted buscar test.

    It uses the largest urn total K whose lower tail P(X <= k) is at least alpha, so a
    target is rejected (p < alpha) exactly when this bound reaches it. None until a
    relevant record has been found, because recall is then undefined.
    """
    if not 0 < alpha < 1:
        raise ValueError("Alpha must be in (0, 1).")
    if observation.relevant_seen == 0:
        return None
    known = observation.sample_relevant + observation.relevant_outside
    largest = known
    for total in range(known, observation.capacity + 1):
        tail = hypergeometric_cdf(
            observation.sample_relevant, observation.urn_size, total, observation.sample_size
        )
        if tail < alpha:
            break
        largest = total
    return Fraction(observation.relevant_seen, observation.relevant_before + largest)


def recall_estimate(observation: UrnObservation) -> Fraction | None:
    """Point estimate: the sample's relevant fraction extrapolated to still-unscreened members."""
    if observation.sample_size == 0 or observation.relevant_seen == 0:
        return None
    remaining = Fraction(observation.sample_relevant, observation.sample_size) * (
        observation.unscreened
    )
    return observation.relevant_seen / (observation.relevant_seen + remaining)


def screens_until_met(
    observation: UrnObservation,
    upcoming: Sequence[ScreeningStatus],
    target_recall: Fraction,
    alpha: Fraction,
) -> int | None:
    """How many pending sample records, all screened irrelevant, would bring p below alpha.

    ``upcoming`` is the current status of each urn member after the observed prefix, in
    sample order. Members already decided outside the sample join it with their known
    label without being counted. An unsure member pauses the projection (None).
    """
    state = observation
    screened = 0
    for status in (None, *upcoming):
        if status == "unsure":
            return None
        if status in _PENDING:
            state = replace(state, sample_size=state.sample_size + 1)
            screened += 1
        elif status is not None:
            relevant = int(status == "include")
            state = replace(
                state,
                sample_size=state.sample_size + 1,
                sample_relevant=state.sample_relevant + relevant,
                decided_outside=state.decided_outside - 1,
                relevant_outside=state.relevant_outside - relevant,
            )
        if recall_p_value(state, target_recall)[1] < alpha:
            return screened
    return None


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode("ascii")


def sample_order(
    document_ids: Iterable[str], *, seed: int, collection_id: str, collection_revision: int
) -> tuple[str, ...]:
    """A reproducible pseudo-random order: ascending SHA-256 of one canonical key per member.

    The key is the JSON array [version, seed, collection_id, collection_revision, document_id],
    so the order never depends on Python's random module, platform or insertion order.
    """

    def key(document_id: str) -> tuple[str, str]:
        material = [SAMPLE_ORDER_VERSION, seed, collection_id, collection_revision, document_id]
        return hashlib.sha256(_canonical(material)).hexdigest(), document_id

    return tuple(sorted(document_ids, key=key))


def urn_digest(document_ids: Iterable[str], *, collection_id: str, collection_revision: int) -> str:
    """SHA-256 identifying exactly which members formed the urn, independent of their order."""
    material = [SAMPLE_ORDER_VERSION, collection_id, collection_revision, sorted(document_ids)]
    return hashlib.sha256(_canonical(material)).hexdigest()


def _decimal_places(value: float) -> float:
    if round(value, MAX_DECIMAL_PLACES) != value:
        raise ValueError(f"Use at most {MAX_DECIMAL_PLACES} decimal places.")
    return value


def _exact(value: float) -> Fraction:
    """The decimal the caller wrote (0.95 is 19/20), not the nearest binary float."""
    return Fraction(repr(value))


class StoppingOptions(BaseModel):
    """Validated assessment parameters; every default is documented and echoed back."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    collection_id: CollectionId
    collection_revision: Revision
    target_recall: Annotated[
        float, Field(gt=0, le=1, allow_inf_nan=False), AfterValidator(_decimal_places)
    ] = DEFAULT_TARGET_RECALL
    confidence: Annotated[
        float, Field(gt=0, lt=1, allow_inf_nan=False), AfterValidator(_decimal_places)
    ] = DEFAULT_CONFIDENCE
    irrelevant_run: int | None = Field(default=None, strict=True, ge=1, le=MAX_DOCUMENT_IDS)
    irrelevant_run_percent: int | None = Field(default=None, strict=True, ge=1, le=100)
    min_screened_percent: int = Field(
        default=DEFAULT_MIN_SCREENED_PERCENT, strict=True, ge=1, le=100
    )
    estimated_relevant: int | None = Field(default=None, strict=True, ge=0, le=MAX_DOCUMENT_IDS)
    key_document_ids: tuple[ContextDocumentIdentity, ...] = Field(
        default=(), max_length=MAX_DOCUMENT_IDS
    )
    sample_seed: int | None = Field(default=None, strict=True, ge=0, le=MAX_SEED)
    sample_after: int | None = Field(default=None, strict=True, ge=0, le=MAX_DOCUMENT_IDS)
    sample_urn_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if self.irrelevant_run is not None and self.irrelevant_run_percent is not None:
            raise ValueError("Give irrelevant_run or irrelevant_run_percent, not both.")
        if (self.sample_seed is None) != (self.sample_after is None):
            raise ValueError("sample_seed and sample_after are given together.")
        if self.sample_urn_sha256 is not None and self.sample_seed is None:
            raise ValueError("sample_urn_sha256 verifies a started sample only.")
        if len(set(self.key_document_ids)) != len(self.key_document_ids):
            raise ValueError("Key document IDs must be unique.")
        return self


class StoppingParameters(BaseModel):
    """The effective parameters, including defaults and the resolved run threshold."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    target_recall: float = Field(gt=0, le=1, allow_inf_nan=False)
    confidence: float = Field(gt=0, lt=1, allow_inf_nan=False)
    irrelevant_run: int = Field(strict=True, ge=1, le=MAX_DOCUMENT_IDS)
    irrelevant_run_basis: Literal["records", "percent_of_collection"]
    irrelevant_run_percent: int | None = Field(strict=True, ge=1, le=100)
    min_screened_percent: int = Field(strict=True, ge=1, le=100)
    estimated_relevant: int | None = Field(strict=True, ge=0, le=MAX_DOCUMENT_IDS)
    key_document_ids: tuple[Identity, ...] = Field(max_length=MAX_DOCUMENT_IDS)
    sample_seed: int | None = Field(strict=True, ge=0, le=MAX_SEED)
    sample_after: int | None = Field(strict=True, ge=0, le=MAX_DOCUMENT_IDS)
    sample_urn_sha256: str | None = Field(pattern=r"^[0-9a-f]{64}$")


class TimelineItem(BaseModel):
    """One member with a current decision, at its recorded screening position."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    position: int = Field(strict=True, ge=1, le=MAX_DOCUMENT_IDS)
    document_id: Identity
    decision: Decision
    first_decided_at: AwareDatetime
    decided_at: AwareDatetime
    first_decision_retained: bool
    tied_with_previous: bool
    includes_so_far: int = Field(strict=True, ge=0, le=MAX_DOCUMENT_IDS)


class DecisionTimeline(BaseModel):
    """Current decisions in recorded order: an explicit, disclosed reconstruction."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    order: Literal["first_recorded_decision_time_at_this_collection_revision"] = (
        "first_recorded_decision_time_at_this_collection_revision"
    )
    screened: int = Field(strict=True, ge=0, le=MAX_DOCUMENT_IDS)
    pending: int = Field(strict=True, ge=0, le=MAX_DOCUMENT_IDS)
    order_reliable: bool
    order_warnings: tuple[str, ...] = Field(max_length=2 * MAX_DOCUMENT_IDS + 1)
    tied_positions: int = Field(strict=True, ge=0, le=MAX_DOCUMENT_IDS)
    items: tuple[TimelineItem, ...] = Field(max_length=MAX_DOCUMENT_IDS)


class ConsecutiveIrrelevantRule(BaseModel):
    """ASReview-style count of excludes since the most recent include or unsure."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: RuleStatus
    threshold: int = Field(strict=True, ge=1, le=MAX_DOCUMENT_IDS)
    threshold_percent_of_collection: float = Field(ge=0, le=100, allow_inf_nan=False)
    current_run: int = Field(strict=True, ge=0, le=MAX_DOCUMENT_IDS)
    current_run_percent_of_collection: float = Field(ge=0, le=100, allow_inf_nan=False)
    last_relevant_at: AwareDatetime | None
    run_document_ids: tuple[Identity, ...] = Field(max_length=MAX_DOCUMENT_IDS)
    explanation: str


class SafeCriterion(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: SafeCriterionName
    status: RuleStatus
    detail: str


class SafeCheckpoint(BaseModel):
    """SAFE phase 2's four stopping conditions; met only when every one is met."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: CheckpointStatus
    criteria: tuple[SafeCriterion, SafeCriterion, SafeCriterion, SafeCriterion]
    key_documents_not_included: tuple[Identity, ...] = Field(max_length=MAX_DOCUMENT_IDS)
    explanation: str


class SampleItem(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_position: int = Field(strict=True, ge=1, le=MAX_DOCUMENT_IDS)
    document_id: Identity
    status: ScreeningStatus
    observed: bool


class RandomSampleRecallTest(BaseModel):
    """buscar: a seeded random sample of the unscreened remainder and H0 "recall < target"."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    method: Literal["buscar-hypergeometric-v1"] = "buscar-hypergeometric-v1"
    status: RecallTestStatus
    target_recall: float = Field(gt=0, le=1, allow_inf_nan=False)
    confidence: float = Field(gt=0, lt=1, allow_inf_nan=False)
    sample_seed: int | None = Field(strict=True, ge=0, le=MAX_SEED)
    sample_after: int = Field(strict=True, ge=0, le=MAX_DOCUMENT_IDS)
    urn_size: int = Field(strict=True, ge=0, le=MAX_DOCUMENT_IDS)
    urn_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_order: tuple[SampleItem, ...] = Field(max_length=MAX_DOCUMENT_IDS)
    next_document_ids: tuple[Identity, ...] = Field(max_length=MAX_DOCUMENT_IDS)
    blocked_by_unsure: Identity | None
    unsure_before_sample: tuple[Identity, ...] = Field(max_length=MAX_DOCUMENT_IDS)
    sample_size: int = Field(strict=True, ge=0, le=MAX_DOCUMENT_IDS)
    sample_relevant: int = Field(strict=True, ge=0, le=MAX_DOCUMENT_IDS)
    relevant_before_sample: int = Field(strict=True, ge=0, le=MAX_DOCUMENT_IDS)
    decided_outside_sample: int = Field(strict=True, ge=0, le=MAX_DOCUMENT_IDS)
    relevant_outside_sample: int = Field(strict=True, ge=0, le=MAX_DOCUMENT_IDS)
    relevant_seen: int = Field(strict=True, ge=0, le=MAX_DOCUMENT_IDS)
    h0_min_relevant_in_urn: int | None = Field(strict=True, ge=1)
    p_value: float | None = Field(ge=0, le=1, allow_inf_nan=False)
    p_value_exact: str | None = Field(pattern=r"^[0-9]+/[0-9]+$")
    alpha: float = Field(gt=0, lt=1, allow_inf_nan=False)
    recall_estimate: float | None = Field(ge=0, le=1, allow_inf_nan=False)
    recall_lower_bound: float | None = Field(ge=0, le=1, allow_inf_nan=False)
    additional_irrelevant_needed: int | None = Field(strict=True, ge=0, le=MAX_DOCUMENT_IDS)
    explanation: str


ASSUMPTIONS = (
    "Recorded order: decisions are ordered by the server wall-clock time saved with each "
    "decision at this exact collection revision. Those times are not a guaranteed global "
    "sequence (clocks can repeat or move backwards). Detected inconsistencies make "
    "order-based rules not_evaluable; undetected clock changes remain possible.",
    "Each member's position is its first recorded decision at this collection revision. For "
    "the consecutive count an include or unsure counts at its latest decision time and an "
    "exclude at its first, the placement that gives the shortest run; equal times and "
    "unretained first decisions never lengthen it.",
    "Unsure counts as possibly relevant: it ends an irrelevant run and pauses the random "
    "sample until you resolve it to include or exclude, but it is never counted as a "
    "relevant record found. An unsure decision recorded before the sample is neither found "
    "nor in the urn, so the recall test is not_evaluable until it is resolved.",
    "Random sampling: the recall test is valid only if you chose the seed before seeing the "
    "sample, use one seed per assessment and screen sample records in the listed order "
    "without skipping. The order is SHA-256 of [version, seed, collection, revision, "
    "document ID]: reproducible, not secret.",
    "Independence: each human decision is treated as correct and independent of the order in "
    "which records were shown; reviewer error and later changes of mind are not modeled.",
    "The p-value is exact (hypergeometric, rational arithmetic) for the current sample size. "
    "It is recomputed after every sampled decision without adjustment for repeated looks, "
    "as in buscar's interactive use.",
)
LIMITATIONS = (
    "Advisory only: nothing is recorded, excluded, stopped or changed. Unscreened and stale "
    "members still need your explicit decision; when to stop is your documented judgment.",
    "Recall is relative to this collection's members only. It cannot see relevant studies "
    "that the search never brought into the collection.",
    "Collections hold at most 100 members, so the statistics are coarse: a small urn often "
    "needs most of its records screened before the test can pass.",
    "The consecutive-irrelevant and SAFE-style checks are heuristics without error "
    "guarantees. SAFE phases 3 (continue with a different model) and 4 (quality checks) are "
    "not implemented.",
    "Stale decisions from an earlier collection revision are pending, not screened; editing "
    "membership starts a new revision whose recorded order begins again.",
)


class ScreeningStoppingAssessment(BaseModel):
    """Advisory evidence about stopping; never a decision, exclusion or stop."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    collection_id: CollectionId
    collection_name: CollectionName
    collection_revision: Revision
    advisory_only: Literal[True] = True
    total_documents: int = Field(strict=True, ge=1, le=MAX_DOCUMENT_IDS)
    counts: ScreeningCounts
    all_screened: bool
    parameters: StoppingParameters
    timeline: DecisionTimeline
    consecutive_irrelevant: ConsecutiveIrrelevantRule
    safe_checkpoint: SafeCheckpoint
    recall_test: RandomSampleRecallTest
    assumptions: tuple[str, ...] = ASSUMPTIONS
    limitations: tuple[str, ...] = LIMITATIONS

    @model_validator(mode="after")
    def consistent(self) -> Self:
        counts = self.counts
        screened = counts.include + counts.exclude + counts.unsure
        pending = counts.unscreened + counts.stale
        decisions = [item.decision for item in self.timeline.items]
        if (
            self.total_documents != screened + pending
            or (self.timeline.screened, self.timeline.pending) != (screened, pending)
            or [item.position for item in self.timeline.items] != list(range(1, screened + 1))
            or (decisions.count("include"), decisions.count("exclude"), decisions.count("unsure"))
            != (counts.include, counts.exclude, counts.unsure)
            or self.all_screened != (pending == 0)
            or len({item.document_id for item in self.timeline.items}) != screened
        ):
            raise ValueError("Stopping fields must agree with one complete screening snapshot.")
        return self


@dataclass(frozen=True)
class RecordedDecision:
    """A member's current decision at this collection revision and when it was recorded."""

    document_id: str
    decision: Decision
    first_decided_at: datetime
    decided_at: datetime
    first_decision_retained: bool


def _percent(part: int, whole: int) -> float:
    return round(100 * part / whole, 2) if whole else 0.0


def _number(value: Fraction) -> str:
    return f"{float(value):.4g}"


def _out_of_range(message: str) -> CollectionError:
    return CollectionError("stopping_parameter_out_of_range", message, 422)


def _invalid_assessment() -> CollectionError:
    return CollectionError(
        "invalid_screening_stopping",
        "The screening snapshot could not be assessed consistently; nothing was saved.",
        409,
    )


def _timeline(
    ordered: Sequence[RecordedDecision], pending: int, warnings: Sequence[str]
) -> DecisionTimeline:
    items = []
    includes = 0
    for position, record in enumerate(ordered, 1):
        includes += record.decision == "include"
        items.append(
            TimelineItem(
                position=position,
                document_id=record.document_id,
                decision=record.decision,
                first_decided_at=record.first_decided_at,
                decided_at=record.decided_at,
                first_decision_retained=record.first_decision_retained,
                tied_with_previous=position > 1
                and record.first_decided_at == ordered[position - 2].first_decided_at,
                includes_so_far=includes,
            )
        )
    return DecisionTimeline(
        screened=len(items),
        pending=pending,
        order_reliable=not warnings,
        order_warnings=tuple(warnings),
        tied_positions=sum(item.tied_with_previous for item in items),
        items=tuple(items),
    )


def _consecutive(
    ordered: Sequence[RecordedDecision], total: int, threshold: int, reliable: bool
) -> ConsecutiveIrrelevantRule:
    last_relevant = max(
        (record.decided_at for record in ordered if record.decision != "exclude"), default=None
    )
    run = tuple(
        record.document_id
        for record in ordered
        if record.decision == "exclude"
        and record.first_decision_retained
        and (last_relevant is None or record.first_decided_at > last_relevant)
    )
    since = (
        "since the most recent include or unsure decision"
        if last_relevant is not None
        else "with no include or unsure decision yet"
    )
    status: RuleStatus = "met" if len(run) >= threshold else "not_met"
    if not reliable:
        status = "not_evaluable"
        explanation = (
            "Recorded decision times are inconsistent (see timeline.order_warnings), so "
            f"consecutive records cannot be counted honestly. The computed run of {len(run)} "
            "is shown for audit only."
        )
    elif status == "met":
        explanation = (
            f"{len(run)} excludes were first recorded {since}; the threshold is {threshold} "
            f"({_percent(threshold, total)}% of {total} members). Heuristic met: a common "
            "signal to consider stopping, not a recall guarantee."
        )
    else:
        explanation = (
            f"{len(run)} excludes were first recorded {since}; {threshold - len(run)} more "
            f"consecutive excludes reach the threshold of {threshold} "
            f"({_percent(threshold, total)}% of {total} members)."
        )
    return ConsecutiveIrrelevantRule(
        status=status,
        threshold=threshold,
        threshold_percent_of_collection=_percent(threshold, total),
        current_run=len(run),
        current_run_percent_of_collection=_percent(len(run), total),
        last_relevant_at=last_relevant,
        run_document_ids=run,
        explanation=explanation,
    )


def _safe(
    ordered: Sequence[RecordedDecision],
    total: int,
    options: StoppingOptions,
    consecutive: ConsecutiveIrrelevantRule,
) -> SafeCheckpoint:
    screened = len(ordered)
    included = {record.document_id for record in ordered if record.decision == "include"}
    keys = options.key_document_ids
    missing = tuple(key for key in keys if key not in included)
    key_status: RuleStatus = "not_evaluable" if not keys else "not_met" if missing else "met"
    key_detail = (
        "No key papers supplied. SAFE designates known relevant key papers before screening; "
        "pass key_document_id to check that each one is a current include."
        if not keys
        else f"{len(keys) - len(missing)} of {len(keys)} key papers are current includes."
    )
    estimate = options.estimated_relevant
    if estimate is None:
        twice_status: RuleStatus = "not_evaluable"
        twice_detail = (
            "Supply estimated_relevant (SAFE's RR_T: relevant records in the whole collection "
            "extrapolated from a random pilot sample). SAFE screens at least twice that; "
            f"{screened} screened so far."
        )
    else:
        twice_status = "met" if screened >= 2 * estimate else "not_met"
        twice_detail = f"{screened} screened; SAFE minimum 2 x {estimate} = {2 * estimate}."
    minimum = options.min_screened_percent
    minimum_status: RuleStatus = "met" if screened * 100 >= minimum * total else "not_met"
    criteria = (
        SafeCriterion(name="key_papers_included", status=key_status, detail=key_detail),
        SafeCriterion(
            name="screened_twice_estimated_relevant", status=twice_status, detail=twice_detail
        ),
        SafeCriterion(
            name="minimum_screened_percent",
            status=minimum_status,
            detail=f"{screened} of {total} members screened ({_percent(screened, total)}%); "
            f"minimum {minimum}%.",
        ),
        SafeCriterion(
            name="consecutive_irrelevant",
            status=consecutive.status,
            detail=f"{consecutive.current_run} consecutive excludes; threshold "
            f"{consecutive.threshold}.",
        ),
    )
    statuses = [criterion.status for criterion in criteria]
    status: CheckpointStatus = "met"
    if "not_met" in statuses:
        status = "not_met"
    elif "not_evaluable" in statuses:
        status = "incomplete"
    explanation = {
        "met": "All four SAFE phase-2 conditions are met. SAFE then continues with a different "
        "model (phase 3) and quality checks (phase 4) before stopping.",
        "not_met": "At least one SAFE phase-2 condition is not met; keep screening.",
        "incomplete": "No condition fails, but at least one cannot be evaluated; supply the "
        "missing inputs before relying on this checkpoint.",
    }[status]
    return SafeCheckpoint(
        status=status,
        criteria=criteria,
        key_documents_not_included=missing,
        explanation=explanation,
    )


def _recall_test(
    collection: PaperCollection,
    statuses: dict[str, ScreeningStatus],
    ordered: Sequence[RecordedDecision],
    options: StoppingOptions,
    reliable: bool,
) -> RandomSampleRecallTest:
    target, alpha = _exact(options.target_recall), 1 - _exact(options.confidence)
    collection_id, revision = collection.collection_id, collection.revision
    if options.sample_seed is None or options.sample_after is None:
        urn = [identifier for identifier, status in statuses.items() if status in _PENDING]
        unsure = tuple(record.document_id for record in ordered if record.decision == "unsure")
        resolve_first = (
            f" Resolve the {len(unsure)} unsure decision(s) in unsure_before_sample first: "
            "until then the test stays not_evaluable."
            if unsure
            else ""
        )
        return RandomSampleRecallTest(
            status="not_started",
            sample_after=len(ordered),
            urn_size=len(urn),
            urn_sha256=urn_digest(urn, collection_id=collection_id, collection_revision=revision),
            sample_order=(),
            next_document_ids=(),
            blocked_by_unsure=None,
            unsure_before_sample=unsure,
            sample_size=0,
            sample_relevant=0,
            relevant_before_sample=0,
            decided_outside_sample=0,
            relevant_outside_sample=0,
            relevant_seen=0,
            h0_min_relevant_in_urn=None,
            p_value=None,
            p_value_exact=None,
            recall_estimate=None,
            recall_lower_bound=None,
            additional_irrelevant_needed=None,
            explanation=f"Not started. When you are ready to stop prioritized screening, choose "
            f"a seed and draw a random sample of the {len(urn)} pending members (sample_after="
            f"{len(ordered)} and sample_urn_sha256 from this response). Screen the sample in the "
            f"listed order; the test then reports p for H0 'recall < {options.target_recall}'."
            + resolve_first,
            target_recall=options.target_recall,
            confidence=options.confidence,
            alpha=float(alpha),
            sample_seed=options.sample_seed,
        )
    after = options.sample_after
    if after > len(ordered):
        raise _out_of_range(
            f"sample_after cannot exceed the {len(ordered)} members with a current decision."
        )
    if 0 < after < len(ordered) and (
        ordered[after - 1].first_decided_at == ordered[after].first_decided_at
    ):
        raise CollectionError(
            "stopping_sample_boundary_tied",
            "Two members on either side of sample_after share one recorded time; the urn "
            "cannot be reconstructed unambiguously. Draw a new sample.",
            409,
        )
    before = ordered[:after]
    before_ids = {record.document_id for record in before}
    urn = [identifier for identifier in statuses if identifier not in before_ids]
    digest = urn_digest(urn, collection_id=collection_id, collection_revision=revision)
    if options.sample_urn_sha256 is not None and options.sample_urn_sha256 != digest:
        raise CollectionError(
            "stopping_sample_urn_changed",
            "The members not screened before sample_after differ from the urn this sample "
            "was drawn from. Nothing was assessed; draw a new sample.",
            409,
        )
    order = sample_order(
        urn, seed=options.sample_seed, collection_id=collection_id, collection_revision=revision
    )
    observed: list[str] = []
    blocked = None
    for identifier in order:
        if statuses[identifier] in ("include", "exclude"):
            observed.append(identifier)
            continue
        if statuses[identifier] == "unsure":
            blocked = identifier
        break
    in_sample = set(observed)
    outside = [
        identifier
        for identifier in urn
        if identifier not in in_sample and statuses[identifier] in ("include", "exclude")
    ]
    observation = UrnObservation(
        urn_size=len(urn),
        sample_size=len(observed),
        sample_relevant=sum(statuses[identifier] == "include" for identifier in observed),
        relevant_before=sum(record.decision == "include" for record in before),
        decided_outside=len(outside),
        relevant_outside=sum(statuses[identifier] == "include" for identifier in outside),
    )
    upcoming = order[len(observed) :]
    unsure_before = tuple(record.document_id for record in before if record.decision == "unsure")
    verified = reliable or options.sample_urn_sha256 is not None
    k_hat = p_value = estimate = bound = needed = None
    status: RecallTestStatus = "not_evaluable"
    if verified and not unsure_before:
        k_hat, p_value = recall_p_value(observation, target)
        status = "met" if p_value < alpha else "not_met"
        estimate = recall_estimate(observation)
        bound = recall_lower_bound(observation, alpha)
        needed = screens_until_met(
            observation, [statuses[identifier] for identifier in upcoming], target, alpha
        )
    return RandomSampleRecallTest(
        status=status,
        sample_after=after,
        urn_size=len(urn),
        urn_sha256=digest,
        sample_order=tuple(
            SampleItem(
                sample_position=position,
                document_id=identifier,
                status=statuses[identifier],
                observed=identifier in in_sample,
            )
            for position, identifier in enumerate(order, 1)
        ),
        next_document_ids=tuple(
            identifier for identifier in upcoming if statuses[identifier] in (*_PENDING, "unsure")
        ),
        blocked_by_unsure=blocked,
        unsure_before_sample=unsure_before,
        sample_size=observation.sample_size,
        sample_relevant=observation.sample_relevant,
        relevant_before_sample=observation.relevant_before,
        decided_outside_sample=observation.decided_outside,
        relevant_outside_sample=observation.relevant_outside,
        relevant_seen=observation.relevant_seen,
        h0_min_relevant_in_urn=k_hat,
        p_value=None if p_value is None else float(p_value),
        p_value_exact=None if p_value is None else f"{p_value.numerator}/{p_value.denominator}",
        recall_estimate=None if estimate is None else float(estimate),
        recall_lower_bound=None if bound is None else float(bound),
        additional_irrelevant_needed=needed,
        explanation=_recall_explanation(
            status,
            observation,
            options,
            p_value,
            alpha,
            bound,
            needed,
            blocked,
            unsure_before=unsure_before,
            verified=verified,
        ),
        target_recall=options.target_recall,
        confidence=options.confidence,
        alpha=float(alpha),
        sample_seed=options.sample_seed,
    )


def _recall_explanation(
    status: RecallTestStatus,
    observation: UrnObservation,
    options: StoppingOptions,
    p_value: Fraction | None,
    alpha: Fraction,
    bound: Fraction | None,
    needed: int | None,
    blocked: str | None,
    *,
    unsure_before: Sequence[str] = (),
    verified: bool = True,
) -> str:
    sample = (
        f"Sample: {observation.sample_size} of {observation.urn_size} urn members observed in "
        f"order, {observation.sample_relevant} relevant. Relevant records found overall: "
        f"{observation.relevant_seen}."
    )
    if status == "not_evaluable" or p_value is None:
        reasons = []
        if unsure_before:
            reasons.append(
                f"{len(unsure_before)} member(s) decided before the sample are unsure (see "
                "unsure_before_sample): they are neither found nor in the urn, so recall "
                "cannot be bounded. Resolve them to include or exclude."
            )
        if not verified:
            reasons.append(
                "Recorded decision times are inconsistent, so the urn behind sample_after "
                "cannot be trusted. Pass the sample_urn_sha256 recorded when you drew the "
                "sample to verify it."
            )
        return " ".join((sample, *reasons))
    target = options.target_recall
    if status == "met":
        text = (
            f"{sample} p = {_number(p_value)} < alpha {_number(alpha)}: reject H0 'recall < "
            f"{target}' at {options.confidence} confidence."
        )
    else:
        text = (
            f"{sample} p = {_number(p_value)} >= alpha {_number(alpha)}: H0 'recall < {target}' "
            "cannot be rejected yet."
        )
        if needed is not None:
            text += (
                f" If the next {needed} pending sample records are all irrelevant, p would "
                "fall below alpha."
            )
    if bound is not None:
        text += (
            f" Recall is at least {_number(bound)} with {options.confidence} confidence "
            "(relative to this collection)."
        )
    if blocked is not None:
        text += (
            f" Sample member {json.dumps(blocked)} is unsure: resolve it to include or exclude "
            "before later sample records count."
        )
    return text


def build_assessment(
    collection: PaperCollection,
    queue: ScreeningQueue,
    recorded: Sequence[RecordedDecision],
    order_warnings: Sequence[str],
    options: StoppingOptions,
) -> ScreeningStoppingAssessment:
    """Pure evaluation of one snapshot; no I/O, clock or randomness beyond the given seed."""
    statuses = {item.document_id: item.status for item in queue.items}
    decided = {
        identifier: status
        for identifier, status in statuses.items()
        if status in ("include", "exclude", "unsure")
    }
    if (
        options.collection_id != collection.collection_id
        or options.collection_revision != collection.revision
        or {record.document_id: record.decision for record in recorded} != decided
        or len(recorded) != len(decided)
    ):
        raise _invalid_assessment()
    total = queue.total_documents
    for key in options.key_document_ids:
        if key not in statuses:
            raise CollectionError(
                "stopping_key_document_not_member", "A key paper is not a collection member.", 422
            )
    if options.estimated_relevant is not None and options.estimated_relevant > total:
        raise _out_of_range(f"estimated_relevant cannot exceed the {total} members.")
    if options.irrelevant_run is not None and options.irrelevant_run > total:
        raise _out_of_range(f"irrelevant_run cannot exceed the {total} members.")
    percent = options.irrelevant_run_percent
    if options.irrelevant_run is None and percent is None:
        percent = DEFAULT_IRRELEVANT_RUN_PERCENT
    threshold = options.irrelevant_run or max(1, -(-(percent or 0) * total // 100))
    ordered = sorted(recorded, key=lambda record: (record.first_decided_at, record.document_id))
    pending = total - len(ordered)
    timeline = _timeline(ordered, pending, order_warnings)
    consecutive = _consecutive(ordered, total, threshold, timeline.order_reliable)
    try:
        return ScreeningStoppingAssessment(
            collection_id=collection.collection_id,
            collection_name=collection.name,
            collection_revision=collection.revision,
            total_documents=total,
            counts=queue.counts,
            all_screened=pending == 0,
            parameters=StoppingParameters(
                target_recall=options.target_recall,
                confidence=options.confidence,
                irrelevant_run=threshold,
                irrelevant_run_basis="records"
                if options.irrelevant_run is not None
                else "percent_of_collection",
                irrelevant_run_percent=percent,
                min_screened_percent=options.min_screened_percent,
                estimated_relevant=options.estimated_relevant,
                key_document_ids=options.key_document_ids,
                sample_seed=options.sample_seed,
                sample_after=options.sample_after,
                sample_urn_sha256=options.sample_urn_sha256,
            ),
            timeline=timeline,
            consecutive_irrelevant=consecutive,
            safe_checkpoint=_safe(ordered, total, options, consecutive),
            recall_test=_recall_test(
                collection, statuses, ordered, options, timeline.order_reliable
            ),
        )
    except ValidationError as exc:
        raise _invalid_assessment() from exc


def _history_too_large() -> CollectionError:
    return CollectionError(
        "screening_stopping_history_too_large",
        f"Reading more than {MAX_HISTORY_ROWS} retained decisions is refused; nothing was "
        "assessed.",
        413,
    )


def _recorded_decisions(
    connection: sqlite3.Connection, collection: PaperCollection, queue: ScreeningQueue
) -> tuple[list[RecordedDecision], list[str]]:
    """Walk each decided member's retained history back to its first decision at this revision.

    Checks what the data can check: newest history equals the current decision, decision
    revisions are contiguous, times never decrease as revisions increase, and no decision
    from an earlier collection revision (including a stale member's latest one) is later
    than one at this revision.
    """
    recorded: list[RecordedDecision] = []
    warnings: list[str] = []
    rows = 0
    earlier: tuple[datetime, str] | None = None
    for item in queue.items:
        latest = item.review
        if item.status == "stale" and latest is not None:
            if earlier is None or latest.updated_at > earlier[0]:
                earlier = (latest.updated_at, item.document_id)
            continue
        if item.status not in ("include", "exclude", "unsure") or latest is None:
            continue
        first, retained, newer = latest, latest.revision == 1, None
        expected, cursor, reached_earlier, warned = latest.revision, MAX_REVISION, False, False
        while not reached_earlier:
            page = _history_records(
                connection, collection, item.document_id, last_revision=cursor, limit=_HISTORY_PAGE
            )
            rows += len(page)
            if rows > MAX_HISTORY_ROWS:
                raise _history_too_large()
            for review in page:
                if review.revision != expected or (newer is None and review != latest):
                    raise _invalid_history()
                if newer is not None and review.updated_at > newer.updated_at and not warned:
                    warned = True
                    warnings.append(
                        f"Member {json.dumps(item.document_id)}: decision revision "
                        f"{newer.revision} has an earlier recorded time than revision "
                        f"{review.revision}."
                    )
                if review.collection_revision < collection.revision:
                    reached_earlier, retained = True, True
                    if earlier is None or review.updated_at > earlier[0]:
                        earlier = (review.updated_at, item.document_id)
                    break
                first, retained, newer = review, review.revision == 1, review
                expected -= 1
            if len(page) < _HISTORY_PAGE:
                break
            cursor = page[-1].revision - 1
        recorded.append(
            RecordedDecision(
                document_id=item.document_id,
                decision=latest.decision,
                first_decided_at=first.updated_at,
                decided_at=latest.updated_at,
                first_decision_retained=retained,
            )
        )
    if earlier is not None and recorded:
        current = min(recorded, key=lambda record: record.first_decided_at)
        if earlier[0] > current.first_decided_at:
            warnings.append(
                f"Member {json.dumps(earlier[1])} has a decision from an earlier collection "
                f"revision recorded after member {json.dumps(current.document_id)} was "
                f"decided at revision {collection.revision}."
            )
    return recorded, warnings


def assess_screening_stopping(
    screening: SQLitePaperScreening,
    collection_id: str,
    *,
    collection_revision: int,
    target_recall: float = DEFAULT_TARGET_RECALL,
    confidence: float = DEFAULT_CONFIDENCE,
    irrelevant_run: int | None = None,
    irrelevant_run_percent: int | None = None,
    min_screened_percent: int = DEFAULT_MIN_SCREENED_PERCENT,
    estimated_relevant: int | None = None,
    key_document_ids: Sequence[str] = (),
    sample_seed: int | None = None,
    sample_after: int | None = None,
    sample_urn_sha256: str | None = None,
) -> ScreeningStoppingAssessment:
    """Read one snapshot of an exact collection revision and assess when stopping is defensible.

    Read-only and advisory: no decision, history, collection, corpus or event write occurs,
    and no member is excluded, hidden or marked as screened.
    """
    try:
        options = StoppingOptions(
            collection_id=collection_id,
            collection_revision=collection_revision,
            target_recall=target_recall,
            confidence=confidence,
            irrelevant_run=irrelevant_run,
            irrelevant_run_percent=irrelevant_run_percent,
            min_screened_percent=min_screened_percent,
            estimated_relevant=estimated_relevant,
            key_document_ids=tuple(key_document_ids),
            sample_seed=sample_seed,
            sample_after=sample_after,
            sample_urn_sha256=sample_urn_sha256,
        )
    except (ValidationError, TypeError) as exc:
        raise CollectionError(
            "invalid_screening_stopping_request",
            "Invalid stopping parameters; check names, types, bounds and combinations.",
            422,
        ) from exc
    with screening.collections.resolved_collection(
        options.collection_id, expected_revision=options.collection_revision
    ) as (connection, collection):
        queue = screening._snapshot(connection, collection)
        recorded, warnings = _recorded_decisions(connection, collection, queue)
    result = build_assessment(collection, queue, recorded, warnings, options)
    if len(result.model_dump_json().encode("utf-8")) > MAX_STOPPING_RESPONSE_BYTES:
        raise CollectionError(
            "screening_stopping_too_large",
            "The complete assessment exceeds its UTF-8 byte limit; no partial assessment.",
            413,
        )
    return result
