"""Pure two-reviewer agreement statistics for categorical screening decisions.

Cohen's kappa, the Fleiss-Cohen-Everitt (1969) large-sample standard error with a Wald
95% interval, prevalence- and bias-adjusted kappa (PABAK) and the descriptive
Landis-Koch (1977) band. Exact rational arithmetic is used until the final square root.
No NumPy, SciPy, model, network or storage access is involved.
"""

from collections.abc import Sequence
from fractions import Fraction
from math import sqrt
from statistics import NormalDist
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

DISPLAY_DECIMALS = 6
MAX_CATEGORIES = 3
Z_95 = NormalDist().inv_cdf(0.975)
KappaUndefinedReason = Literal["no_shared_decisions", "chance_agreement_is_one"]
LandisKochBand = Literal["poor", "slight", "fair", "moderate", "substantial", "almost_perfect"]
_BAND_UPPER_BOUNDS: tuple[tuple[Fraction, LandisKochBand], ...] = (
    (Fraction(1, 5), "slight"),
    (Fraction(2, 5), "fair"),
    (Fraction(3, 5), "moderate"),
    (Fraction(4, 5), "substantial"),
)
_Probability = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
_Coefficient = Annotated[float, Field(ge=-1.0, le=1.0, allow_inf_nan=False)]


def landis_koch_band(kappa: Fraction | float) -> LandisKochBand:
    """Map kappa to Landis and Koch's labels: <0, 0-0.20, 0.21-0.40, ..., 0.81-1.00.

    Cut points are applied to the unrounded value with upper bounds inclusive, so exactly
    0.40 is "fair". Floats are read from their shortest decimal representation.
    """
    value = Fraction(str(kappa)) if isinstance(kappa, float) else kappa
    if value < 0:
        return "poor"
    for upper, band in _BAND_UPPER_BOUNDS:
        if value <= upper:
            return band
    return "almost_perfect"


def _display(value: Fraction | float) -> float:
    return round(float(value), DISPLAY_DECIMALS) + 0.0


def _table(
    categories: Sequence[str], confusion: Sequence[Sequence[int]]
) -> tuple[tuple[str, ...], tuple[tuple[int, ...], ...]]:
    names = tuple(categories)
    rows = tuple(tuple(row) for row in confusion)
    size = len(names)
    if (
        not 2 <= size <= MAX_CATEGORIES
        or len(set(names)) != size
        or len(rows) != size
        or any(len(row) != size for row in rows)
        or any(type(count) is not int or count < 0 for row in rows for count in row)
    ):
        raise ValueError("Use 2-3 distinct categories and a square table of nonnegative counts.")
    return names, rows


def _statistics(categories: Sequence[str], confusion: Sequence[Sequence[int]]) -> dict[str, object]:
    names, rows = _table(categories, confusion)
    size = len(names)
    compared = sum(map(sum, rows))
    agreements = sum(rows[index][index] for index in range(size))
    result: dict[str, object] = {
        "categories": names,
        "confusion": rows,
        "compared": compared,
        "agreements": agreements,
        "disagreements": compared - agreements,
        "observed_agreement": None,
        "expected_agreement": None,
        "kappa": None,
        "kappa_undefined_reason": None,
        "kappa_standard_error": None,
        "kappa_ci95_low": None,
        "kappa_ci95_high": None,
        "pabak": None,
        "landis_koch_band": None,
    }
    if compared == 0:
        return {**result, "kappa_undefined_reason": "no_shared_decisions"}
    p = [[Fraction(count, compared) for count in row] for row in rows]
    first = [sum(row, Fraction(0)) for row in p]
    second = [sum((p[row][column] for row in range(size)), Fraction(0)) for column in range(size)]
    observed = Fraction(agreements, compared)
    expected = sum((first[index] * second[index] for index in range(size)), Fraction(0))
    pabak = (size * observed - 1) / (size - 1)
    result.update(
        observed_agreement=_display(observed),
        expected_agreement=_display(expected),
        pabak=_display(pabak),
    )
    if expected == 1:
        # Both reviewers used one and the same category for every shared paper: 0 / 0.
        return {**result, "kappa_undefined_reason": "chance_agreement_is_one"}
    kappa = (observed - expected) / (1 - expected)
    complement = 1 - kappa
    diagonal = sum(
        (
            p[index][index] * (1 - (first[index] + second[index]) * complement) ** 2
            for index in range(size)
        ),
        Fraction(0),
    )
    off_diagonal = complement**2 * sum(
        (
            p[row][column] * (second[row] + first[column]) ** 2
            for row in range(size)
            for column in range(size)
            if row != column
        ),
        Fraction(0),
    )
    correction = (kappa - expected * complement) ** 2
    variance = (diagonal + off_diagonal - correction) / (compared * (1 - expected) ** 2)
    error = sqrt(max(float(variance), 0.0))
    return {
        **result,
        "kappa": _display(kappa),
        "kappa_standard_error": _display(error),
        "kappa_ci95_low": _display(max(-1.0, float(kappa) - Z_95 * error)),
        "kappa_ci95_high": _display(min(1.0, float(kappa) + Z_95 * error)),
        "landis_koch_band": landis_koch_band(kappa),
    }


class AgreementStatistics(BaseModel):
    """Agreement of two reviewers on one categorical scale; rows are the first reviewer.

    Every value is recomputed from ``confusion`` on validation, so a stored or edited
    report cannot carry statistics that disagree with its own counts.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    categories: tuple[str, ...] = Field(min_length=2, max_length=MAX_CATEGORIES)
    confusion: tuple[tuple[int, ...], ...] = Field(min_length=2, max_length=MAX_CATEGORIES)
    compared: int = Field(strict=True, ge=0)
    agreements: int = Field(strict=True, ge=0)
    disagreements: int = Field(strict=True, ge=0)
    observed_agreement: _Probability | None
    expected_agreement: _Probability | None
    kappa: _Coefficient | None
    kappa_undefined_reason: KappaUndefinedReason | None
    kappa_standard_error: Annotated[float, Field(ge=0.0, allow_inf_nan=False)] | None
    kappa_ci95_low: _Coefficient | None
    kappa_ci95_high: _Coefficient | None
    pabak: _Coefficient | None
    landis_koch_band: LandisKochBand | None

    @model_validator(mode="after")
    def recomputed(self) -> Self:
        if self.model_dump() != _statistics(self.categories, self.confusion):
            raise ValueError("Agreement statistics must be recomputed from their own counts.")
        return self


def agreement_statistics(
    categories: Sequence[str], confusion: Sequence[Sequence[int]]
) -> AgreementStatistics:
    """Cohen's kappa, its FCE-1969 standard error and Wald 95% CI, PABAK and band.

    p_o = sum_i n_ii / n; p_e = sum_i p_i. * p_.i; kappa = (p_o - p_e) / (1 - p_e).
    Var(kappa) = [sum_i p_ii (1 - (p_i. + p_.i)(1 - kappa))^2
                  + (1 - kappa)^2 sum_{i != j} p_ij (p_.i + p_j.)^2
                  - (kappa - p_e (1 - kappa))^2] / (n (1 - p_e)^2).
    CI = kappa -/+ 1.959964 * SE, clipped to [-1, 1]. PABAK = (k p_o - 1) / (k - 1) for the
    k categories of the scale. Kappa is None, with a reason, when n = 0 or p_e = 1.
    """
    return AgreementStatistics.model_validate(_statistics(categories, confusion))
