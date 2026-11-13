"""Hand-computed agreement statistics: kappa, FCE-1969 SE, Wald CI, PABAK and bands."""

import math

import pytest
from pydantic import ValidationError

from storage.screening_agreement import (
    Z_95,
    AgreementStatistics,
    agreement_statistics,
    landis_koch_band,
)

THREE = ("include", "exclude", "unsure")
TWO = ("include_or_unsure", "exclude")


def test_perfect_agreement_has_kappa_one_and_a_degenerate_interval() -> None:
    # p_o = 1; p_e = 0.4 * 0.4 + 0.6 * 0.6 = 0.52; kappa = 0.48 / 0.48 = 1.
    # FCE: diagonal term = sum p_ii = 1, off-diagonal 0, correction (1 - 0)^2 = 1 -> Var 0.
    stats = agreement_statistics(THREE, [[4, 0, 0], [0, 6, 0], [0, 0, 0]])
    assert (stats.compared, stats.agreements, stats.disagreements) == (10, 10, 0)
    assert stats.observed_agreement == 1.0 and stats.expected_agreement == 0.52
    assert stats.kappa == 1.0 and stats.kappa_undefined_reason is None
    assert stats.kappa_standard_error == 0.0
    assert (stats.kappa_ci95_low, stats.kappa_ci95_high) == (1.0, 1.0)
    assert stats.pabak == 1.0 and stats.landis_koch_band == "almost_perfect"


def test_chance_level_agreement_has_kappa_zero_and_scale_dependent_pabak() -> None:
    # p_o = 0.5, p_e = 0.25 + 0.25 = 0.5, kappa = 0. FCE with kappa = 0: diagonal terms
    # 0.25 * (1 - 1)^2 each = 0; off-diagonal 0.25 * 1^2 + 0.25 * 1^2 = 0.5; correction
    # (0 - 0.5)^2 = 0.25; Var = (0.5 - 0.25) / (8 * 0.25) = 0.125.
    binary = agreement_statistics(TWO, [[2, 2], [2, 2]])
    three = agreement_statistics(THREE, [[2, 2, 0], [2, 2, 0], [0, 0, 0]])
    for stats in (binary, three):
        assert stats.observed_agreement == 0.5 and stats.expected_agreement == 0.5
        assert stats.kappa == 0.0 and stats.landis_koch_band == "slight"
        assert stats.kappa_standard_error == pytest.approx(math.sqrt(0.125), abs=1e-6)
        assert stats.kappa_ci95_low == pytest.approx(-Z_95 * math.sqrt(0.125), abs=1e-6)
        assert stats.kappa_ci95_high == pytest.approx(0.692952, abs=1e-6)
    # PABAK = (k p_o - 1) / (k - 1) depends on the scale's category count k.
    assert binary.pabak == 0.0
    assert three.pabak == 0.25


def test_published_two_by_two_example_matches_the_fce_standard_error() -> None:
    # n = 50; p_o = 35 / 50 = 0.7; rows (0.5, 0.5), columns (0.6, 0.4); p_e = 0.5; kappa 0.4.
    # A = 0.4 (1 - 1.1 * 0.6)^2 + 0.3 (1 - 0.9 * 0.6)^2 = 0.10972
    # B = 0.6^2 (0.1 * 1.1^2 + 0.2 * 0.9^2) = 0.10188; C = (0.4 - 0.5 * 0.6)^2 = 0.01
    # Var = (A + B - C) / (50 * 0.25) = 0.016128.
    stats = agreement_statistics(("include", "exclude"), [[20, 5], [10, 15]])
    assert stats.kappa == 0.4
    assert stats.kappa_standard_error == pytest.approx(math.sqrt(0.016128), abs=1e-6)
    assert stats.kappa_ci95_low == pytest.approx(0.4 - Z_95 * math.sqrt(0.016128), abs=1e-6)
    assert stats.kappa_ci95_high == pytest.approx(0.648908, abs=1e-6)
    assert stats.pabak == pytest.approx(0.4, abs=1e-12)
    assert stats.landis_koch_band == "fair"  # 0.40 belongs to 0.21-0.40


@pytest.mark.parametrize(
    ("table", "kappa", "error"),
    [
        # Reference values agree with statsmodels 0.14 cohens_kappa (also FCE 1969). Using
        # (p_i. + p_.j)^2 instead of (p_.i + p_j.)^2 would give 0.221345 and 0.130401.
        ([[4, 1, 0], [1, 5, 0], [0, 1, 0]], (0.75 - 67 / 144) / (1 - 67 / 144), 0.219001),
        ([[7, 3, 1], [2, 9, 4], [0, 2, 5]], 0.442254, 0.128969),
    ],
)
def test_asymmetric_tables_match_reference_values_and_are_rater_symmetric(
    table: list[list[int]], kappa: float, error: float
) -> None:
    stats = agreement_statistics(THREE, table)
    assert stats.kappa == pytest.approx(kappa, abs=1e-6)
    assert stats.kappa_standard_error == pytest.approx(error, abs=1e-6)
    transposed = agreement_statistics(THREE, [list(column) for column in zip(*table, strict=True)])
    assert transposed.kappa == stats.kappa
    assert transposed.kappa_standard_error == stats.kappa_standard_error


def test_demo_table_partitions_twelve_papers() -> None:
    # r1 rows / r2 columns: p_o = 9/12, p_e = (5*5 + 6*7 + 1*0) / 144 = 67/144.
    stats = agreement_statistics(THREE, [[4, 1, 0], [1, 5, 0], [0, 1, 0]])
    assert (stats.compared, stats.agreements, stats.disagreements) == (12, 9, 3)
    assert stats.observed_agreement == 0.75
    assert stats.expected_agreement == pytest.approx(67 / 144, abs=1e-6)
    assert stats.pabak == 0.625 and stats.landis_koch_band == "moderate"


def test_complete_disagreement_is_poor_and_clipped_to_minus_one() -> None:
    stats = agreement_statistics(("include", "exclude"), [[0, 5], [5, 0]])
    assert stats.kappa == -1.0 and stats.pabak == -1.0
    assert (stats.kappa_ci95_low, stats.kappa_ci95_high) == (-1.0, -1.0)
    assert stats.landis_koch_band == "poor"


@pytest.mark.parametrize(
    ("table", "reason", "observed", "pabak"),
    [
        ([[5, 0, 0], [0, 0, 0], [0, 0, 0]], "chance_agreement_is_one", 1.0, 1.0),
        ([[0, 0, 0], [0, 0, 0], [0, 0, 7]], "chance_agreement_is_one", 1.0, 1.0),
        ([[0, 0, 0], [0, 0, 0], [0, 0, 0]], "no_shared_decisions", None, None),
    ],
)
def test_undefined_kappa_is_an_explicit_null_with_a_reason(
    table: list[list[int]], reason: str, observed: float | None, pabak: float | None
) -> None:
    stats = agreement_statistics(THREE, table)
    assert stats.kappa is None and stats.kappa_undefined_reason == reason
    assert stats.kappa_standard_error is None
    assert stats.kappa_ci95_low is None and stats.kappa_ci95_high is None
    assert stats.landis_koch_band is None
    assert stats.observed_agreement == observed and stats.pabak == pabak


@pytest.mark.parametrize(
    ("kappa", "band"),
    [
        (-0.01, "poor"),
        (0.0, "slight"),
        (0.2, "slight"),
        (0.200001, "fair"),
        (0.4, "fair"),
        (0.6, "moderate"),
        (0.8, "substantial"),
        (0.81, "almost_perfect"),
        (1.0, "almost_perfect"),
    ],
)
def test_landis_koch_upper_bounds_are_inclusive(kappa: float, band: str) -> None:
    assert landis_koch_band(kappa) == band


def test_forged_statistics_cannot_be_loaded_beside_their_counts() -> None:
    stats = agreement_statistics(("include", "exclude"), [[20, 5], [10, 15]])
    forged = {**stats.model_dump(), "kappa": 0.9}
    with pytest.raises(ValidationError):
        AgreementStatistics.model_validate(forged)
    assert AgreementStatistics.model_validate_json(stats.model_dump_json()) == stats


@pytest.mark.parametrize(
    ("categories", "table"),
    [
        (("include",), [[1]]),
        (("a", "b", "c", "d"), [[1] * 4] * 4),
        (("a", "a"), [[1, 0], [0, 1]]),
        (("a", "b"), [[1, 0, 0], [0, 1, 0]]),
        (("a", "b"), [[1, -1], [0, 1]]),
        (("a", "b"), [[1, True], [0, 1]]),
        (("a", "b"), [[1.0, 0], [0, 1]]),
    ],
)
def test_invalid_tables_are_rejected(categories: tuple[str, ...], table: list[list[int]]) -> None:
    with pytest.raises(ValueError, match="square table"):
        agreement_statistics(categories, table)
