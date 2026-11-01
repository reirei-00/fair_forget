"""Severity brackets cover the scale and keep pending assignments explicit."""

import math
from copy import deepcopy

import pytest

from warbias_kit.warbias.severity_brackets import bracket_for_score, validate_brackets

BRACKETS = [
    {"code": "very_low", "label": "Very low", "lower": 0, "upper": 20, "upper_inclusive": False},
    {"code": "low", "label": "Low", "lower": 20, "upper": 40, "upper_inclusive": False},
    {"code": "moderate", "label": "Moderate", "lower": 40, "upper": 60, "upper_inclusive": False},
    {"code": "high", "label": "High", "lower": 60, "upper": 80, "upper_inclusive": False},
    {"code": "very_high", "label": "Very high", "lower": 80, "upper": 100, "upper_inclusive": True},
]


@pytest.mark.parametrize(
    ("score", "code"),
    [
        (0, "very_low"),
        (20, "low"),
        (40, "moderate"),
        (60, "high"),
        (80, "very_high"),
        (100, "very_high"),
    ],
)
def test_exact_boundaries_use_the_inclusive_lower_bound(score, code):
    result = bracket_for_score(score, BRACKETS)
    assert result["code"] == code
    assert result == next(row for row in BRACKETS if row["code"] == code)


@pytest.mark.parametrize(
    ("edge", "lower_code", "upper_code"),
    [
        (20, "very_low", "low"),
        (40, "low", "moderate"),
        (60, "moderate", "high"),
        (80, "high", "very_high"),
    ],
)
def test_real_values_near_a_boundary_remain_in_their_interval(edge, lower_code, upper_code):
    assert bracket_for_score(edge - 1e-10, BRACKETS)["code"] == lower_code
    assert bracket_for_score(edge + 1e-10, BRACKETS)["code"] == upper_code


@pytest.mark.parametrize(
    ("edge", "code"), [(20, "low"), (40, "moderate"), (60, "high"), (80, "very_high")]
)
def test_adjacent_float_roundoff_keeps_boundary_means_in_the_upper_interval(edge, code):
    assert bracket_for_score(math.nextafter(edge, -math.inf), BRACKETS)["code"] == code
    assert bracket_for_score(math.nextafter(edge, math.inf), BRACKETS)["code"] == code


def test_pending_and_unconfigured_brackets_stay_null():
    assert bracket_for_score(None, BRACKETS) is None
    assert bracket_for_score(40, None) is None
    assert bracket_for_score(None, None) is None


def test_lookup_preserves_original_configuration_and_returns_a_copy():
    before = deepcopy(BRACKETS)
    result = bracket_for_score(40, BRACKETS)
    result["label"] = "Changed"
    assert BRACKETS == before


@pytest.mark.parametrize("score", [-1e-10, 100 + 1e-10, True, "40", float("nan"), float("inf")])
def test_invalid_scores_are_rejected(score):
    with pytest.raises(ValueError, match="finite numbers"):
        bracket_for_score(score, BRACKETS)


@pytest.mark.parametrize(
    "change",
    [
        lambda rows: rows.pop(),
        lambda rows: rows.reverse(),
        lambda rows: rows[1].update(code=rows[0]["code"]),
        lambda rows: rows[1].update(label=rows[0]["label"]),
        lambda rows: rows[0].update(lower=1),
        lambda rows: rows[-1].update(upper=99),
        lambda rows: rows[1].update(lower=21),
        lambda rows: rows[1].update(lower=19),
        lambda rows: rows[2].update(upper=rows[2]["lower"]),
        lambda rows: rows[0].update(upper_inclusive=True),
        lambda rows: rows[-1].update(upper_inclusive=False),
        lambda rows: rows[-1].update(upper_inclusive=1),
        lambda rows: rows[1].update(lower=float("nan")),
        lambda rows: rows[-1].update(upper=float("inf")),
        lambda rows: rows[0].update(lower=False),
        lambda rows: rows[0].update(code=""),
        lambda rows: rows[0].update(label=" "),
        lambda rows: rows[0].pop("upper_inclusive"),
    ],
)
def test_invalid_partitions_are_rejected_even_for_pending_scores(change):
    rows = deepcopy(BRACKETS)
    change(rows)
    with pytest.raises(ValueError):
        validate_brackets(rows)
    with pytest.raises(ValueError):
        bracket_for_score(None, rows)


@pytest.mark.parametrize("brackets", [[], {}, "five brackets"])
def test_declared_brackets_require_five_records(brackets):
    with pytest.raises(ValueError, match="five severity brackets"):
        bracket_for_score(40, brackets)
