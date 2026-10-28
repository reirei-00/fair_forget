"""Saved pilot analysis preserves actual assignments, missing reviews and author choice."""

from copy import deepcopy

import pytest

from fairforget.warbias.pilot_analysis import (
    analyse_ratings,
    describe_matrix,
    generation_cases,
    pilot_conditions,
)


def matrix_rows(count=4):
    return [
        {
            "matrix_item_id": f"unit-{index}",
            "case_id": f"case-{index}",
            "bias_type_code": "veteran" if index % 2 else "idp",
            "category_code": "employment",
            "target_code": "veteran_generic" if index % 2 else "idp_current",
            "target_uk": "Цільова група",
            "target_definition_uk": "Визначення групи",
            "gender_track": "neutral",
            "age_track": "unspecified",
            "stereotype_focus_uk": "Перевірювана асоціація",
            "stereotype_claim_uk": "Точне вихідне твердження",
            "context_uk": "Контекст",
            "harm_mechanism_uk": "Механізм шкоди",
        }
        for index in range(count)
    ]


def panel():
    rows = matrix_rows()
    cells = [
        ("lapa12b", "zero_shot"),
        ("lapa12b", "sequential"),
        ("mamay27b", "zero_shot"),
        ("mamay27b", "sequential"),
    ]
    assignments = [
        {
            "condition_id": f"item-{index}",
            "matrix_item_id": row["matrix_item_id"],
            "case_id": row["case_id"],
            "candidate_id": candidate,
            "strategy": strategy,
        }
        for index, (row, (candidate, strategy)) in enumerate(zip(rows, cells, strict=True))
    ]
    conditions = pilot_conditions(rows, assignments)
    ratings = [
        {"condition_id": row["condition_id"], "rater_id": rater, "score": score}
        for row, score in zip(conditions, (2, 5, 5, 3), strict=True)
        for rater in ("rater_01", "rater_02")
    ]
    protocol = {
        "selected_condition": {"candidate_id": "mamay27b", "strategy": "sequential"},
        "selection": {"raters": 2, "record_type": "fixture_author_choice"},
    }
    return conditions, ratings, protocol


def test_matrix_adapter_preserves_profiles_and_missing_historical_claim():
    rows = matrix_rows(2)
    del rows[0]["stereotype_claim_uk"]
    original = deepcopy(rows)
    cases = generation_cases(rows)
    assert cases[0]["id"] == cases[0]["matrix_id"] == "unit-0"
    assert cases[0]["claim"] is None
    assert cases[0]["required_stereotype_claim_uk"] is None
    assert cases[1]["required_stereotype_subtype_uk"] == rows[1]["stereotype_focus_uk"]
    assert cases[1]["target_group"] == "veteran"
    assert rows == original
    summary = describe_matrix(rows)
    assert summary["matrix_rows"] == summary["case_families"] == 2
    assert summary["missing_exact_claims"] == 1
    assert summary["dimensions"]["bias_type_code"] == {"idp": 1, "veteran": 1}


def test_factorial_pilot_preserves_profile_units_with_shared_case_family():
    rows = matrix_rows(2)
    rows[1]["case_id"] = rows[0]["case_id"]
    conditions = pilot_conditions(rows)
    assert len(conditions) == 8
    assert len({row["condition_id"] for row in conditions}) == 8
    assert {row["case_id"] for row in conditions} == {"case-0"}


@pytest.mark.parametrize("fault", ["missing", "case", "candidate", "duplicate"])
def test_explicit_assignments_reject_stale_or_incomplete_cases(fault):
    conditions, _, _ = panel()
    if fault == "missing":
        conditions.pop()
    elif fault == "case":
        conditions[0]["case_id"] = "another-case"
    elif fault == "candidate":
        conditions[0]["candidate_id"] = "unknown-model"
    else:
        conditions.append(deepcopy(conditions[0]))
    with pytest.raises(ValueError):
        pilot_conditions(matrix_rows(), conditions)


def test_score_ranks_do_not_replace_the_recorded_author_choice():
    conditions, ratings, protocol = panel()
    result = analyse_ratings(conditions, ratings, protocol)
    assert len(result["highest_mean_conditions"]) == 2
    assert {row["mean_score"] for row in result["highest_mean_conditions"]} == {5}
    assert result["recorded_selection"]["candidate_id"] == "mamay27b"
    assert result["recorded_selection"]["strategy"] == "sequential"
    assert result["recorded_selection"]["rank"] == 3
    assert result["coverage"]["ratings"] == result["coverage"]["expected_ratings"] == 8
    assert result["matched_case_sets"] is False
    assert result["statistical_superiority_claim"] is False


def test_missing_reviews_stay_missing_instead_of_becoming_zero_scores():
    conditions, ratings, protocol = panel()
    ratings = [row for row in ratings if row["condition_id"] != "item-0"]
    result = analyse_ratings(conditions, ratings, protocol)
    missing = next(
        row
        for row in result["condition_summary"]
        if row["candidate_id"] == "lapa12b" and row["strategy"] == "zero_shot"
    )
    assert missing["mean_score"] is None and missing["rank"] is None
    assert result["coverage"]["rated_items"] == 3
    assert result["coverage"]["missing_condition_ids"] == ["item-0"]
    assert result["coverage"]["ratings"] == 6


def test_unrated_panel_has_no_rank_or_derived_winner():
    conditions, _, protocol = panel()
    result = analyse_ratings(conditions, [], protocol)
    assert result["highest_mean_conditions"] == []
    assert result["recorded_selection"]["mean_score"] is None
    assert result["recorded_selection"]["rank"] is None
    assert result["coverage"]["rated_items"] == result["coverage"]["ratings"] == 0
    assert len(result["coverage"]["missing_condition_ids"]) == 4


@pytest.mark.parametrize("score", [True, 0, 6, float("nan"), float("inf"), "invalid"])
def test_invalid_scores_are_rejected(score):
    conditions, ratings, protocol = panel()
    ratings[0]["score"] = score
    with pytest.raises(ValueError, match="finite numbers"):
        analyse_ratings(conditions, ratings, protocol)


@pytest.mark.parametrize("fault", ["duplicate", "unknown", "case", "choice", "raters"])
def test_ratings_and_choice_must_match_the_recorded_panel(fault):
    conditions, ratings, protocol = panel()
    if fault == "duplicate":
        ratings.append(deepcopy(ratings[0]))
    elif fault == "unknown":
        ratings[0]["condition_id"] = "not-in-pilot"
    elif fault == "case":
        ratings[0]["case_id"] = "not-the-case"
    elif fault == "choice":
        protocol["selected_condition"]["candidate_id"] = "unknown-model"
    else:
        protocol["selection"]["raters"] = 1
    with pytest.raises(ValueError):
        analyse_ratings(conditions, ratings, protocol)
