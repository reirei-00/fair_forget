"""Agreement uses individual judgments and keeps missing decisions unavailable."""

from copy import deepcopy

import pytest

from fairforget.warbias.annotation_agreement import (
    agreement_report,
    binary_agreement,
    compare_reviews,
)


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        ([25, 25, 25, 25], 0),
        ([0, 50, 50, 0], -1),
        ([50, 0, 0, 50], 1),
        ([100, 0, 0, 0], 1),
        ([0, 0, 0, 100], 1),
    ],
)
def test_ac1_known_binary_tables(counts, expected):
    assert binary_agreement(counts)["ac1"] == expected


def test_missing_positive_class_is_not_perfect_positive_agreement():
    result = binary_agreement([100, 0, 0, 0])
    assert result["positive_agreement"] is None
    assert result["negative_agreement"] == 1
    assert binary_agreement([0, 0, 0, 0])["ac1"] is None


@pytest.fixture
def judgments():
    common = {
        "target_group": "idp",
        "text_sha256": "text",
        "context_sha256": "context",
        "model": "jev-1.14.0",
    }
    predictions = [
        {
            **common,
            "id": "x",
            "case_family": "a",
            "has_problem": True,
            "problem_types": ["unnatural"],
        },
        {**common, "id": "y", "case_family": "b", "has_problem": False, "problem_types": []},
    ]
    reviews = [
        {
            **prediction,
            "reviewer": person,
            "has_problem": prediction["has_problem"] if person == "A" else False,
            "problem_types": prediction["problem_types"] if person == "A" else [],
        }
        for prediction in predictions
        for person in ["A", "B"]
    ]
    return predictions, reviews


def test_pair_coefficients_alpha_and_set_agreement_have_known_values(judgments):
    predictions, reviews = judgments
    result = agreement_report(predictions, reviews, ["unnatural"], resamples=50)["overall"]
    assert result["jev_human"]["mean_pair_ac1"] == pytest.approx(0.6)
    assert result["jev_human"]["item_weighted_observed_agreement"]["value"] == 0.75
    assert result["human_human"]["mean_pair_ac1"] == pytest.approx(0.2)
    assert result["human_human"]["nominal_alpha"] == 0
    assert result["conditional_jaccard"]["value"] == 0.5
    assert result["coverage"]["both_empty_issue_pairs"] == 2


def test_bootstrap_is_reproducible_with_whole_case_families(judgments):
    a, b = judgments
    first = agreement_report(a, b, ["unnatural"], resamples=40, seed=17)
    assert first == agreement_report(a, b, ["unnatural"], resamples=40, seed=17)
    assert first["bootstrap"]["unit"] == "semantic_case_family"


def test_disagreements_remain_individual_and_missing_judgments_are_not_negative(judgments):
    predictions, reviews = judgments
    reviews.append(
        {**deepcopy(reviews[0]), "reviewer": "C", "has_problem": None, "problem_types": None}
    )
    result = agreement_report(predictions, reviews, ["unnatural"], resamples=20)["overall"]
    assert result["coverage"]["human_review_rows"] == 5
    assert result["coverage"]["paired_binary_judgments"] == 4
    assert result["coverage"]["unknown_issue_sets"] == 1


@pytest.mark.parametrize("field", ["text_sha256", "context_sha256", "case_family"])
def test_text_or_context_mismatch_prevents_agreement(judgments, field):
    predictions, reviews = judgments
    reviews[0][field] = "changed"
    with pytest.raises(ValueError, match="different text or context"):
        agreement_report(predictions, reviews, ["unnatural"], resamples=10)


def test_duplicate_person_item_is_not_an_additional_vote(judgments):
    predictions, reviews = judgments
    reviews.append(deepcopy(reviews[0]))
    with pytest.raises(ValueError, match="one independent judgment"):
        agreement_report(predictions, reviews, ["unnatural"], resamples=10)


def test_paired_comparison_has_zero_change_for_identical_predictions(judgments):
    predictions, reviews = judgments
    result = compare_reviews(predictions, predictions, reviews, ["unnatural"], resamples=40)
    assert result["mean_pair_ac1_change"]["value"] == 0
    assert result["mean_pair_ac1_change"]["ci95"] == {
        "low": 0,
        "high": 0,
        "valid_resamples": 40,
    }
    assert result["baseline"]["binary_pairs"]["positive_agreement"] == pytest.approx(2 / 3)


def test_prompt_comparison_rejects_different_item_sets(judgments):
    predictions, reviews = judgments
    with pytest.raises(ValueError, match="complete human item cohort"):
        compare_reviews(predictions, predictions[:1], reviews, ["unnatural"], resamples=10)


def test_missing_human_labels_do_not_become_zero_prompt_change(judgments):
    predictions, reviews = judgments
    for row in reviews:
        row.update(has_problem=None, problem_types=None)
    result = compare_reviews(predictions, predictions, reviews, ["unnatural"], resamples=20)
    assert result["mean_pair_ac1_change"]["value"] is None
    assert result["mean_pair_ac1_change"]["ci95"]["valid_resamples"] == 0
