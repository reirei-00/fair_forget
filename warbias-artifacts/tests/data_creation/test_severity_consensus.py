"""Final severity labels require independent agreement on individual brackets."""

from copy import deepcopy

import pytest

from warbias_kit.warbias.severity_consensus import bracket_consensus, validate_consensus

BRACKETS = [
    {
        "code": code,
        "label": label,
        "lower": lower,
        "upper": upper,
        "upper_inclusive": upper == 100,
    }
    for code, label, lower, upper in (
        ("very_low", "Very low", 0, 20),
        ("low", "Low", 20, 40),
        ("moderate", "Moderate", 40, 60),
        ("high", "High", 60, 80),
        ("very_high", "Very high", 80, 100),
    )
]
POLICY = {
    "target_scale_version": 2,
    "severity_brackets": BRACKETS,
    "consensus": {
        "method": "strict_majority",
        "minimum_support": 2,
        "input": "individual_rating_brackets",
    },
}


def rating(score, reviewer, version=2, case="C1"):
    return {
        "case_id": case,
        "rater_id": reviewer,
        "rating_id": f"{case}-{reviewer}",
        "score": score,
        "scale_version": version,
    }


@pytest.mark.parametrize(
    ("scores", "label"),
    [
        ([0, 0.1], "very_low"),
        ([0.2, 0.399], "low"),
        ([0.4, 0.599], "moderate"),
        ([0.6, 0.799], "high"),
        ([0.8, 1], "very_high"),
    ],
)
def test_two_independent_ratings_in_one_bracket_assign_a_label(scores, label):
    ratings = [rating(score, f"R{index}") for index, score in enumerate(scores)]
    before = deepcopy(ratings)
    result = bracket_consensus(ratings, POLICY)
    assert result["severity_label"] == label
    assert result["severity_label_status"] == "consensus"
    assert result["severity_label_support"] == result["severity_label_raters"] == 2
    assert result["severity_label_votes"][label] == 2
    assert sum(result["severity_label_votes"].values()) == 2
    assert len(result["severity_label_votes"]) == 5
    assert result["severity_needs_review"] is False
    assert result["severity_label_method"] == "bracket_consensus"
    assert ratings == before


@pytest.mark.parametrize("scores", [[0.241, 0.827], [0.36, 0.415], [0.666, 0.814]])
def test_two_different_brackets_remain_in_review_even_when_scores_are_close(scores):
    result = bracket_consensus([rating(score, f"R{i}") for i, score in enumerate(scores)], POLICY)
    assert result["severity_label"] is result["severity_label_name"] is None
    assert result["severity_label_status"] == "disagreement"
    assert result["severity_label_support"] == 1
    assert result["severity_label_raters"] == 2
    assert result["severity_needs_review"] is True


def test_three_ratings_can_form_a_strict_majority_different_from_the_mean_bracket():
    result = bracket_consensus([rating(0.2, "R1"), rating(0.2, "R2"), rating(1, "R3")], POLICY)
    assert result["severity_label"] == "low"
    assert result["severity_label_support"] == 2
    assert result["severity_label_raters"] == 3
    assert result["severity_label_votes"] == {
        "very_low": 0,
        "low": 2,
        "moderate": 0,
        "high": 0,
        "very_high": 1,
    }


@pytest.mark.parametrize("scores", [[0.1, 0.5, 0.9], [0.2, 0.3, 0.8, 0.9]])
def test_distinct_votes_and_even_ties_require_review(scores):
    result = bracket_consensus([rating(score, f"R{i}") for i, score in enumerate(scores)], POLICY)
    assert result["severity_label"] is None
    assert result["severity_label_status"] == "disagreement"
    assert result["severity_needs_review"] is True


def test_single_rating_and_unknown_version_keep_support_visible_and_label_pending():
    result = bracket_consensus([rating(0.4, "R1"), rating(0.45, "R2", version=None)], POLICY)
    assert result["severity_label"] is None
    assert result["severity_label_status"] == "insufficient_ratings"
    assert result["severity_label_support"] == result["severity_label_raters"] == 1
    assert result["severity_label_votes"]["moderate"] == 1


def test_empty_and_unversioned_records_have_distinct_pending_statuses():
    absent = bracket_consensus([], POLICY)
    unknown = bracket_consensus([rating(0.4, "R1", version=None)], POLICY)
    assert absent["severity_label_status"] == "not_annotated"
    assert unknown["severity_label_status"] == "insufficient_ratings"
    for result in (absent, unknown):
        assert result["severity_label"] is None
        assert result["severity_label_support"] == result["severity_label_raters"] == 0
        assert all(count == 0 for count in result["severity_label_votes"].values())


def test_repeated_reviewers_and_rating_ids_cannot_supply_support():
    source = rating(0.4, "R1")
    with pytest.raises(ValueError, match="distinct active ratings"):
        bracket_consensus([source, {**source, "rating_id": "new"}], POLICY)
    with pytest.raises(ValueError, match="distinct active ratings"):
        bracket_consensus([source, {**source, "rater_id": "R2"}], POLICY)


def test_ratings_from_different_cases_cannot_be_pooled():
    with pytest.raises(ValueError, match="one taxonomy case"):
        bracket_consensus([rating(0.4, "R1"), rating(0.4, "R2", case="C2")], POLICY)


@pytest.mark.parametrize("score", [-0.1, 1.1, True, float("nan"), float("inf"), "0.4"])
def test_invalid_scores_are_rejected_before_version_filtering(score):
    with pytest.raises(ValueError, match="finite numbers"):
        bracket_consensus([rating(score, "R1", version=None)], POLICY)


@pytest.mark.parametrize("version", [True, 2.0, "2", 0])
def test_unrecorded_versions_remain_unknown_and_invalid_versions_fail(version):
    with pytest.raises(ValueError, match="integers or null"):
        bracket_consensus([rating(0.4, "R1", version=version)], POLICY)


@pytest.mark.parametrize(
    "change",
    [
        {
            "consensus": {
                "method": "mean",
                "minimum_support": 2,
                "input": "individual_rating_brackets",
            }
        },
        {
            "consensus": {
                "method": "strict_majority",
                "minimum_support": 1,
                "input": "individual_rating_brackets",
            }
        },
        {
            "consensus": {
                "method": "strict_majority",
                "minimum_support": True,
                "input": "individual_rating_brackets",
            }
        },
        {
            "consensus": {
                "method": "strict_majority",
                "minimum_support": 2,
                "input": "case_mean_bracket",
            }
        },
        {"target_scale_version": None},
        {"severity_brackets": None},
    ],
)
def test_consensus_requires_an_explicit_supported_policy(change):
    with pytest.raises(ValueError):
        validate_consensus({**POLICY, **change})


def test_legacy_protocol_has_no_implicit_consensus_method():
    assert validate_consensus({}) is None
    assert bracket_consensus([rating(0.4, "R1")], {}) == {}
