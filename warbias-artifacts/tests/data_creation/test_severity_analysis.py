"""Decision explanations and CDFs preserve rating eligibility and case weights."""

from copy import deepcopy
from statistics import fmean

import pytest

from warbias_kit.io import digest
from warbias_kit.warbias.severity import assign_taxonomy_severity
from warbias_kit.warbias.severity_analysis import severity_decisions, severity_distribution

POLICY = {
    "schema_version": 1,
    "aggregation": "mean",
    "minimum_ratings": 2,
    "target_scale_version": 2,
    "disagreement_threshold": 0.4,
}


def assigned_cases(specifications, *, protocol=POLICY):
    taxonomy, ratings, evidence = [], [], []
    for identity, values in specifications.items():
        taxonomy.append(
            {
                "public_case_id": identity,
                "target_group": "idp",
                "claim_uk": "[TARGET] не хочуть працювати.",
                "claim_en": "[TARGET] do not want to work.",
            }
        )
        examples = ["ВПО не хочуть працювати."]
        evidence.append(
            {"case_id": identity, "examples_uk": examples, "examples_sha256": digest(examples)}
        )
        for index, (score, version) in enumerate(values):
            ratings.append(
                {
                    "case_id": identity,
                    "rating_id": f"{identity}-rating-{index}",
                    "rater_id": f"R{index}",
                    "score": score,
                    "scale_version": version,
                }
            )
    return assign_taxonomy_severity(taxonomy, ratings, evidence, protocol)


def triplets(counts):
    return [
        {"id": f"{identity}-{index}", "case_id": identity, "actor": "idp"}
        for identity, count in counts.items()
        for index in range(count)
    ]


def test_decisions_show_sum_mean_and_disagreement_without_changing_assignment():
    taxonomy = assigned_cases({"C1": [(0.241, 2), (0.827, 2)]})
    before = deepcopy(taxonomy)
    row = severity_decisions(taxonomy, POLICY)[0]
    assert row["eligible_scores_100"] == pytest.approx([24.1, 82.7])
    assert row["eligible_sum_100"] == pytest.approx(106.8)
    assert row["eligible_mean_100"] == pytest.approx(53.4)
    assert row["severity_score"] == pytest.approx(53.4)
    assert row["rating_gap_100"] == pytest.approx(58.6)
    assert row["severity_status"] == "high_disagreement"
    assert row["excluded_ratings"] == []
    assert taxonomy == before


def test_pending_case_shows_eligible_arithmetic_and_excluded_version():
    taxonomy = assigned_cases({"C1": [(0.521, None), (0.616, 2)], "C2": [(0.4, 2)]})
    first, direct = severity_decisions(taxonomy, POLICY)
    assert first["eligible_scores_100"] == pytest.approx([61.6])
    assert first["eligible_n"] == 1
    assert first["eligible_sum_100"] == pytest.approx(61.6)
    assert first["excluded_ratings"][0]["score_100"] == pytest.approx(52.1)
    assert first["excluded_ratings"][0]["scale_version"] is None
    assert first["eligible_mean_100"] is first["severity_score"] is None
    assert first["rating_gap_100"] is None
    assert direct["eligible_sum_100"] == 40
    assert direct["eligible_mean_100"] is direct["rating_gap_100"] is None


def test_decisions_include_authorized_unversioned_scores_without_relabeling():
    policy = {**POLICY, "include_unversioned_ratings": True}
    taxonomy = assigned_cases({"C1": [(0.5, None), (0.7, 2), (1, 1)]}, protocol=policy)
    before = deepcopy(taxonomy)
    row = severity_decisions(taxonomy, policy)[0]
    assert row["eligible_scores_100"] == [50, 70]
    assert row["eligible_n"] == 2 and row["eligible_unversioned_n"] == 1
    assert row["eligible_mean_100"] == row["severity_mean_score"] == 60
    assert row["severity_level"] == 4
    assert [rating["scale_version"] for rating in row["excluded_ratings"]] == [1]
    assert taxonomy == before
    taxonomy[0]["severity_unversioned_n"] = 0
    with pytest.raises(ValueError, match="eligible rating arithmetic"):
        severity_decisions(taxonomy, policy)


def test_decisions_require_the_assignment_eligibility_policy():
    taxonomy = assigned_cases({"C1": [(0.2, 2), (0.4, 2)]})
    with pytest.raises(ValueError, match="rating policy differs"):
        severity_decisions(taxonomy, {**POLICY, "include_unversioned_ratings": True})


def test_zero_is_a_scored_decision_and_missing_annotations_are_empty():
    taxonomy = assigned_cases({"zero": [(0, 2), (0, 2)], "missing": []})
    zero, missing = severity_decisions(taxonomy, POLICY)
    assert zero["eligible_mean_100"] == zero["severity_score"] == zero["rating_gap_100"] == 0
    assert missing["eligible_scores_100"] == []
    assert missing["eligible_sum_100"] == 0 and missing["eligible_n"] == 0
    assert missing["eligible_mean_100"] is missing["rating_gap_100"] is None


@pytest.mark.parametrize("change", [{"severity_score": 15}, {"severity_n": 3}])
def test_decision_rejects_arithmetic_that_differs_from_assignment(change):
    taxonomy = assigned_cases({"C1": [(0.2, 2), (0.4, 2)]})
    taxonomy[0].update(change)
    with pytest.raises(ValueError, match="eligible rating arithmetic"):
        severity_decisions(taxonomy, POLICY)


@pytest.mark.parametrize("field", ["rater_id", "rating_id"])
def test_decision_rejects_duplicate_rater_or_rating_id(field):
    taxonomy = assigned_cases({"C1": [(0.2, 2), (0.4, 2)]})
    taxonomy[0]["severity_ratings"][1][field] = taxonomy[0]["severity_ratings"][0][field]
    with pytest.raises(ValueError, match="duplicate independent ratings"):
        severity_decisions(taxonomy, POLICY)


@pytest.mark.parametrize("version", [True, 2.0, "2"])
def test_decision_does_not_infer_a_confirmed_version_from_other_types(version):
    taxonomy = assigned_cases({"C1": [(0.2, 2), (0.4, 2)]})
    taxonomy[0]["severity_ratings"][0]["scale_version"] = version
    with pytest.raises(ValueError, match="integers or null"):
        severity_decisions(taxonomy, POLICY)


def test_cdf_collapses_ties_and_uses_separate_case_and_triplet_weights():
    taxonomy = assigned_cases(
        {
            "C1": [(0.2, 2), (0.2, 2)],
            "C2": [(0.2, 2), (0.2, 2)],
            "C3": [(0.8, 2), (0.8, 2)],
            "pending": [(0.4, 2)],
        }
    )
    rows = triplets({"C1": 1, "C2": 2, "C3": 3, "pending": 4})
    result = severity_distribution(taxonomy, rows)
    summary, cdf = result["summary"], result["cdf"]
    assert summary == {
        "taxonomy_cases": 4,
        "assigned_cases": 3,
        "pending_cases": 1,
        "triplets": 10,
        "assigned_triplets": 6,
        "pending_triplets": 4,
        "case_mean": 40,
        "case_median": 20,
        "case_min": 20,
        "case_max": 80,
        "triplet_weighted_mean": 50,
    }
    assert [row["severity_score"] for row in cdf] == [20, 80]
    assert cdf[0] == {
        "severity_score": 20,
        "case_count": 2,
        "triplet_count": 3,
        "cumulative_cases": 2,
        "cumulative_triplets": 3,
        "case_fraction": 2 / 3,
        "triplet_fraction": 0.5,
    }
    assert cdf[-1]["case_fraction"] == cdf[-1]["triplet_fraction"] == 1
    assert [row["case_fraction"] for row in cdf] == sorted(row["case_fraction"] for row in cdf)
    assert [row["triplet_fraction"] for row in cdf] == sorted(
        row["triplet_fraction"] for row in cdf
    )
    assert summary["triplet_weighted_mean"] == fmean([20, 20, 20, 80, 80, 80])


def test_cases_without_triplets_still_have_a_case_distribution():
    taxonomy = assigned_cases({"C1": [(0, 2), (0, 2)], "C2": [(1, 2), (1, 2)]})
    result = severity_distribution(taxonomy, [])
    assert result["summary"]["case_mean"] == 50
    assert result["summary"]["assigned_triplets"] == 0
    assert result["summary"]["triplet_weighted_mean"] is None
    assert all(row["triplet_fraction"] is None for row in result["cdf"])
    assert result["cdf"][-1]["case_fraction"] == 1


@pytest.mark.parametrize("with_pending", [False, True])
def test_empty_assignment_has_no_cdf_or_zero_filled_severity(with_pending):
    taxonomy = assigned_cases({"pending": [(0.4, 2)]}) if with_pending else []
    rows = triplets({"pending": 2}) if with_pending else []
    result = severity_distribution(taxonomy, rows)
    assert result["cdf"] == []
    assert result["summary"]["assigned_cases"] == result["summary"]["assigned_triplets"] == 0
    for key in ("case_mean", "case_median", "case_min", "case_max", "triplet_weighted_mean"):
        assert result["summary"][key] is None


@pytest.mark.parametrize("score", [-1, 101, True, float("nan"), float("inf")])
def test_distribution_rejects_invalid_scores(score):
    taxonomy = assigned_cases({"C1": [(0.2, 2), (0.4, 2)]})
    taxonomy[0]["severity_score"] = score
    with pytest.raises(ValueError, match="finite numbers"):
        severity_distribution(taxonomy, [])


def test_distribution_rejects_duplicate_cases_and_pooled_language_ids():
    taxonomy = assigned_cases({"C1": [(0.2, 2), (0.4, 2)]})
    with pytest.raises(ValueError, match="Duplicate public_case_id"):
        severity_distribution(taxonomy + taxonomy, [])
    rows = triplets({"C1": 1})
    with pytest.raises(ValueError, match="Duplicate id"):
        severity_distribution(taxonomy, rows + rows)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"case_id": "outside"}, "selected taxonomy cases"),
        ({"language": "en"}, "aligned English is not a second sample"),
        ({"actor": "veteran"}, "actor differs"),
        ({"id": ""}, "nonempty triplet IDs"),
    ],
)
def test_distribution_rejects_unknown_or_incompatible_triplet_scope(change, message):
    taxonomy = assigned_cases({"C1": [(0.2, 2), (0.4, 2)]})
    rows = triplets({"C1": 1})
    rows[0].update(change)
    with pytest.raises(ValueError, match=message):
        severity_distribution(taxonomy, rows)
