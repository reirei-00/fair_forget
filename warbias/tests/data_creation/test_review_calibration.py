from copy import deepcopy

import pytest

from fairforget.warbias.review_calibration import family_split, partition_rows, select_review


def test_split_keeps_variants_together_and_is_independent_of_item_order():
    rows = [
        {"id": f"{group}-{case}-{variant}", "case_family": f"{group}-{case}", "target_group": group}
        for group in ("idp", "veteran")
        for case in range(10)
        for variant in range(3)
    ]
    split = family_split(rows)
    assert family_split(rows[::-1]) == split
    development = partition_rows(rows, split, "development")
    test = partition_rows(rows, split, "test")
    assert len(development) == 48
    assert len(test) == 12
    assert not {row["case_family"] for row in test} & {row["case_family"] for row in development}
    assert {row["id"] for row in development + test} == {row["id"] for row in rows}


def test_split_rejects_unknown_or_changed_families():
    rows = [
        {"id": "a", "case_family": "a", "target_group": "idp"},
        {"id": "b", "case_family": "b", "target_group": "idp"},
    ]
    split = family_split(rows)
    changed = deepcopy(rows)
    changed[0]["target_group"] = "veteran"
    with pytest.raises(ValueError, match="frozen"):
        partition_rows(changed, split, "test")
    with pytest.raises(ValueError, match="one target"):
        family_split(rows + [{"id": "c", "case_family": "a", "target_group": "veteran"}])
    with pytest.raises(ValueError, match="fraction"):
        family_split(rows, test_fraction=1)


def test_partition_accepts_request_context_without_changing_it():
    rows = [
        {"id": "a", "case_family": "a", "target_group": "idp"},
        {"id": "b", "case_family": "b", "target_group": "idp"},
    ]
    split = family_split(rows)
    requests = [
        {"id": row["id"], "case_family": row["case_family"], "context": {"target_group": "idp"}}
        for row in rows
    ]
    selected = partition_rows(requests, split, "test")
    assert len(selected) == 1
    assert "target_group" not in selected[0]


def test_higher_ac1_cannot_hide_worse_problem_agreement():
    development = {"mean_pair_ac1_change": {"value": 0.2}}
    test = {
        "mean_pair_ac1_change": {"ci95": {"low": 0.1}},
        "baseline": {
            "binary_pairs": {"positive_agreement": 0.3},
            "conditional_jaccard": {"value": 0.2},
        },
        "candidate": {
            "binary_pairs": {"positive_agreement": 0},
            "conditional_jaccard": {"value": 0.2},
        },
    }
    result = select_review(development, test)
    assert result["selected"] == "baseline"
    assert result["checks"]["test_positive_agreement_does_not_decrease"] is False
    test["candidate"]["binary_pairs"]["positive_agreement"] = 0.4
    assert select_review(development, test)["selected"] == "candidate"
    test["mean_pair_ac1_change"]["ci95"]["low"] = -0.01
    assert select_review(development, test)["selected"] == "baseline"
