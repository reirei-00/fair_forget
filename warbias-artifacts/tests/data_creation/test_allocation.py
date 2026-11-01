import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

from warbias_kit.io import read_rows
from warbias_kit.warbias.allocation import (
    reproduce_base_allocation,
    validate_recorded_allocation,
)


@pytest.fixture
def small_design():
    cases = [
        {
            "case_id": name,
            "intrinsic_attributes": {"state": "fixed"},
            "reporters": ["peer", "manager"],
        }
        for name in ("case_a", "case_b")
    ]
    profiles = []
    for case in cases:
        attributes = [({}, "base")]
        for dimension, values in [("age", ["young", "older"]), ("gender", ["women", "men"])]:
            attributes.extend(({dimension: value}, f"{dimension}-{value}") for value in values)
        profiles.extend(
            {
                "case_id": case["case_id"],
                "profile_id": f"{case['case_id']}.{suffix}",
                "intrinsic_attributes": case["intrinsic_attributes"],
                "additional_tags": added,
                "generalized_attributes": {},
            }
            for added, suffix in attributes
        )
    design = {
        "base_selection": {
            "budget": 8,
            "seed": 17,
            "composition": {
                "baseline_unattributed": 2,
                "baseline_attributed": 2,
                "single_attribute_unattributed": 4,
            },
        }
    }
    return cases, profiles, design, {}


def test_budget_selects_complete_balanced_blocks_and_retains_context(small_design):
    cases, profiles, design, substitutions = small_design
    original = deepcopy(small_design)
    rows = reproduce_base_allocation(*small_design)
    assert len(rows) == 8
    assert [row["matrix_item_id"] for row in rows] == [f"WB-{i:04d}" for i in range(1, 9)]
    assert Counter(row["case_id"] for row in rows) == {"case_a": 4, "case_b": 4}
    assert Counter(key for row in rows for key in row["additional_tags"]) == {"age": 2, "gender": 2}
    for case in cases:
        controls = [
            row for row in rows if row["case_id"] == case["case_id"] and not row["additional_tags"]
        ]
        assert len(controls) == 2
        assert sum(row["reporter"] is None for row in controls) == 1
    assert all(row["intrinsic_attributes"] == {"state": "fixed"} for row in rows)
    assert reproduce_base_allocation(cases[::-1], profiles[::-1], design, substitutions) == rows
    assert small_design == original
    assert validate_recorded_allocation(cases, profiles, rows, design, substitutions) == {
        "base": 8,
        "profile_extension": 0,
        "total": 8,
    }


def test_explicit_substitutions_replace_whole_blocks_before_assigning_ids(small_design):
    cases, profiles, design, _ = small_design
    first = reproduce_base_allocation(*small_design)
    old = [row for row in first if row["case_id"] == "case_a" and row["additional_tags"]]
    dimension = next(iter(old[0]["additional_tags"]))
    new = [
        p
        for p in profiles
        if p["case_id"] == "case_a"
        and p["additional_tags"]
        and dimension not in p["additional_tags"]
    ]
    substitutions = {
        "seed": 17,
        "complete_alternative_value_blocks": True,
        "removed_ids": [row["design_unit_id"] for row in old],
        "added_ids": [row["profile_id"] + "-f0" for row in new],
    }
    result = reproduce_base_allocation(cases, profiles, design, substitutions)
    ids = {row["design_unit_id"] for row in result}
    assert not ids.intersection(substitutions["removed_ids"])
    assert set(substitutions["added_ids"]) <= ids
    assert len(result) == 8
    with pytest.raises(ValueError, match="complete"):
        reproduce_base_allocation(
            cases,
            profiles,
            design,
            {
                **substitutions,
                "removed_ids": substitutions["removed_ids"][:1],
                "added_ids": substitutions["added_ids"][:1],
            },
        )


@pytest.mark.parametrize(
    "failure",
    [
        "missing_case",
        "duplicate_profile",
        "duplicate_logical",
        "intrinsic",
        "missing_baseline",
        "budget",
        "replacement",
    ],
)
def test_invalid_inputs_fail_instead_of_changing_the_sample(small_design, failure):
    cases, profiles, design, substitutions = deepcopy(small_design)
    if failure == "missing_case":
        profiles[0]["case_id"] = "absent"
    elif failure == "duplicate_profile":
        profiles.append(deepcopy(profiles[0]))
    elif failure == "duplicate_logical":
        profiles.append({**profiles[1], "profile_id": "different-id-same-profile"})
    elif failure == "intrinsic":
        profiles[1]["intrinsic_attributes"] = {"state": "changed"}
    elif failure == "missing_baseline":
        profiles = [row for row in profiles if row["profile_id"] != "case_a.base"]
    elif failure == "budget":
        design["base_selection"]["budget"] = 7
    else:
        substitutions = {"removed_ids": ["absent"], "added_ids": ["absent-again"]}
    with pytest.raises(ValueError):
        reproduce_base_allocation(cases, profiles, design, substitutions)


def _gender_panel(small_design):
    cases, profiles, design, substitutions = deepcopy(small_design)
    base = reproduce_base_allocation(cases, profiles, design, substitutions)
    parent = next(
        row
        for row in base
        if row["case_id"] == "case_a" and not row["additional_tags"] and row["reporter"] is None
    )
    extensions = [
        {
            **parent,
            "matrix_item_id": f"WB-{index:04d}",
            "parent_matrix_item_id": parent["matrix_item_id"],
            "design_stage": "profile_extension",
            "extension_kind": "gender",
            "additional_tags": {"gender": gender},
        }
        for index, gender in enumerate(["women", "men"], 9)
    ]
    design["extensions"] = {
        "total": 2,
        "groups": [
            {
                "kind": "gender",
                "case_count": 1,
                "case_ids": ["case_a"],
                "variants_per_case": 2,
                "variant_count": 2,
            }
        ],
    }
    return cases, profiles, base + extensions, design, substitutions


def test_extension_validation_requires_explicit_complete_panels(small_design):
    arguments = _gender_panel(small_design)
    assert validate_recorded_allocation(*arguments)["total"] == 10
    for failure in ("duplicate_variant", "wrong_parent", "stale_base_id"):
        cases, profiles, allocations, design, substitutions = deepcopy(arguments)
        if failure == "duplicate_variant":
            allocations[-1]["additional_tags"] = allocations[-2]["additional_tags"]
        elif failure == "wrong_parent":
            allocations[-1]["parent_matrix_item_id"] = "absent"
        else:
            allocations[0]["matrix_item_id"], allocations[1]["matrix_item_id"] = (
                allocations[1]["matrix_item_id"],
                allocations[0]["matrix_item_id"],
            )
        with pytest.raises(ValueError):
            validate_recorded_allocation(cases, profiles, allocations, design, substitutions)


def test_recorded_1000_base_and_200_extension_panels_reproduce_exactly():
    root = Path(__file__).resolve().parents[2]
    inputs = root / "protocol/inputs"
    cases = read_rows(inputs / "construction_cases.jsonl")
    profiles = read_rows(inputs / "candidate_profiles.jsonl")
    allocations = read_rows(inputs / "profile_design.jsonl")
    design = json.loads((root / "protocol/allocation_design.json").read_text())
    substitutions = json.loads((inputs / "profile_replacements.json").read_text())
    references = {
        row["case_id"]: row["occupation_reference"]
        for row in allocations
        if row["design_stage"] == "base" and "occupation_reference" in row
    }
    assert len(references) == 4
    design["base_selection"]["occupation_reference_by_case"] = references
    reproduced = reproduce_base_allocation(cases, profiles, design, substitutions)
    expected = sorted(
        [row for row in allocations if row["design_stage"] == "base"],
        key=lambda row: row["matrix_item_id"],
    )
    assert reproduced == expected
    assert validate_recorded_allocation(cases, profiles, allocations, design, substitutions) == {
        "base": 1000,
        "profile_extension": 200,
        "total": 1200,
    }
