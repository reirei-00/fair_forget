"""Portable enumeration preserves frozen identities and profile semantics."""

from copy import deepcopy

import pytest

from fairforget.warbias.coverage import prepare_generation_cases
from fairforget.warbias.coverage_design import enumerate_items


@pytest.fixture
def design():
    case = {
        "case_id": "ssuk_vet_legitimacy_case_002",
        "target_group": "veteran",
        "intrinsic_attributes": {"service_period": "ato_oos"},
        "reporters": ["hiring manager", "recruiter"],
        "source_claim_uk": "[TARGET] мають менш значущий військовий досвід.",
        "context": "громада",
        "category_code": "experience",
        "category_uk": "Досвід",
        "counter_intent": "Assess individual experience.",
        "unrelated_boundary": "Use a separate everyday event.",
        "forbidden_counters": ["blanket praise"],
        "evidence_ids": "source",
    }
    applicability = {
        "case_id": case["case_id"],
        "target_group": case["target_group"],
        "intrinsic_attributes": case["intrinsic_attributes"],
        "allowed_axes": ["age", "occupation"],
        "target_subgroups": ["veteran_generic"],
        "occupation_reference": "current",
        "review_required_axes": {},
    }
    axes = {"age": ["young", "older"], "occupation": ["it", "retail"]}
    allocation = {
        "matrix_item_id": "WB-0001",
        "case_id": case["case_id"],
        "intrinsic_attributes": case["intrinsic_attributes"],
        "additional_tags": {"age": "older", "gender": "women"},
        "extension_kind": "gender_age",
        "reporter": "recruiter",
    }
    return [case], [applicability], axes, [allocation]


def test_one_axis_inventory_adds_frozen_panel_and_matches_original_stable_identifier(design):
    before = deepcopy(design)
    items = enumerate_items(*design, existing_ids=["WB-0001"])
    assert design == before
    assert len(items) == 18
    baseline = next(row for row in items if row["kind"] == "baseline" and row["reporter"] is None)
    assert baseline["draft_item_id"] == "WB-SYN-019b52dc39310e34"
    assert baseline["generalized_attributes"] == {"service_phase": "earlier_conflict_phase"}
    panel = [row for row in items if row["kind"] == "recorded_gender_age_panel"]
    assert len(panel) == 3
    cohort = [row for row in items if row["cohort_item_id"]]
    assert len(cohort) == 1
    assert cohort[0]["cohort_item_id"] == "WB-0001"
    assert cohort[0]["reporter"] == "recruiter"
    assert cohort[0]["attributes"]["gender"] == "women"
    assert cohort[0]["source_status"] == "existing_annotation_cohort"
    assert all(row["human_validated"] is False for row in items)
    rows = prepare_generation_cases(
        items,
        design[0],
        [{"evidence_id": "source", "source_type": "feedback"}],
        existing_ids=["WB-0001"],
    )
    assert len(rows) == 17
    assert all(row["id"] != cohort[0]["draft_item_id"] for row in rows)


def test_frozen_occupation_alias_and_temporal_reference_remain_distinct(design):
    case = design[0][0]
    design[3].append(
        {
            "matrix_item_id": "WB-0002",
            "case_id": case["case_id"],
            "intrinsic_attributes": case["intrinsic_attributes"],
            "additional_tags": {"occupation": "family_it"},
            "occupation_reference": "previous",
            "reporter": None,
        }
    )
    items = enumerate_items(*design, existing_ids=["WB-0002"])
    assert len(items) == 21
    current = [
        row
        for row in items
        if row["attributes"].get("occupation") == "it" and row["occupation_reference"] == "current"
    ]
    previous = [
        row
        for row in items
        if row["attributes"].get("occupation") == "it" and row["occupation_reference"] == "previous"
    ]
    assert len(current) == len(previous) == 3
    assert {row["draft_item_id"] for row in current}.isdisjoint(
        row["draft_item_id"] for row in previous
    )


def test_derived_displacement_phase_is_not_an_independent_dimension(design):
    design[1][0]["allowed_axes"] = ["displacement_wave"]
    design[2].clear()
    design[2]["displacement_wave"] = ["before_full_scale", "during_full_scale"]
    items = enumerate_items(*design)
    assert len(items) == 12
    later = next(
        row for row in items if row["attributes"].get("displacement_wave") == "during_full_scale"
    )
    assert later["generalized_attributes"] == {
        "service_phase": "earlier_conflict_phase",
        "displacement_phase": "later_conflict_phase",
    }


@pytest.mark.parametrize("existing", [["unknown"], ["WB-0001", "WB-0001"]])
def test_existing_ids_must_bind_to_distinct_frozen_allocations(design, existing):
    with pytest.raises(ValueError, match="Existing IDs"):
        enumerate_items(*design, existing_ids=existing)


def test_applicability_and_frozen_allocations_cannot_change_intrinsics(design):
    design[1][0]["intrinsic_attributes"] = {"service_period": "full_scale"}
    with pytest.raises(ValueError, match="intrinsic"):
        enumerate_items(*design)
    design[1][0]["intrinsic_attributes"] = design[0][0]["intrinsic_attributes"]
    design[3][0]["additional_tags"]["service_period"] = "full_scale"
    with pytest.raises(ValueError, match="intrinsic"):
        enumerate_items(*design)


def test_unresolved_axes_and_conflicting_phase_aliases_are_rejected(design):
    design[1][0]["review_required_axes"] = {"age": "Pending eligibility decision."}
    with pytest.raises(ValueError, match="unresolved"):
        enumerate_items(*design)
    design[1][0]["review_required_axes"] = {}
    design[3][0]["generalized_attributes"] = {"service_phase": "later_conflict_phase"}
    with pytest.raises(ValueError, match="phase aliases"):
        enumerate_items(*design)
