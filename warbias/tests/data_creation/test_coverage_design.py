"""Coverage selection keeps historical profiles and matched subgroup panels."""

import json
from copy import deepcopy

import pytest

from fairforget.warbias.coverage_design import occupation_counts, select_items


def case(identity="case", **changes):
    return {
        "case_id": identity,
        "target_group": "veteran",
        "category_code": "employment",
        "intrinsic_attributes": {"gender": "women"},
        "reporters": ["manager", "recruiter"],
        **changes,
    }


def profile(source, attributes, *, subgroup="veteran_generic", cohort=False, reference=None):
    attributes = {**source["intrinsic_attributes"], **attributes}
    return [
        {
            "draft_item_id": f"{source['case_id']}-{subgroup}-{attributes}-{reporter}",
            "case_id": source["case_id"],
            "target_group": source["target_group"],
            "target_code": subgroup,
            "attributes": json.dumps(attributes),
            "occupation_reference": reference,
            "reporter": reporter,
            "cohort_item_id": "WB-0001" if cohort and reporter is None else "",
        }
        for reporter in [None, *source["reporters"]]
    ]


def allocation(source, attributes, **changes):
    return {
        "case_id": source["case_id"],
        "intrinsic_attributes": source["intrinsic_attributes"],
        "additional_tags": attributes,
        **changes,
    }


def test_sector_selection_is_stable_preserves_alias_allocations_and_does_not_mutate_inputs():
    source = case()
    sectors = ["it", "healthcare", "education", "transport", "trades", "retail"]
    items = sum(
        [profile(source, {"occupation": value}, reference="current") for value in sectors], []
    )
    items += profile(source, {"age": "young"})
    preserved = [
        allocation(source, {"occupation": "family_it"}),
        allocation(source, {"age": "young_adult"}),
    ]
    before = deepcopy(items)
    selected = select_items(items, [source], preserved, occupation_limit=2, seed=17)
    assert items == before
    assert len(selected) == 9
    assert occupation_counts(selected)["veteran"]["it"] == 1
    assert {row["draft_item_id"] for row in selected} == {
        row["draft_item_id"]
        for row in select_items(reversed(items), [source], preserved, occupation_limit=2, seed=17)
    }


def test_all_preserved_sector_profiles_override_a_smaller_limit_and_retain_temporal_reference():
    source = case()
    sectors = ["it", "healthcare", "education", "transport"]
    items = sum(
        [profile(source, {"occupation": value}, reference="previous") for value in sectors], []
    )
    preserved = [
        allocation(source, {"occupation": value}, occupation_reference="previous")
        for value in sectors
    ]
    assert select_items(items, [source], preserved, occupation_limit=2) == items
    with pytest.raises(ValueError, match="every preserved allocation"):
        select_items(items, [source], [allocation(source, {"occupation": "it"})])


def test_abroad_subgroups_receive_the_same_sectors_and_counts_report_actual_coverage():
    source = case(target_group="displaced_abroad")
    items = []
    for subgroup in ["refugee", "temporary_protection"]:
        for sector in ["it", "healthcare", "education", "transport", "retail"]:
            items += profile(source, {"occupation": sector}, subgroup=subgroup, reference="current")
    selected = select_items(items, [source], [], occupation_limit=3)
    values = {
        subgroup: {
            json.loads(row["attributes"])["occupation"]
            for row in selected
            if row["target_code"] == subgroup
        }
        for subgroup in ["refugee", "temporary_protection"]
    }
    assert values["refugee"] == values["temporary_protection"]
    assert len(selected) == 18
    assert set(occupation_counts(selected)["displaced_abroad"].values()) == {2}


def test_deferred_dimensions_keep_intrinsic_attributes_frozen_profiles_and_cohort_profiles():
    source = case(intrinsic_attributes={"rank": "officer"})
    items = profile(source, {}) + profile(source, {"service_duration": "short_service"})
    items += profile(source, {"service_duration": "long_service"}, cohort=True)
    items += profile(source, {"age": "older"})
    selected = select_items(items, [source], [], deferred_dimensions=["rank", "service_duration"])
    assert len(selected) == 9
    assert all(json.loads(row["attributes"])["rank"] == "officer" for row in selected)
    assert all(
        json.loads(row["attributes"]).get("service_duration") != "short_service" for row in selected
    )
    frozen = [allocation(source, {"service_duration": "short_service"})]
    assert select_items(items, [source], frozen, deferred_dimensions=["service_duration"]) == items


def test_sector_balance_spreads_values_across_cases_without_claiming_complete_small_panels():
    sources = [case(str(index)) for index in range(6)]
    items = [
        row
        for source in sources
        for sector in ["it", "healthcare", "education", "transport", "retail", "trades"]
        for row in profile(source, {"occupation": sector}, reference="current")
    ]
    selected = select_items(items, sources, [], occupation_limit=2)
    assert set(occupation_counts(selected)["veteran"].values()) == {2}


@pytest.mark.parametrize(
    "changes",
    [
        {"occupation_limit": 0},
        {"occupation_limit": True},
        {"seed": True},
        {"deferred_dimensions": ["made_up"]},
    ],
)
def test_invalid_selection_settings_are_rejected(changes):
    source = case()
    with pytest.raises(ValueError):
        select_items(profile(source, {}), [source], [], **changes)


def test_incomplete_framings_and_missing_subgroup_sectors_are_rejected():
    source = case()
    with pytest.raises(ValueError, match="all three"):
        select_items(profile(source, {})[:2], [source], [])
    items = profile(source, {"occupation": "it"}, subgroup="refugee", reference="current")
    items += profile(
        source, {"occupation": "it"}, subgroup="temporary_protection", reference="current"
    )
    items += profile(source, {"occupation": "healthcare"}, subgroup="refugee", reference="current")
    with pytest.raises(ValueError, match="same target subgroups"):
        select_items(items, [source], [])
