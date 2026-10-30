"""Enumerate source-bound coverage profiles before selecting a generation subset."""

import hashlib
import json

from warbias_kit.io import unique_index
from warbias_kit.warbias.coverage import (
    OCCUPATION_REFERENCES,
    TARGET_CODES,
    normalize_attributes,
)

PHASES = {
    "service_period": (
        "service_phase",
        {
            "ato_oos": "earlier_conflict_phase",
            "full_scale": "later_conflict_phase",
            "both": "both_conflict_phases",
        },
    ),
    "displacement_wave": (
        "displacement_phase",
        {
            "before_full_scale": "earlier_conflict_phase",
            "during_full_scale": "later_conflict_phase",
        },
    ),
}


def _generalized(attributes):
    result = {}
    for axis, (name, values) in PHASES.items():
        if axis in attributes:
            if attributes[axis] not in values:
                raise ValueError("A temporal profile attribute needs a recorded phase value")
            result[name] = values[attributes[axis]]
    return result


def _inputs(cases, applicability, axis_values):
    cases = unique_index(cases, "case_id")
    rules = unique_index(applicability, "case_id")
    if not cases or cases.keys() != rules.keys():
        raise ValueError("Coverage cases and applicability must identify the same nonempty set")
    if not isinstance(axis_values, dict) or not axis_values:
        raise ValueError("Coverage enumeration needs explicit dimension values")
    for axis, values in axis_values.items():
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) or not value.strip() for value in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError("Each dimension needs distinct nonempty string values")
        for value in values:
            normalize_attributes({axis: value})
    for identity, rule in rules.items():
        case = cases[identity]
        if rule.get("target_group") != case.get("target_group"):
            raise ValueError("Applicability must preserve the case target group")
        fixed = normalize_attributes(rule.get("intrinsic_attributes"))
        if fixed != normalize_attributes(case.get("intrinsic_attributes")):
            raise ValueError("Applicability must preserve intrinsic case attributes")
        axes, subgroups = rule.get("allowed_axes"), rule.get("target_subgroups")
        if (
            not isinstance(axes, list)
            or len(set(axes)) != len(axes)
            or set(axes) - axis_values.keys()
            or set(axes) & fixed.keys()
        ):
            raise ValueError(
                "Allowed axes need declared values and cannot replace intrinsic attributes"
            )
        if set(axes) & rule.get("review_required_axes", {}).keys():
            raise ValueError("An unresolved dimension cannot be newly enumerated")
        allowed = case.get("target_subgroups") or [TARGET_CODES.get(case["target_group"])]
        if (
            not isinstance(subgroups, list)
            or not subgroups
            or any(not isinstance(value, str) or not value.strip() for value in subgroups)
            or len(set(subgroups)) != len(subgroups)
            or set(subgroups) != set(allowed)
        ):
            raise ValueError("Applicability must identify the case's allowed target subgroups")
        reporters = case.get("reporters")
        if (
            not isinstance(reporters, list)
            or len(reporters) != 2
            or any(not isinstance(value, str) or not value.strip() for value in reporters)
            or len(set(reporters)) != 2
        ):
            raise ValueError("Each coverage case needs two distinct allowed reporters")
    return cases, rules


def enumerate_items(cases, applicability, axis_values, preserved_allocations, *, existing_ids=()):
    """Build baselines, single-axis variants and every frozen profile, with three framings."""
    cases, rules = _inputs(cases, applicability, axis_values)
    allocations = unique_index(preserved_allocations, "matrix_item_id")
    existing_ids = list(existing_ids)
    if (
        any(not isinstance(value, str) or not value.strip() for value in existing_ids)
        or len(set(existing_ids)) != len(existing_ids)
        or set(existing_ids) - allocations.keys()
    ):
        raise ValueError("Existing IDs must identify distinct preserved allocation records")
    profiles = {}

    def add(identity, subgroup, attributes, reference, kind):
        attributes = normalize_attributes(attributes)
        rule = rules[identity]
        fixed = normalize_attributes(rule["intrinsic_attributes"])
        if subgroup not in rule["target_subgroups"] or any(
            attributes.get(axis) != value for axis, value in fixed.items()
        ):
            raise ValueError(
                "A frozen profile cannot change its case's intrinsic attributes or subgroup"
            )
        reference = reference if "occupation" in attributes else None
        if "occupation" in attributes and reference not in OCCUPATION_REFERENCES:
            raise ValueError("An occupation profile needs an explicit temporal reference")
        key = identity, subgroup, json.dumps(attributes, sort_keys=True), reference
        profiles.setdefault(
            key,
            {
                "case_id": identity,
                "target_group": rule["target_group"],
                "target_code": subgroup,
                "attributes": attributes,
                "generalized_attributes": _generalized(attributes),
                "occupation_reference": reference,
                "kind": kind,
            },
        )
        return key

    for identity, rule in rules.items():
        fixed = normalize_attributes(rule["intrinsic_attributes"])
        for subgroup in rule["target_subgroups"]:
            add(identity, subgroup, fixed, rule.get("occupation_reference"), "baseline")
            for axis in rule["allowed_axes"]:
                for value in axis_values[axis]:
                    add(
                        identity,
                        subgroup,
                        {**fixed, axis: value},
                        rule.get("occupation_reference"),
                        "single_attribute",
                    )
    allocation_keys = {}
    for identity, row in allocations.items():
        case = cases.get(row.get("case_id"))
        if case is None:
            raise ValueError("Every preserved allocation must resolve to a coverage case")
        fixed = normalize_attributes(row["intrinsic_attributes"])
        if fixed != normalize_attributes(case["intrinsic_attributes"]):
            raise ValueError("Frozen allocations must preserve intrinsic case attributes")
        subgroups = rules[case["case_id"]]["target_subgroups"]
        subgroup = row.get("target_code") or TARGET_CODES.get(case["target_group"])
        if subgroup is None and len(subgroups) == 1:
            subgroup = subgroups[0]
        attributes = {**fixed, **normalize_attributes(row["additional_tags"])}
        expected_generalized = _generalized(attributes)
        if any(
            expected_generalized.get(name) != value
            for name, value in row.get("generalized_attributes", {}).items()
        ):
            raise ValueError("Recorded phase aliases must match their profile attributes")
        kind = (
            "recorded_gender_age_panel"
            if row.get("extension_kind") == "gender_age"
            else "recorded_profile"
        )
        profile = add(
            case["case_id"], subgroup, attributes, row.get("occupation_reference", "current"), kind
        )
        reporter = row.get("reporter") or None
        if reporter is not None and reporter not in case["reporters"]:
            raise ValueError("A frozen allocation needs an allowed reporter")
        logical = *profile, reporter
        if logical in allocation_keys.values():
            raise ValueError("Preserved allocations cannot duplicate a logical item")
        allocation_keys[identity] = logical
    items = {}
    for profile, row in profiles.items():
        for reporter in [None, *cases[row["case_id"]]["reporters"]]:
            logical = *profile, reporter
            identity = hashlib.sha256(json.dumps(logical, ensure_ascii=False).encode()).hexdigest()[
                :16
            ]
            items[logical] = {
                **row,
                "reporter": reporter,
                "draft_item_id": "WB-SYN-" + identity,
                "cohort_item_id": "",
                "source_status": "synthetic_generation_planned",
                "human_validated": False,
            }
    if len({row["draft_item_id"] for row in items.values()}) != len(items):
        raise ValueError("Coverage item identifiers collided")
    for identity in existing_ids:
        row = items[allocation_keys[identity]]
        row["cohort_item_id"] = identity
        row["source_status"] = "existing_annotation_cohort"
    return list(items.values())
