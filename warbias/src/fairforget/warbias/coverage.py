"""Prepare missing case, profile and reporter combinations for generation."""

import json
from copy import deepcopy

from fairforget.io import canonical_json, digest, unique_index
from fairforget.warbias.evidence import bind_evidence
from fairforget.warbias.profile_wording import render_reporter, render_target
from fairforget.warbias.profiles import SUPPORTED_ATTRIBUTES

TARGET_CODES = {"idp": "idp_current", "veteran": "veteran_generic"}
OCCUPATION_REFERENCES = {"current", "previous", "sought"}


def _text(row, field):
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Coverage inputs need a nonempty {field}")
    return value


def _named_values(value):
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict) or any(
        not isinstance(key, str) or not key.strip() or not isinstance(item, str) or not item.strip()
        for key, item in value.items()
    ):
        raise ValueError("Coverage attributes must use names and nonempty string values")
    return dict(value)


def _attributes(value):
    result = _named_values(value)
    if not set(result) <= SUPPORTED_ATTRIBUTES:
        raise ValueError("Coverage attributes must use supported names")
    return result


def normalize_attributes(value):
    """Validate named profile values and resolve recorded age and sector aliases."""
    result = _attributes(value)
    if result.get("age") == "young_adult":
        result["age"] = "young"
    occupation = result.get("occupation", "")
    if occupation.startswith("family_"):
        result["occupation"] = occupation.removeprefix("family_")
    return result


def _case_index(cases):
    cases = list(cases)
    for case in cases:
        for field in (
            "case_id",
            "source_claim_uk",
            "context",
            "category_code",
            "category_uk",
            "counter_intent",
            "unrelated_boundary",
            "evidence_ids",
        ):
            _text(case, field)
        if case.get("target_group") not in (*TARGET_CODES, "displaced_abroad"):
            raise ValueError("Unknown coverage target group")
        _attributes(case.get("intrinsic_attributes"))
        reporters = case.get("reporters")
        if (
            not isinstance(reporters, list)
            or len(reporters) != 2
            or any(not isinstance(value, str) or not value.strip() for value in reporters)
            or len(set(reporters)) != 2
        ):
            raise ValueError("Each coverage case needs two distinct allowed reporters")
        counters = case.get("forbidden_counters")
        if not isinstance(counters, list) or any(
            not isinstance(value, str) or not value.strip() for value in counters
        ):
            raise ValueError("Each coverage case needs explicit forbidden_counters")
    return unique_index(cases, "case_id")


def _subgroup(case, target_code):
    allowed = case.get("target_subgroups")
    if allowed is None and case["target_group"] in TARGET_CODES:
        allowed = [TARGET_CODES[case["target_group"]]]
    if not isinstance(allowed, list) or target_code not in allowed:
        raise ValueError("The target subgroup must be explicitly allowed by its case")
    definitions = case.get("target_subgroup_definitions_uk", {})
    definition = definitions.get(target_code)
    if case["target_group"] == "displaced_abroad" and (
        not isinstance(definition, str) or not definition.strip()
    ):
        raise ValueError("Abroad subgroups need explicit Ukrainian definitions")
    return definition


def prepare_generation_cases(items, cases, sources, *, existing_ids=()):
    """Validate a full coverage manifest and omit its recorded annotation cohort."""
    cases = _case_index(cases)
    existing_ids = list(existing_ids)
    if any(not isinstance(item, str) or not item.strip() for item in existing_ids) or len(
        set(existing_ids)
    ) != len(existing_ids):
        raise ValueError("Existing annotation IDs must be distinct nonempty strings")
    existing = set(existing_ids)
    seen_existing, identifiers, combinations = set(), set(), set()
    output = []
    for item in items:
        identity = _text(item, "draft_item_id")
        if identity in identifiers:
            raise ValueError("Duplicate coverage item ID")
        identifiers.add(identity)
        case = cases.get(item.get("case_id"))
        if case is None:
            raise ValueError("Every coverage item must resolve to a semantic case")
        if item.get("target_group") != case["target_group"]:
            raise ValueError("Coverage target group differs from its semantic case")
        subgroup = _text(item, "target_code")
        subgroup_uk = _subgroup(case, subgroup)
        attributes = normalize_attributes(item.get("attributes"))
        intrinsic = _attributes(case["intrinsic_attributes"])
        generalized = _named_values(
            item.get("generalized_attributes", case.get("generalized_attributes", {}))
        )
        if any(
            attributes.get(key) != value for key, value in normalize_attributes(intrinsic).items()
        ):
            raise ValueError("Coverage profiles must preserve every intrinsic case attribute")
        reporter = item.get("reporter") or None
        if reporter is not None and reporter not in case["reporters"]:
            raise ValueError("Coverage reporter must be allowed by its semantic case")
        reference = item.get("occupation_reference") or None
        if "occupation" in attributes:
            if reference not in OCCUPATION_REFERENCES:
                raise ValueError("Occupation profiles need an explicit temporal reference")
        elif reference is not None:
            raise ValueError("An occupation reference needs an occupation attribute")
        profile = (case["case_id"], subgroup, canonical_json(attributes), reference)
        combination = (*profile, reporter)
        if combination in combinations:
            raise ValueError("Duplicate logical case, profile and reporter combination")
        combinations.add(combination)
        cohort_id = item.get("cohort_item_id") or None
        if cohort_id is not None:
            if cohort_id not in existing or cohort_id in seen_existing:
                raise ValueError("The manifest must identify distinct existing annotation IDs")
            seen_existing.add(cohort_id)
            continue
        if item.get("human_validated") in (True, "true", "True"):
            raise ValueError("New coverage items cannot be marked human validated")
        row = {
            **deepcopy(case),
            "id": identity,
            "matrix_id": identity,
            "matrix_item_id": identity,
            "profile_id": "WB-PROFILE-" + digest(profile)[:16],
            "target_code": subgroup,
            "bias_type_code": case["target_group"],
            "intrinsic_attributes": intrinsic,
            "additional_tags": {
                key: value for key, value in attributes.items() if key not in intrinsic
            },
            "generalized_attributes": generalized,
            "occupation_reference": reference,
            "reporter": reporter,
            "reporter_uk": render_reporter(reporter),
            "framing": "unattributed" if reporter is None else "attributed",
            "gender_track": {"women": "feminine", "men": "masculine"}.get(
                attributes.get("gender"), "unspecified"
            ),
            "age_track": {
                "young": "young_adult",
                "middle_aged": "middle_age",
                "older": "older_adult",
            }.get(attributes.get("age"), "unspecified"),
            "claim": case["source_claim_uk"],
            "context_uk": case["context"],
            "required_stereotype_claim_uk": case["source_claim_uk"],
            "required_context_domain_uk": case["context"],
            "matrix_schema": "case_profile",
            "row_origin": "coverage_extension",
            "generation_status": "not_generated",
            "human_annotation_status": "not_annotated",
            "human_validated": False,
        }
        if subgroup_uk is not None:
            row["target_subgroup_uk"] = subgroup_uk
        row["target_uk"] = render_target(row, attributes, reference, subgroup_uk)
        output.append(row)
    if seen_existing != existing:
        raise ValueError("The full coverage manifest must preserve every existing annotation ID")
    return bind_evidence(output, sources)
