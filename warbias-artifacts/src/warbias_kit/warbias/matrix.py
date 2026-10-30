"""Expand semantic cases through an explicit profile and framing design."""

from collections import Counter

from warbias_kit.io import canonical_json, unique_index
from warbias_kit.warbias.evidence import bind_evidence
from warbias_kit.warbias.taxonomy import apply_categories


def _attributes(record, field):
    value = record.get(field)
    if not isinstance(value, dict):
        raise ValueError(f"A case or profile needs {field}")
    return value


def _effective_attributes(intrinsic, additional):
    if any(key in additional and additional[key] != value for key, value in intrinsic.items()):
        raise ValueError("A profile cannot change an intrinsic case attribute")
    return {**intrinsic, **additional}


def _logical_key(row):
    attributes = _effective_attributes(row["intrinsic_attributes"], row["additional_tags"])
    if attributes.get("age") == "young_adult":
        attributes["age"] = "young"
    occupation = attributes.get("occupation")
    if isinstance(occupation, str) and occupation.startswith("family_"):
        attributes["occupation"] = occupation.removeprefix("family_")
    return (
        row["case_id"],
        canonical_json(attributes),
        row.get("reporter"),
        row.get("occupation_reference"),
        canonical_json(row.get("generalized_attributes", {})),
    )


def _index(rows, field):
    rows = list(rows)
    if any(not isinstance(row.get(field), str) or not row[field].strip() for row in rows):
        raise ValueError(f"Each record needs a nonempty {field}")
    return unique_index(rows, field)


def _inputs(cases, profiles):
    cases, profiles = list(cases), list(profiles)
    if not cases or not profiles:
        raise ValueError("Declare the semantic cases and candidate profiles")
    indexed_cases = _index(cases, "case_id")
    indexed_profiles = _index(profiles, "profile_id")
    for case in cases:
        if case.get("target_group") not in ("idp", "veteran"):
            raise ValueError("A construction case needs its recorded target group")
        for field in ("source_claim_uk", "context", "category_code", "category_uk"):
            if not isinstance(case.get(field), str) or not case[field].strip():
                raise ValueError(f"A construction case needs {field}")
        _attributes(case, "intrinsic_attributes")
        reporters = case.get("reporters")
        if (
            not isinstance(reporters, list)
            or any(not isinstance(value, str) or not value.strip() for value in reporters)
            or len(set(reporters)) != len(reporters)
        ):
            raise ValueError("Declare distinct allowed reporters for each case")
    combinations = set()
    for profile in profiles:
        case = indexed_cases.get(profile.get("case_id"))
        if case is None:
            raise ValueError("Each candidate profile must belong to a construction case")
        intrinsic = _attributes(profile, "intrinsic_attributes")
        if intrinsic != case["intrinsic_attributes"]:
            raise ValueError("Candidate profiles must preserve intrinsic case attributes")
        _effective_attributes(intrinsic, _attributes(profile, "additional_tags"))
        if "generalized_attributes" in profile:
            _attributes(profile, "generalized_attributes")
        key = _logical_key(profile)
        if key in combinations:
            raise ValueError("Duplicate logical candidate profile")
        combinations.add(key)
    return indexed_cases, indexed_profiles


def _matches_profile(allocation, profile):
    return all(
        profile.get(field) == allocation.get(field)
        for field in ("case_id", "intrinsic_attributes", "additional_tags")
    ) and allocation.get(
        "generalized_attributes", profile.get("generalized_attributes", {})
    ) == profile.get("generalized_attributes", {})


def _validate_allocation(allocation, case, profiles, allocations):
    intrinsic = _attributes(allocation, "intrinsic_attributes")
    additional = _attributes(allocation, "additional_tags")
    if intrinsic != case["intrinsic_attributes"]:
        raise ValueError("Design allocations must preserve intrinsic case attributes")
    effective = _effective_attributes(intrinsic, additional)
    if not isinstance(allocation.get("profile_id"), str) or not allocation["profile_id"].strip():
        raise ValueError("Every allocation needs a profile_id")
    if "generalized_attributes" in allocation:
        _attributes(allocation, "generalized_attributes")
    reporter, framing = allocation.get("reporter"), allocation.get("framing")
    if not (
        framing == "unattributed"
        and reporter is None
        or framing == "attributed"
        and reporter in case["reporters"]
    ):
        raise ValueError("The design framing and reporter must match the case")
    if allocation.get("design_stage") == "base":
        profile = profiles.get(allocation.get("profile_id"))
        if profile is None or not _matches_profile(allocation, profile):
            raise ValueError("A base allocation must match its candidate profile")
    elif allocation.get("design_stage") == "profile_extension":
        parent = allocations.get(allocation.get("parent_matrix_item_id"))
        if (
            parent is None
            or parent.get("design_stage") != "base"
            or parent.get("case_id") != allocation["case_id"]
            or parent.get("intrinsic_attributes") != intrinsic
        ):
            raise ValueError(
                "An extension needs a base parent with the same case and intrinsic attributes"
            )
        candidate = profiles.get(allocation["profile_id"])
        if candidate is not None and not _matches_profile(allocation, candidate):
            raise ValueError("An extension cannot redefine a candidate profile")
    else:
        raise ValueError("Unknown profile design stage")
    return effective


def build_matrix(cases, profiles, allocations, revision, sources):
    """Join cases and profile allocations without selecting by generated text."""
    cases, profiles = _inputs(cases, profiles)
    allocations = _index(allocations, "matrix_item_id")
    if not allocations:
        raise ValueError("The profile design is empty")
    rows, logical_units = [], set()
    for item, allocation in allocations.items():
        case = cases.get(allocation.get("case_id"))
        if case is None:
            raise ValueError("Every design allocation must resolve to a semantic case")
        effective = _validate_allocation(allocation, case, profiles, allocations)
        profile = profiles.get(allocation.get("profile_id"), {})
        group = case["target_group"]
        row = {
            **case,
            **profile,
            **allocation,
            "matrix_item_id": item,
            "bias_type_code": group,
            "target_code": "idp_current" if group == "idp" else "veteran_generic",
            "gender_track": {"women": "feminine", "men": "masculine"}.get(
                effective.get("gender"), "unspecified"
            ),
            "age_track": {
                "young": "young_adult",
                "young_adult": "young_adult",
                "middle_aged": "middle_age",
                "middle_age": "middle_age",
                "older": "older_adult",
                "older_adult": "older_adult",
            }.get(effective.get("age"), "unspecified"),
            "category_code": case["category_code"],
            "category_uk": case["category_uk"],
            "claim": case["source_claim_uk"],
            "context_uk": case["context"],
            "evidence_ids": case["evidence_ids"],
            "row_origin": "profile_design",
            "generation_status": "not_generated",
            "human_annotation_status": "not_annotated",
            "matrix_schema": "case_profile",
        }
        key = _logical_key(row)
        if key in logical_units:
            raise ValueError("Duplicate logical case, profile and reporter combination")
        logical_units.add(key)
        rows.append(row)
    for addition in revision.get("added_cases", []):
        if (
            not isinstance(addition.get("row_origin"), str)
            or not addition["row_origin"].strip()
            or addition.get("generation_status") != "not_generated"
            or addition.get("human_annotation_status") != "not_annotated"
        ):
            raise ValueError(
                "Feedback additions must retain their origin and pending review status"
            )
        rows.append(
            {**addition, "design_stage": "feedback", "claim": addition["stereotype_claim_uk"]}
        )
    apply_categories(rows, revision, assign_cases=True)
    unique_index(rows, "matrix_item_id")
    return bind_evidence(rows, sources)


def matrix_view(matrix):
    """Return a compact, readable CSV view of the generation design."""
    fields = (
        "matrix_item_id",
        "case_id",
        "category_code",
        "category_uk",
        "claim",
        "profile_id",
        "bias_type_code",
        "gender_track",
        "age_track",
        "occupation",
        "occupation_reference",
        "reporter",
        "framing",
        "design_stage",
        "evidence_ids",
        "source_type",
        "profile_attributes",
    )
    result = []
    for row in matrix:
        attributes = _effective_attributes(
            row.get("intrinsic_attributes", {}), row.get("additional_tags", {})
        )
        values = {
            **row,
            "occupation": attributes.get("occupation", ""),
            "profile_attributes": canonical_json(attributes),
        }
        result.append(
            {field: "" if values.get(field) is None else str(values[field]) for field in fields}
        )
    return result


def describe_design(cases, profiles, allocations, matrix):
    """Count the declared candidate pool, selected design and generation inputs."""
    cases, profiles = _inputs(cases, profiles)
    allocations = _index(allocations, "matrix_item_id")
    matrix = _index(matrix, "matrix_item_id")
    if not set(allocations) <= set(matrix) or any(
        row.get("design_stage") != "feedback"
        for item, row in matrix.items()
        if item not in allocations
    ):
        raise ValueError("The generation matrix must cover the full declared profile design")
    stages = Counter(row["design_stage"] for row in allocations.values())
    return {
        "candidate_combinations": sum(
            1 + len(cases[row["case_id"]]["reporters"]) for row in profiles.values()
        ),
        "selected_base": stages["base"],
        "profile_extensions": stages["profile_extension"],
        "feedback_additions": len(matrix) - len(allocations),
        "generation_rows": len(matrix),
    }
