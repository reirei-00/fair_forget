"""Prepare explicit profile allocations for source generation and blind review."""

SUPPORTED_ATTRIBUTES = {
    "age",
    "caregiving",
    "community_type",
    "disability",
    "discharge_recency",
    "displacement_origin",
    "displacement_wave",
    "dual_status",
    "ethnicity",
    "gender",
    "help_seeking",
    "housing_type",
    "language",
    "occupation",
    "origin_region",
    "pets",
    "prior_occupation",
    "rank",
    "recruitment_route",
    "region",
    "registration",
    "service_duration",
    "service_period",
    "service_role",
    "service_state",
}


def build_profile_matrix(cases, profiles, allocations):
    """Resolve an explicit frozen allocation without inventing eligible combinations."""
    from fairforget.io import digest, unique_index

    cases = unique_index(cases, "case_id")
    profiles = unique_index(profiles, "profile_id")
    unique_index(allocations)
    output = []
    for allocation in allocations:
        case = cases[allocation["case_id"]]
        profile = profiles[allocation["profile_id"]]
        intrinsic = dict(case.get("intrinsic_attributes", {}))
        additional = dict(profile.get("additional_tags", {}))
        if set(intrinsic) & set(additional):
            raise ValueError("Profile attributes cannot overwrite a case's intrinsic attributes")
        if profile.get("target_group", case["target_group"]) != case["target_group"]:
            raise ValueError("Profile target group differs from the source case")
        row = {
            **allocation,
            "target_group": case["target_group"],
            "claim_family_id": case["claim_family_id"],
            "case_spec_sha256": case["case_spec_sha256"],
            "intrinsic_attributes": intrinsic,
            "additional_tags": additional,
            "profile_sha256": digest(profile),
            "human_validated": False,
        }
        row["allocation_sha256"] = digest(allocation)
        output.append(row)
    return output
