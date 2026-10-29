"""Reconcile recorded selection changes with the current generation design."""

from fairforget.io import unique_index


def _identities(values, label):
    values = list(values)
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{label} needs nonempty IDs")
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate ID in {label}")
    return set(values)


def _index(rows, field, label):
    rows = list(rows)
    _identities((row.get(field) for row in rows), label)
    return unique_index(rows, field)


def _reason(record, field="reason"):
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"A selection change needs its recorded {field}")
    return value


def _stage(scope, unit, stage, start, removed=0, added=0):
    return {
        "scope": scope,
        "unit": unit,
        "stage": stage,
        "start": start,
        "removed": removed,
        "added": added,
        "result": start - removed + added,
    }


def _case_history(record, historical, cases, feedback):
    scope = record["scope_change"]
    removed = _index(scope["removed_cases"], "case_id", "case scope removals")
    added = _identities(scope["added_case_ids"], "case scope additions")
    superseded = _index(record["supersessions"], "case_id", "case supersessions")
    _reason(scope, "criterion")
    historical_ids, selected = set(historical), set(cases)
    inventory = selected | set(superseded)
    if (
        set(removed) & added
        or selected & set(superseded)
        or historical_ids - inventory != set(removed)
        or inventory - historical_ids != added
    ):
        raise ValueError("Case scope changes must match the historical and current case IDs")
    children = set()
    ledger = []
    for identity, row in removed.items():
        if {value["bias_type_code"] for value in historical[identity]} != {row["bias_type_code"]}:
            raise ValueError("A case removal must preserve its historical bias type")
        ledger.append(
            {
                "scope": "source_scope",
                "case_id": identity,
                "action": "remove",
                "reason": _reason(row),
                "current_status": "not_selected",
                "replacement_case_ids": [],
            }
        )
    for identity in sorted(added):
        ledger.append(
            {
                "scope": "source_scope",
                "case_id": identity,
                "action": "add",
                "reason": "Added in the recorded case specifications.",
                "current_status": "selected" if identity in selected else "superseded",
                "replacement_case_ids": [],
            }
        )
    for identity, row in superseded.items():
        replacements = _identities(row["replacement_case_ids"], "supersession children")
        if not replacements or not replacements <= selected or children & replacements:
            raise ValueError("Supersession children must be distinct selected cases")
        if not replacements <= added:
            raise ValueError("Supersession children must be recorded case additions")
        if any(
            "source_case_ids" in cases[child] and identity not in cases[child]["source_case_ids"]
            for child in replacements
        ):
            raise ValueError("Supersession children must retain their recorded source case")
        children.update(replacements)
        ledger.append(
            {
                "scope": "compound_supersession",
                "case_id": identity,
                "action": "supersede",
                "reason": _reason(row),
                "current_status": "superseded",
                "replacement_case_ids": sorted(replacements),
            }
        )
    for identity in sorted(feedback):
        ledger.append(
            {
                "scope": "current_generation",
                "case_id": identity,
                "action": "add",
                "reason": "Explicit feedback case in the generation matrix.",
                "current_status": "selected",
                "replacement_case_ids": [],
            }
        )
    stages = [
        _stage("case_design", "case_id", "source_scope", len(historical), len(removed), len(added)),
        _stage("case_design", "case_id", "compound_supersession", len(inventory), len(superseded)),
        _stage("current_generation", "case_id", "feedback_cases", len(cases), added=len(feedback)),
    ]
    return stages, ledger


def _replacement_history(record, profiles, allocations):
    removed = _identities(record["removed_ids"], "profile replacement removals")
    added = _identities(record["added_ids"], "profile replacement additions")
    all_selected = _index(allocations.values(), "design_unit_id", "selected design units")
    base = [row for row in allocations.values() if row["design_stage"] == "base"]
    selected = _index(base, "design_unit_id", "selected base design units")
    if removed & added or removed & set(all_selected) or not added <= set(selected):
        raise ValueError("Profile substitutions must match the selected base design units")
    if len(removed) != len(added):
        raise ValueError("Recorded profile substitutions must preserve the base design size")
    # The recorded substitutions concern unattributed (-f0) candidate profiles.
    candidate_units = {f"{identity}-f0" for identity in profiles}
    if not removed <= candidate_units:
        raise ValueError("Removed profile alternatives must remain in the candidate pool")
    reason = _reason(record)
    ledger = [
        {
            "design_unit_id": identity,
            "action": action,
            "reason": reason,
            "current_status": status,
        }
        for identities, action, status in (
            (removed, "remove", "candidate_only"),
            (added, "add", "selected"),
        )
        for identity in sorted(identities)
    ]
    return (
        _stage(
            "base_design",
            "design_unit_id",
            "profile_substitution",
            len(base),
            len(removed),
            len(added),
        ),
        ledger,
    )


def _count(record, field):
    value = record.get(field)
    if type(value) is not int or value < 0:
        raise ValueError(f"A profile exclusion needs a nonnegative {field}")
    return value


def _exclusion_history(record, allocations):
    all_removed = _identities(record["removed_ids"], "historical profile exclusions")
    start = _count(record, "previous_rows_per_language")
    if (
        start != len(allocations)
        or _count(record, "removed_rows_per_language") != len(all_removed)
        or _count(record, "remaining_rows_per_language") != start - len(all_removed)
        or not all_removed <= set(allocations)
    ):
        raise ValueError("Historical exclusions must reconcile with the retained profile design")
    removed, stages, ledger = set(), [], []
    for number, decision in enumerate(record["decisions"], start=1):
        identities = _identities(decision["removed_ids"], "exclusion decision")
        if (
            removed & identities
            or _count(decision, "previous_rows_per_language") != start
            or _count(decision, "removed_rows_per_language") != len(identities)
            or _count(decision, "remaining_rows_per_language") != start - len(identities)
            or not identities <= all_removed
        ):
            raise ValueError("Historical exclusion decisions must form a disjoint count chain")
        _reason(decision, "criterion")
        reason = _reason(decision)
        stages.append(
            _stage(
                "historical_reference",
                "triplet_id_per_language",
                f"profile_exclusion_{number}",
                start,
                len(identities),
            )
        )
        for identity in sorted(identities):
            row = allocations[identity]
            ledger.append(
                {
                    "matrix_item_id": identity,
                    "case_id": row["case_id"],
                    "profile_id": row["profile_id"],
                    "reason": reason,
                    "historical_status": "removed",
                    "current_status": "retained",
                }
            )
        removed.update(identities)
        start -= len(identities)
    if removed != all_removed or start != record["remaining_rows_per_language"]:
        raise ValueError("Historical exclusion decisions must cover the complete removal ledger")
    return stages, ledger


def reconcile_selection(
    case_changes,
    profile_replacements,
    profile_exclusions,
    *,
    historical_matrix,
    cases,
    profiles,
    allocations,
    matrix,
):
    """Return count stages and change ledgers without changing source records."""
    historical_rows = _index(historical_matrix, "matrix_item_id", "historical matrix")
    historical = {}
    for row in historical_rows.values():
        historical.setdefault(row["case_id"], []).append(row)
    cases = _index(cases, "case_id", "construction cases")
    profiles = _index(profiles, "profile_id", "candidate profiles")
    allocations = _index(allocations, "matrix_item_id", "profile allocations")
    matrix = _index(matrix, "matrix_item_id", "generation matrix")
    if not allocations or any(
        row.get("case_id") not in cases
        or row.get("design_stage") not in {"base", "profile_extension"}
        or row.get("design_stage") == "base"
        and (
            row.get("profile_id") not in profiles
            or profiles[row["profile_id"]].get("case_id") != row.get("case_id")
        )
        for row in allocations.values()
    ):
        raise ValueError("Allocations must resolve to selected cases and declared design stages")
    if {row["case_id"] for row in allocations.values()} != set(cases) or any(
        row.get("case_id") not in cases for row in profiles.values()
    ):
        raise ValueError("The profile design must cover the selected construction cases")
    if not set(allocations) <= set(matrix) or any(
        any(
            matrix[identity].get(field) != row.get(field)
            for field in ("case_id", "profile_id", "design_stage", "design_unit_id")
        )
        for identity, row in allocations.items()
    ):
        raise ValueError("The generation matrix must retain every declared allocation")
    feedback = [row for identity, row in matrix.items() if identity not in allocations]
    if any(row.get("design_stage") != "feedback" for row in feedback):
        raise ValueError("Extra generation inputs must be explicit feedback additions")
    feedback_cases = _identities((row["case_id"] for row in feedback), "feedback cases")
    if feedback_cases & set(cases):
        raise ValueError("Feedback additions must use distinct new case IDs")
    stages, case_ledger = _case_history(case_changes, historical, cases, feedback_cases)
    replacement_stage, replacement_ledger = _replacement_history(
        profile_replacements, profiles, allocations
    )
    exclusion_stages, exclusion_ledger = _exclusion_history(profile_exclusions, allocations)
    stages.extend([replacement_stage, *exclusion_stages])
    stages.append(
        _stage(
            "current_generation",
            "matrix_item_id",
            "feedback_additions",
            len(allocations),
            added=len(feedback),
        )
    )
    return {
        "stages": stages,
        "case_changes": case_ledger,
        "profile_replacements": replacement_ledger,
        "profile_exclusions": exclusion_ledger,
    }
