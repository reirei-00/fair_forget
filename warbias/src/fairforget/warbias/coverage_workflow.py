"""Rebuild the completed coverage design and verify saved generation outputs."""

from collections import Counter
from pathlib import Path

from fairforget.cli_support import load_json
from fairforget.dataset_workflow import load_inputs
from fairforget.io import digest, file_digest, read_rows, unique_index
from fairforget.warbias.coverage import prepare_generation_cases
from fairforget.warbias.coverage_design import enumerate_items, select_items
from fairforget.warbias.evidence import source_records
from fairforget.warbias.schema import FIELDS

CONTEXT_FIELDS = (
    "case_id",
    "target_group",
    "target_code",
    "profile_id",
    "intrinsic_attributes",
    "additional_tags",
    "occupation_reference",
    "reporter",
    "target_uk",
    "reporter_uk",
    "claim",
    "evidence_ids",
    "source_type",
)


def prepare_coverage(project):
    """Rebuild historical inputs before applying the current dataset scope."""
    project = Path(project)
    scope = load_json(project / "protocol/dataset_scope.json")
    if scope.get("schema_version") != 1 or scope.get("source_language") != "uk":
        raise ValueError("Declare a versioned Ukrainian dataset scope")
    record = scope["recorded_design"]
    for relative, expected in scope["source_files_sha256"].items():
        if file_digest(project / relative) != expected:
            raise ValueError(f"Recorded coverage source differs: {relative}")
    inputs = load_inputs(project)
    protocol = load_json(project / "protocol/coverage_extension.json")
    dimensions = load_json(project / protocol["dimensions"]["catalog"])
    applicability = read_rows(project / protocol["dimensions"]["applicability_by_case"])
    rules = unique_index(applicability, "case_id")
    cohort = protocol["case_selection"]["annotation_cohort"]
    if file_digest(project / cohort["path"]) != cohort["sha256"]:
        raise ValueError("The annotation cohort differs from its recorded hash")
    existing_ids = [row["id"] for row in read_rows(project / cohort["path"])]
    categories = {
        row["case_id"]: {name: row[name] for name in ("category_code", "category_uk")}
        for row in inputs["matrix_reviewed"]
    }
    additions = read_rows(project / protocol["case_selection"]["additional_cases"])
    cases = [
        {
            **case,
            **{
                name: rules[case["case_id"]][name]
                for name in ("intrinsic_attributes", "target_subgroups", "occupation_reference")
            },
            **categories.get(case["case_id"], {}),
        }
        for case in [*inputs["construction_cases"], *additions]
    ]
    candidates = enumerate_items(
        cases,
        applicability,
        dimensions["axis_values"],
        inputs["profile_design"],
        existing_ids=existing_ids,
    )
    items = select_items(
        candidates,
        cases,
        inputs["profile_design"],
        occupation_limit=dimensions["occupation_limit_per_case"],
        deferred_dimensions=dimensions["deferred_new_dimensions"],
        seed=dimensions["occupation_selection_seed"],
    )
    sources = source_records(
        inputs["sources"], inputs["source_feedback"], inputs["source_compilation"]
    )
    new_inputs = prepare_generation_cases(items, cases, sources, existing_ids=existing_ids)
    if digest(new_inputs) != record["new_inputs_sha256"]:
        raise ValueError("Rebuilt coverage inputs differ from the completed generation")
    if len(candidates) != record["candidate_triplets"] or len(items) != record["total_triplets"]:
        raise ValueError("Rebuilt coverage counts differ from the recorded design")
    groups = set(scope["included_target_groups"])
    if (
        not groups
        or groups - {row["target_group"] for row in cases}
        or groups & set(scope["deferred_target_groups"])
    ):
        raise ValueError("Dataset scope must name known coverage groups")
    selected = [row for row in items if row["target_group"] in groups]
    current_inputs = [row for row in new_inputs if row["target_group"] in groups]
    current_cohort = [row["cohort_item_id"] for row in selected if row.get("cohort_item_id")]
    if set(current_cohort) != set(existing_ids):
        raise ValueError("The current scope must preserve every existing annotation item")
    counts = {
        "existing_triplets": len(current_cohort),
        "new_triplets": len(current_inputs),
        "total_triplets": len(selected),
        "by_group": {
            group: {
                "existing": sum(
                    bool(row.get("cohort_item_id"))
                    for row in selected
                    if row["target_group"] == group
                ),
                "new": sum(row["target_group"] == group for row in current_inputs),
                "total": sum(row["target_group"] == group for row in selected),
            }
            for group in sorted(groups)
        },
    }
    if counts != scope["expected_counts"]:
        raise ValueError("The current dataset counts differ from its selected scope")
    return {
        "scope": scope,
        "dimensions": dimensions,
        "cases": [row for row in cases if row["target_group"] in groups],
        "items": selected,
        "inputs": current_inputs,
        "cohort_ids": current_cohort,
        "counts": counts,
    }


def _verify_context(rows, inputs):
    expected = unique_index(inputs)
    actual = unique_index(rows)
    if actual.keys() != expected.keys():
        raise ValueError("Saved generation IDs differ from the rebuilt input IDs")
    for identity, row in actual.items():
        if any(row.get(name) != expected[identity].get(name) for name in CONTEXT_FIELDS):
            raise ValueError("Saved generation context differs from its input")
        if row.get("language") != "uk" or row.get("human_validated") is not False:
            raise ValueError("Saved coverage rows must be Ukrainian and not human validated")
        if any(not isinstance(row.get(name), str) or not row[name].strip() for name in FIELDS):
            raise ValueError("Saved coverage rows need all three generated sentences")


def verify_generated_rows(plan, rows):
    """Check the complete current corpus without discarding flagged triplets."""
    rows = list(rows)
    _verify_context(rows, plan["inputs"])
    expected_order = [row["id"] for row in plan["inputs"]]
    if [row["id"] for row in rows] != expected_order:
        raise ValueError("Saved generation order differs from its recorded input order")
    summary = {
        "status": "verified",
        "new_triplets": len(rows),
        "by_group": dict(Counter(row["target_group"] for row in rows)),
        "rows_sha256": digest(rows),
        "human_validated": False,
        "rows_with_mechanical_issues": sum(bool(row.get("mechanical_issues")) for row in rows),
    }
    return summary
