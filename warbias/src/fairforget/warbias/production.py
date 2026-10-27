"""Full generation is gated by an explicit evidence-bound human selection."""

from copy import deepcopy
from pathlib import Path

from fairforget.io import digest, unique_index, write_json, write_rows
from fairforget.warbias.generation import (
    freeze_file,
    generate_triplets,
    input_id,
    validate_cases,
    validate_source_language,
)
from fairforget.warbias.pilot import verify_pilot_plan


def verify_production_result(rows, manifest):
    """Require the exact complete generation artifact before a downstream stage."""
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "warbias_full_generation_result"
        or manifest.get("status") != "complete"
        or manifest.get("manifest_sha256")
        != digest({key: value for key, value in manifest.items() if key != "manifest_sha256"})
    ):
        raise ValueError("A complete content-bound full-generation manifest is required")
    validate_cases(rows)
    unique_index(rows)
    language = manifest.get("source_language")
    validate_source_language(language)
    if (
        manifest.get("rows_sha256") != digest(rows)
        or manifest.get("row_count") != len(rows)
        or manifest.get("case_count") != len({row["case_id"] for row in rows})
        or manifest.get("unit_count") != len(rows)
        or manifest.get("case_ids") != [row["case_id"] for row in rows]
        or manifest.get("unit_ids") != [row.get("generation", {}).get("input_id") for row in rows]
    ):
        raise ValueError("Full-generation rows differ from their complete manifest")
    for row in rows:
        generation = row.get("generation", {})
        if (
            row.get("language") != language
            or row.get("id") != generation.get("input_id")
            or generation.get("purpose") != "full_generation"
            or not generation.get("raw_call_sha256")
            or not manifest.get("production_plan_sha256")
            or not manifest.get("selection_sha256")
            or generation.get("production_plan_sha256") != manifest["production_plan_sha256"]
            or generation.get("selection_sha256") != manifest["selection_sha256"]
        ):
            raise ValueError("Full-generation row language or selection lineage differs")
    return manifest


def generate_full(cases, pilot_plan, selection_receipt, output, *, provider=None, execute=False):
    """Run only the selected model/strategy, without changing its frozen prompts."""
    from fairforget.warbias.selection import verify_selection

    verify_pilot_plan(pilot_plan)
    selected = verify_selection(selection_receipt, pilot_plan)
    cases = deepcopy(list(cases))
    validate_cases(cases)
    plan = {
        "schema_version": 1,
        "kind": "warbias_full_generation",
        "pilot_plan_sha256": pilot_plan["plan_sha256"],
        "selection_sha256": digest(selection_receipt),
        "selected_condition": selected,
        "cases": cases,
        "cases_sha256": digest(cases),
        "case_ids": [case["case_id"] for case in cases],
        "unit_ids": [input_id(case) for case in cases],
    }
    plan["plan_sha256"] = digest(plan)
    output = Path(output)
    freeze_file(output / "production-plan.json", plan)
    rows = generate_triplets(
        cases,
        selected["candidate"],
        selected["strategy"],
        selected["strategy_config"],
        selected["source_language"],
        output / "generation",
        condition_id=selected["condition_id"],
        provenance={
            "purpose": "full_generation",
            "pilot_plan_sha256": pilot_plan["plan_sha256"],
            "selection_sha256": plan["selection_sha256"],
            "production_plan_sha256": plan["plan_sha256"],
        },
        provider=provider,
        execute=execute,
    )
    if rows is None:
        return None
    write_rows(output / "triplets.jsonl", rows)
    manifest = {
        "schema_version": 1,
        "kind": "warbias_full_generation_result",
        "status": "complete",
        "production_plan_sha256": plan["plan_sha256"],
        "selection_sha256": plan["selection_sha256"],
        "source_language": selected["source_language"],
        "case_ids": plan["case_ids"],
        "unit_ids": plan["unit_ids"],
        "case_count": len(set(plan["case_ids"])),
        "unit_count": len(cases),
        "row_count": len(rows),
        "rows_sha256": digest(rows),
        "human_validated": False,
    }
    manifest["manifest_sha256"] = digest(manifest)
    write_json(output / "production-result.json", manifest)
    return rows
