"""Full generation is gated by an explicit evidence-bound human selection."""

import re
from copy import deepcopy
from pathlib import Path

from warbias_kit.io import digest, unique_index, write_json, write_rows
from warbias_kit.warbias.generation import (
    STRATEGIES,
    freeze_file,
    generate_triplets,
    input_id,
    validate_cases,
    validate_source_language,
    verify_strategy,
)
from warbias_kit.warbias.pilot import verify_pilot_plan


def resolve_final_strategy(selected, final_strategy=None):
    """Allow revised prompt bytes while preserving the selected strategy and settings."""
    pilot_strategy = selected["strategy_config"]
    strategy = deepcopy(pilot_strategy if final_strategy is None else final_strategy)
    verify_strategy(selected["strategy"], strategy)
    if {key: value for key, value in strategy.items() if key != "prompts"} != {
        key: value for key, value in pilot_strategy.items() if key != "prompts"
    }:
        raise ValueError("Final prompts cannot change the selected response fields or settings")
    return strategy


def _prompt_hashes(strategy):
    return {stage: prompt["sha256"] for stage, prompt in strategy["prompts"].items()}


def _verify_prompt_revision(revision):
    if not isinstance(revision, dict) or set(revision) != {
        "strategy",
        "pilot_prompt_sha256",
        "final_prompt_sha256",
    }:
        raise ValueError("Full-generation prompt revision is incomplete")
    name = revision["strategy"]
    stages = STRATEGIES.get(name) if isinstance(name, str) else None
    for name in ("pilot_prompt_sha256", "final_prompt_sha256"):
        hashes = revision[name]
        if (
            stages is None
            or not isinstance(hashes, dict)
            or set(hashes) != set(stages)
            or any(
                not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value)
                for value in hashes.values()
            )
        ):
            raise ValueError("Full-generation prompt revision hashes differ")


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
    revision = manifest.get("prompt_revision")
    if revision is not None:
        _verify_prompt_revision(revision)
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
            or generation.get("execution_config_sha256") != manifest.get("execution_config_sha256")
            or generation.get("api_equivalence") != manifest.get("api_equivalence")
            or generation.get("prompt_revision") != revision
            or revision is not None
            and (
                generation.get("strategy") != revision["strategy"]
                or generation.get("prompt_sha256") != revision["final_prompt_sha256"]
            )
        ):
            raise ValueError("Full-generation row language or selection lineage differs")
    return manifest


def write_production_result(rows, plan, output, *, source_language, lineage=None):
    """Write the common full-generation rows and content-bound result manifest."""
    output = Path(output)
    write_rows(output / "triplets.jsonl", rows)
    manifest = {
        "schema_version": 1,
        "kind": "warbias_full_generation_result",
        "status": "complete",
        "production_plan_sha256": plan["plan_sha256"],
        "selection_sha256": plan["selection_sha256"],
        "source_language": source_language,
        "case_ids": plan["case_ids"],
        "unit_ids": plan["unit_ids"],
        "case_count": len(set(plan["case_ids"])),
        "unit_count": len(plan["cases"]),
        "row_count": len(rows),
        "rows_sha256": digest(rows),
        "human_validated": False,
        **(lineage or {}),
    }
    manifest["manifest_sha256"] = digest(manifest)
    write_json(output / "production-result.json", manifest)
    return manifest


def generate_full(
    cases,
    pilot_plan,
    selection_receipt,
    output,
    *,
    final_strategy=None,
    provider=None,
    execute=False,
):
    """Run the selected condition, with an optional explicit final prompt revision."""
    from warbias_kit.warbias.selection import verify_selection

    verify_pilot_plan(pilot_plan)
    selected = verify_selection(selection_receipt, pilot_plan)
    strategy = resolve_final_strategy(selected, final_strategy)
    revision = (
        {
            "strategy": selected["strategy"],
            "pilot_prompt_sha256": _prompt_hashes(selected["strategy_config"]),
            "final_prompt_sha256": _prompt_hashes(strategy),
        }
        if final_strategy is not None
        else None
    )
    revision_binding = {"prompt_revision": revision} if revision is not None else {}
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
        **({"final_strategy": strategy, **revision_binding} if revision is not None else {}),
    }
    plan["plan_sha256"] = digest(plan)
    output = Path(output)
    freeze_file(output / "production-plan.json", plan)
    rows = generate_triplets(
        cases,
        selected["candidate"],
        selected["strategy"],
        strategy,
        selected["source_language"],
        output / "generation",
        condition_id=selected["condition_id"],
        provenance={
            "purpose": "full_generation",
            "pilot_plan_sha256": pilot_plan["plan_sha256"],
            "selection_sha256": plan["selection_sha256"],
            "production_plan_sha256": plan["plan_sha256"],
            **revision_binding,
        },
        provider=provider,
        execute=execute,
    )
    if rows is None:
        return None
    write_production_result(
        rows, plan, output, source_language=selected["source_language"], lineage=revision_binding
    )
    return rows
