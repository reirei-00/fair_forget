"""Frozen pilot cases crossed with explicit model candidates and two strategies."""

import re
from copy import deepcopy
from pathlib import Path

from warbias_kit.io import digest, unique_index, write_json, write_rows
from warbias_kit.warbias.generation import (
    STRATEGIES,
    condition_definition,
    freeze_file,
    freeze_strategies,
    generate_triplets,
    input_id,
    validate_candidate,
    validate_cases,
    validate_source_language,
    verify_strategy,
)


def freeze_pilot(cases, candidates, strategies, *, source_language):
    """Freeze supplied cases without sampling or choosing a preferred model."""
    validate_source_language(source_language)
    cases, candidates = deepcopy(list(cases)), deepcopy(list(candidates))
    validate_cases(cases)
    for candidate in candidates:
        validate_candidate(candidate)
        candidate.setdefault("revision", None)
        candidate.setdefault("settings", {})
    unique_index(candidates, "candidate_id")
    if len({(row["provider"], row["model"]) for row in candidates}) < 2:
        raise ValueError("The model-selection pilot needs at least two distinct model candidates")
    conditions = [
        {
            "condition_id": f"condition_{index + 1:02}",
            "candidate_id": candidate["candidate_id"],
            "strategy": strategy,
        }
        for index, (candidate, strategy) in enumerate(
            (candidate, strategy) for candidate in candidates for strategy in STRATEGIES
        )
    ]
    plan = {
        "schema_version": 1,
        "kind": "warbias_generation_pilot",
        "source_language": source_language,
        "cases": cases,
        "case_ids": [case["case_id"] for case in cases],
        "unit_ids": [input_id(case) for case in cases],
        "cases_sha256": digest(cases),
        "candidates": candidates,
        "strategies": freeze_strategies(strategies),
        "conditions": conditions,
    }
    plan["plan_sha256"] = digest(plan)
    return plan


def verify_pilot_plan(plan):
    """Reject stale cases, prompts or an incomplete candidate-by-strategy grid."""
    if (
        plan.get("schema_version") != 1
        or plan.get("kind") != "warbias_generation_pilot"
        or plan.get("plan_sha256") != digest({k: v for k, v in plan.items() if k != "plan_sha256"})
    ):
        raise ValueError("Pilot plan identity differs")
    validate_cases(plan["cases"])
    if (
        plan.get("case_ids") != [case["case_id"] for case in plan["cases"]]
        or plan.get("unit_ids") != [input_id(case) for case in plan["cases"]]
        or plan.get("cases_sha256") != digest(plan["cases"])
    ):
        raise ValueError("Frozen pilot cases differ")
    validate_source_language(plan.get("source_language"))
    candidates = unique_index(plan["candidates"], "candidate_id")
    for candidate in candidates.values():
        validate_candidate(candidate)
    if len({(row["provider"], row["model"]) for row in candidates.values()}) < 2:
        raise ValueError("The model-selection pilot needs at least two distinct model candidates")
    if set(plan["strategies"]) != set(STRATEGIES):
        raise ValueError("Pilot prompting strategies differ")
    for name, strategy in plan["strategies"].items():
        verify_strategy(name, strategy)
    conditions = plan["conditions"]
    unique_index(conditions, "condition_id")
    expected = {(candidate, strategy) for candidate in candidates for strategy in STRATEGIES}
    actual = [(row["candidate_id"], row["strategy"]) for row in conditions]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError(
            "Pilot conditions must cover every candidate and both strategies exactly once"
        )
    return plan


def verify_pilot_result(plan, result, rows=None):
    """Verify a complete pilot receipt; optional rows must be the entire saved grid.

    A blinded subset can verify the manifest first, then match each selected row
    to its individual hash without changing the full pilot's coverage claim.
    """
    verify_pilot_plan(plan)
    if (
        result.get("schema_version") != 1
        or result.get("kind") != "warbias_generation_pilot_result"
        or result.get("pilot_plan_sha256") != plan["plan_sha256"]
        or result.get("result_sha256")
        != digest({key: value for key, value in result.items() if key != "result_sha256"})
    ):
        raise ValueError("Pilot result differs from its frozen plan or content identity")
    candidates = unique_index(plan["candidates"], "candidate_id")
    expected = set()
    for condition in plan["conditions"]:
        definition = condition_definition(
            plan["cases"],
            candidates[condition["candidate_id"]],
            condition["strategy"],
            plan["strategies"][condition["strategy"]],
            plan["source_language"],
            condition_id=condition["condition_id"],
            provenance={"pilot_plan_sha256": plan["plan_sha256"], "purpose": "pilot"},
        )
        expected.update(
            digest({"condition_sha256": digest(definition), "input_id": unit})
            for unit in plan["unit_ids"]
        )
    hashes = result.get("row_sha256", {})
    if (
        result.get("case_count") != len(set(plan["case_ids"]))
        or result.get("unit_count") != len(plan["cases"])
        or result.get("condition_count") != len(plan["conditions"])
        or result.get("row_count") != len(expected)
        or not isinstance(result.get("rows_sha256"), str)
        or not re.fullmatch(r"[a-f0-9]{64}", result["rows_sha256"])
        or not isinstance(hashes, dict)
        or set(hashes) != expected
        or any(
            not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value)
            for value in hashes.values()
        )
    ):
        raise ValueError("Pilot result does not cover the complete frozen generation grid")
    if rows is not None:
        indexed = unique_index(rows)
        if (
            result.get("rows_sha256") != digest(rows)
            or {item: digest(row) for item, row in indexed.items()} != hashes
        ):
            raise ValueError("Pilot output text or generation evidence changed after collection")
    return result


def run_pilot(plan, output, *, provider=None, provider_factory=None, execute=False):
    """Preview by default; load one candidate provider at a time when executing.

    Factories receive ``config=candidate``. A provider's optional ``close`` method
    runs after its two strategies, including on failure. Adapters without close
    must manage their runtime resources themselves.
    """
    if provider is not None and provider_factory is not None:
        raise ValueError("Supply a provider or a candidate provider factory, not both")
    verify_pilot_plan(plan)
    output = Path(output)
    freeze_file(output / "pilot-plan.json", plan)
    rows = []
    for candidate in plan["candidates"]:
        candidate_provider = (
            provider_factory(config=deepcopy(candidate))
            if execute and provider_factory is not None
            else provider
        )
        try:
            for condition in plan["conditions"]:
                if condition["candidate_id"] != candidate["candidate_id"]:
                    continue
                generated = generate_triplets(
                    plan["cases"],
                    candidate,
                    condition["strategy"],
                    plan["strategies"][condition["strategy"]],
                    plan["source_language"],
                    output / condition["condition_id"],
                    condition_id=condition["condition_id"],
                    provenance={"pilot_plan_sha256": plan["plan_sha256"], "purpose": "pilot"},
                    provider=candidate_provider,
                    execute=execute,
                )
                if generated is not None:
                    rows.extend(generated)
        finally:
            if execute and provider_factory is not None:
                close = getattr(candidate_provider, "close", None)
                if callable(close):
                    close()
                close = None
                candidate_provider = None
    if not execute:
        return None
    write_rows(output / "triplets.jsonl", rows)
    result = {
        "schema_version": 1,
        "kind": "warbias_generation_pilot_result",
        "pilot_plan_sha256": plan["plan_sha256"],
        "case_count": len(set(plan["case_ids"])),
        "unit_count": len(plan["cases"]),
        "condition_count": len(plan["conditions"]),
        "row_count": len(rows),
        "rows_sha256": digest(rows),
        "row_sha256": {row["id"]: digest(row) for row in rows},
        "human_validated": False,
    }
    result["result_sha256"] = digest(result)
    verify_pilot_result(plan, result, rows)
    write_json(output / "pilot-result.json", result)
    return rows
