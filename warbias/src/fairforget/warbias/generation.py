"""Two explicit prompting strategies over the same frozen source cases.

Sequential generation passes S to C, then S and C to U. Human assessment
belongs to the subsequent blinded pilot review, not between model calls.
"""

import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path

from fairforget.io import digest, unique_index, write_json
from fairforget.warbias.provider import prompt_record, request_record, response_text, run_requests
from fairforget.warbias.schema import FIELDS, STAGE_FIELD, STAGES, structural_issues

STRATEGIES = {"zero_shot": ("triplet",), "sequential": STAGES}
GENERATION_CONTRACT = "warbias_triplet_generation_v1"


def freeze_strategies(strategies):
    """Snapshot prompt bytes and explicit response-field mappings for both strategies."""
    if set(strategies) != set(STRATEGIES):
        raise ValueError("The pilot needs zero_shot and sequential prompting strategies")
    frozen = deepcopy(strategies)
    for name, strategy in frozen.items():
        prompts = strategy.get("prompts", {})
        if set(prompts) != set(STRATEGIES[name]):
            raise ValueError(f"{name}: declare its exact ordered stage prompts")
        strategy["prompts"] = {
            stage: prompt_record(prompt) if not isinstance(prompt, dict) else prompt
            for stage, prompt in prompts.items()
        }
        strategy.setdefault("response_fields", dict(zip(FIELDS, FIELDS)))
        verify_strategy(name, strategy)
    return frozen


def verify_strategy(name, strategy):
    if name not in STRATEGIES or set(strategy.get("prompts", {})) != set(STRATEGIES[name]):
        raise ValueError("Unknown strategy or missing stage prompt")
    fields = strategy.get("response_fields", {})
    if (
        set(fields) != set(FIELDS)
        or any(not isinstance(value, str) or not value for value in fields.values())
        or len(set(fields.values())) != len(FIELDS)
    ):
        raise ValueError("Declare one distinct response key for every triplet field")
    for prompt in strategy["prompts"].values():
        text = prompt.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Every generation prompt needs its exact nonempty text")
        if prompt.get("sha256") != hashlib.sha256(text.encode("utf-8")).hexdigest():
            raise ValueError("Generation prompt text differs from its SHA256")


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Generation response contains a duplicate field")
        result[key] = value
    return result


def parse_response(text, response_fields, expected_fields=FIELDS):
    """Decode only the declared JSON fields, preserving the provider's exact strings."""
    payload = json.loads(text, object_pairs_hook=_json_object)
    keys = {response_fields[field] for field in expected_fields}
    if (
        not isinstance(payload, dict)
        or set(payload) != keys
        or any(not isinstance(value, str) or not value.strip() for value in payload.values())
    ):
        raise ValueError("Generation response must contain exactly the declared string fields")
    return {field: payload[response_fields[field]] for field in expected_fields}


def input_id(row):
    """Identify one matrix/profile unit independently of its semantic case family."""
    return row.get("id") or row.get("matrix_id") or row.get("case_id")


def validate_cases(cases):
    if not cases:
        raise ValueError("Generation needs an explicitly selected nonempty case set")
    indexed = {}
    for case in cases:
        item = input_id(case)
        if not isinstance(item, str) or not item.strip():
            raise ValueError("Generation input IDs must be nonempty strings")
        if not isinstance(case.get("case_id"), str) or not case["case_id"].strip():
            raise ValueError("Every generation input needs its semantic case ID")
        if item in indexed:
            raise ValueError(f"Duplicate generation input ID: {item}")
        indexed[item] = case
    return indexed


def validate_candidate(candidate):
    for field in ("candidate_id", "provider", "model"):
        if not isinstance(candidate.get(field), str) or not candidate[field].strip():
            raise ValueError(f"Each candidate needs an explicit {field}")
    if not isinstance(candidate.get("settings", {}), dict):
        raise ValueError("Candidate generation settings must be an object")
    if candidate.get("revision") is not None and (
        not isinstance(candidate["revision"], str) or not candidate["revision"].strip()
    ):
        raise ValueError("Model revision must be a recorded string or null")


def freeze_file(path, value):
    """Resume an identical plan while refusing to overwrite a different experiment."""
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError("Existing output belongs to a different generation plan")
    else:
        write_json(path, value)


def condition_definition(
    cases, candidate, strategy_name, strategy, source_language, *, condition_id, provenance
):
    """Shared identity for planned requests, generated rows and blinded review checks."""
    validate_cases(cases)
    validate_candidate(candidate)
    verify_strategy(strategy_name, strategy)
    if not isinstance(source_language, str) or not source_language.strip():
        raise ValueError("Declare the generation source language explicitly")
    if not isinstance(condition_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]*", condition_id
    ):
        raise ValueError("A condition needs a nonempty portable identifier")
    return {
        "contract": GENERATION_CONTRACT,
        "condition_id": condition_id,
        "candidate": deepcopy(candidate),
        "strategy": strategy_name,
        "strategy_config": deepcopy(strategy),
        "source_language": source_language,
        "cases_sha256": digest(cases),
        "provenance": deepcopy(provenance),
        "stage_order": list(STRATEGIES[strategy_name]),
    }


def generate_triplets(
    cases,
    candidate,
    strategy_name,
    strategy,
    source_language,
    output,
    *,
    condition_id,
    provenance,
    provider=None,
    execute=False,
):
    """Preview or resume one condition; caches retain exact prompts and parent text."""
    output = Path(output)
    definition = condition_definition(
        cases,
        candidate,
        strategy_name,
        strategy,
        source_language,
        condition_id=condition_id,
        provenance=provenance,
    )
    freeze_file(output / "condition.json", definition)
    condition_hash = digest(definition)
    parents = {input_id(case): {} for case in cases}
    calls = {input_id(case): [] for case in cases}
    for stage in STRATEGIES[strategy_name]:
        requests = []
        for case in cases:
            item = input_id(case)
            state = {
                "case": case,
                "source_language": source_language,
                "previous": {
                    strategy["response_fields"][field]: text
                    for field, text in parents[item].items()
                },
            }
            request = request_record(item, stage, state, candidate, strategy["prompts"][stage])
            request.update(
                generation_contract=GENERATION_CONTRACT,
                condition_sha256=condition_hash,
                case_sha256=digest(case),
            )
            requests.append(request)
        raw = run_requests(requests, output / stage, provider=provider, execute=execute)
        if raw is None:
            # Later stages depend on real preceding text; do not preview invented parents.
            return None
        expected = unique_index(requests)
        for call in raw:
            item = call["id"]
            if (
                call.get("request") != expected[item]
                or call.get("request_sha256") != digest(expected[item])
                or call.get("response_sha256") != digest(call.get("response"))
            ):
                raise ValueError("Generation request/response evidence differs")
            fields = FIELDS if stage == "triplet" else (STAGE_FIELD[stage],)
            parents[item].update(
                parse_response(response_text(call), strategy["response_fields"], fields)
            )
            calls[item].append(digest(call))
    rows = []
    for case in cases:
        item = input_id(case)
        row = {
            **deepcopy(case),
            "id": item
            if provenance.get("purpose") == "full_generation"
            else digest({"condition_sha256": condition_hash, "input_id": item}),
            "case_id": case["case_id"],
            "case": deepcopy(case),
            "language": source_language,
            **parents[item],
            "generation": {
                **deepcopy(provenance),
                "condition_id": condition_id,
                "input_id": item,
                "condition_sha256": condition_hash,
                "candidate_id": candidate["candidate_id"],
                "strategy": strategy_name,
                "provider": candidate["provider"],
                "model": candidate["model"],
                "revision": candidate.get("revision"),
                "revision_status": "recorded" if candidate.get("revision") else "unrecorded",
                "settings": deepcopy(candidate.get("settings", {})),
                "prompt_sha256": {
                    key: prompt["sha256"] for key, prompt in strategy["prompts"].items()
                },
                "raw_call_sha256": calls[item],
            },
            "human_validated": False,
            "review_status": "pending",
        }
        row["mechanical_issues"] = structural_issues(row)
        rows.append(row)
    return rows
