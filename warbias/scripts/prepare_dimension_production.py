"""Prepare reviewed, source-bound Ukrainian dimension slots for the existing Lapa runner."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from fairforget.dataset_workflow import load_inputs
from fairforget.io import digest, file_digest, read_rows, unique_index, write_json, write_rows
from fairforget.warbias.coverage import TARGET_CODES, prepare_generation_cases
from fairforget.warbias.evidence import source_records
from fairforget.warbias.ukrainian_request import RESPONSE_FIELDS, request_state

ROLES = {"S": "stereotype", "C": "counter_stereotype", "U": "unrelated"}


def sentences(row):
    return {role: row[field] for role, field in ROLES.items()}


def parse_csv(path, delimiter=","):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def repair_history(identities, originals, review_inputs, annotations, decisions):
    history = []
    for identity in identities:
        record = originals[identity]
        prior = sentences(record)
        text_hash = digest(prior)
        review_input = review_inputs[identity]
        review = annotations[identity]
        if (
            digest(review_input["context"]["sentences"]) != review_input["text_sha256"]
            or review_input["text_sha256"] != review["text_sha256"]
            or digest(review_input["context"]) != review["context_sha256"]
            or review["context_sha256"] != review_input["context_sha256"]
        ):
            raise ValueError(f"Review hashes do not match for {identity}")
        evidence = {
            "source_id": identity,
            "sentences": prior,
            "text_sha256": text_hash,
            "original_row_sha256": digest(record),
        }
        if identity.startswith("WB-SYN-") and review["text_sha256"] != text_hash:
            raise ValueError(f"Archived synthetic text differs from its review for {identity}")
        if review["text_sha256"] == text_hash and review["has_problem"]:
            evidence["sentence_review"] = review["sentences"]
            evidence["review_sha256"] = digest(review)
        if record.get("mechanical_issues"):
            evidence["mechanical_issues"] = record["mechanical_issues"]
        if review["text_sha256"] != text_hash:
            evidence["superseded_text_review_sha256"] = digest(review)
        decision = decisions.get(identity)
        if decision and decision["decision"] == "repair":
            long_names = {ROLES[role]: prior[role] for role in ROLES}
            if decision["uk_text_sha256"] != digest(long_names):
                raise ValueError(f"Source text differs from comparison review for {identity}")
            evidence["comparison_review"] = decision["reason"]
            evidence["comparison_review_sha256"] = digest(decision)
        if not (evidence.get("sentence_review") or evidence.get("comparison_review")):
            raise ValueError(f"Missing applicable repair reasons for {identity}")
        history.append(evidence)
    if not history:
        raise ValueError("A repair slot requires at least one prior source")
    return history


def build(workspace, project, output):
    previous = workspace / "outputs"
    plan = previous / "taxonomy_metric_completion_20261120"
    expanded = previous / "all_dimension_balance_20261121"
    archived_reviews = previous / "warbias_gpt_review_20261101/full"
    registry = project / "protocol/inputs"
    bindings = {}

    def bound(path):
        path = path.resolve()
        label = (
            "producer/" + str(path.relative_to(project))
            if path.is_relative_to(project)
            else "workspace/" + str(path.relative_to(workspace))
        )
        bindings[label] = file_digest(path)
        return path

    slots = unique_index(
        read_rows(bound(plan / "triplet_slots.jsonl"))
        + read_rows(bound(expanded / "actor_additional_triplets.jsonl")),
        "slot_id",
    )
    split = unique_index(parse_csv(bound(expanded / "split_components.tsv"), "\t"), "component")
    taxonomy = unique_index(
        parse_csv(bound(previous / "metric_data_gap_audit_20261120/inputs/current_taxonomy.csv")),
        "public_case_id",
    )
    source_cases = unique_index(
        read_rows(bound(registry / "construction_cases.jsonl"))
        + read_rows(bound(registry / "coverage_case_additions.jsonl")),
        "case_id",
    )
    instructions = unique_index(
        read_rows(bound(registry / "case_instructions_uk.jsonl")), "case_id"
    )
    cross_guides = unique_index(
        read_rows(bound(registry / "actor_counterpart_instructions_uk.jsonl")), "public_source_case"
    )
    eligible = {
        r["canonical_source_case"]
        for r in json.loads(bound(plan / "actor_applicability.json").read_text())
        if r["eligible_for_actor_swap"]
    }
    if set(cross_guides) != eligible:
        raise ValueError("Actor guidance does not cover the complete canonical proposition set")
    occupation = {
        (r["case_id"], r["axis"]): r["occupation_reference"]
        for r in json.loads(bound(plan / "axis_blocks.json").read_text())
        if r["axis"] == "occupation"
    }
    declared = unique_index(json.loads(bound(plan / "case_matrix.json").read_text()), "case_id")
    original = unique_index(
        read_rows(
            bound(previous / "warbias_release_cleanup_20261102/verified_export/data/uk.jsonl")
        )
        + read_rows(
            bound(
                previous / "warbias_coverage_generation_20261030/hybrid_20261031/api/triplets.jsonl"
            )
        )
        + read_rows(bound(output / "sources/jupiter-triplets.jsonl")),
        "id",
    )
    review_inputs = unique_index(read_rows(bound(archived_reviews / "review-inputs.jsonl")), "id")
    annotations = unique_index(read_rows(bound(archived_reviews / "annotations.jsonl")), "id")
    decisions = unique_index(
        read_rows(
            bound(previous / "shared_claim_matrix_20261120/candidate_review_decisions.jsonl")
        ),
        "id",
    )
    data = load_inputs(project)
    sources = source_records(
        data["sources"], data["source_feedback"], data.get("source_compilation", ())
    )
    definitions, items, lineage = {}, [], {}
    for slot_id, slot in sorted(slots.items()):
        if not slot["action"].startswith("lapa_"):
            continue
        actor, action = slot["actor"], slot["action"]
        public_case = (
            slot.get("source_case") or slot.get("native_case") or slot["common_proposition_source"]
        )
        source = taxonomy[public_case]
        internal = source["case_id"]
        base = source_cases[internal]
        alternate_actor = action == "lapa_actor_counterpart"
        if alternate_actor != (source["target_group"] != actor):
            raise ValueError(f"Actor and counterpart flag disagree at {slot_id}")
        if (
            slot["claim_uk"] != source["claim_uk"]
            or slot["component"] != declared[public_case]["component"]
        ):
            raise ValueError(f"Taxonomy claim or component mismatch at {slot_id}")
        component = split[slot["component"]]
        if public_case not in component["source_cases"].split("|"):
            raise ValueError(f"Case outside its split at {slot_id}")
        # Counterparts and standardised 'community member' framings keep the
        # original semantics, each with the reporter pair for this comparison.
        standard_reporters = slot["reporter"] != "another veteran" and (
            alternate_actor
            or "matched_actor" in slot.get("requirements", [])
            or slot.get("common_proposition_source") is not None
        )
        case_key = internal + ("__" + actor + "_counterpart" if alternate_actor else "")
        if standard_reporters and "another veteran" in base["reporters"]:
            case_key += "__shared_reporter"
        reporters = [
            "community member" if standard_reporters and role == "another veteran" else role
            for role in base["reporters"]
        ]
        guidance = cross_guides[public_case] if alternate_actor else None
        case = {
            **base,
            "case_id": case_key,
            "target_group": actor,
            "target_subgroups": [TARGET_CODES[actor]],
            "source_claim_uk": slot["claim_uk"],
            "context": guidance["context_uk"] if guidance else source["context_uk"],
            "reporters": reporters,
            "intrinsic_attributes": declared[public_case]["intrinsic_attributes"],
        }
        if definitions.setdefault(case_key, case) != case:
            raise ValueError(f"Conflicting definitions for {case_key}")
        reference = (
            occupation.get((public_case, "occupation"))
            if "occupation" in slot["attributes"]
            else None
        )
        if slot.get("occupation_reference") and slot["occupation_reference"] != reference:
            raise ValueError(f"Occupation time reference changed at {slot_id}")
        items.append(
            {
                "draft_item_id": slot_id,
                "case_id": case_key,
                "target_group": actor,
                "target_code": TARGET_CODES[actor],
                "attributes": slot["attributes"],
                "occupation_reference": reference,
                "reporter": None if slot["reporter"] == "unattributed" else slot["reporter"],
            }
        )
        lineage[slot_id] = {
            "public_source_case": public_case,
            "production_case": slot["case_id"],
            "instruction_source_case_id": internal,
            "source_definition_sha256": digest(base),
            "actor": actor,
            "action": action,
            "component": slot["component"],
            "split": component["split"],
            "generation_slot_sha256": digest(slot),
        }
        if guidance:
            lineage[slot_id]["counterpart_guidance"] = guidance
        if action == "lapa_repair":
            lineage[slot_id]["repair_sources"] = repair_history(
                slot["candidate_ids"], original, review_inputs, annotations, decisions
            )
    matrix = prepare_generation_cases(items, definitions.values(), sources)
    for row in matrix:
        row.update(lineage[row["id"]])
        instruction = instructions[row["instruction_source_case_id"]]
        for stage in ROLES:
            # Empty preview parents only validate role, wording and all instruction
            # bindings here. Live C/U use the actual text collected by the runner.
            request_state(
                row,
                stage,
                {},
                response_field=RESPONSE_FIELDS[stage],
                task_uk="Перевірка",
                instructions=instruction,
            )
    if {r["matrix_item_id"] for r in matrix} != set(lineage):
        raise ValueError("Incomplete or duplicate generation matrix")
    totals = Counter(r["action"] for r in matrix)
    if totals != {"lapa_repair": 3567, "lapa_missing": 3086, "lapa_actor_counterpart": 8268}:
        raise ValueError(f"Iteration allocation changed: {totals}")
    path = output / "matrix.jsonl"
    tmp_path = output / "matrix.pending.jsonl"
    write_rows(tmp_path, matrix)
    tmp_path.replace(path)
    write_json(
        output / "matrix-verification.json",
        {
            "rows": len(matrix),
            "by_action": dict(totals),
            "by_split": dict(Counter(r["split"] for r in matrix)),
            "by_actor": dict(Counter(r["actor"] for r in matrix)),
            "repair_sources": sum(len(r.get("repair_sources", [])) for r in matrix),
            "sha256": file_digest(path),
            "source_files_sha256": bindings,
            "generation_executed": False,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    build(args.workspace.resolve(), args.project.resolve(), args.output.resolve())
