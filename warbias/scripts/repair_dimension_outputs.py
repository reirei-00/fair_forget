"""Prepare and recheck versioned corrections for confirmed first-pass output defects."""

import argparse
import fcntl
import os
import tempfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

from fairforget.cli_support import load_json
from fairforget.io import digest, file_digest, read_rows, unique_index, write_json, write_rows
from fairforget.warbias.output_defects import defects, repair_context, triplet_text

GENERATION_KEYS = (
    "strategy",
    "candidate_id",
    "provider",
    "model",
    "revision",
    "settings",
    "execution_config_sha256",
    "source_protocol_sha256",
    "prompt_sha256",
)


@contextmanager
def lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".repair-selection.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def frozen(path, value, *, lines=False):
    # A crash may leave some complete outputs. The same inputs resume safely.
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        temp = Path(tmp.name)
    try:
        (write_rows if lines else write_json)(temp, value)
        if path.exists():
            if file_digest(path) != file_digest(temp):
                raise ValueError(f"Existing version has different evidence: {path.name}")
        else:
            temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
    return file_digest(path)


def proof(path, directory):
    return {"path": os.path.relpath(path, directory), "sha256": file_digest(path)}


def bound(directory, descriptor):
    path = directory / descriptor["path"]
    if file_digest(path) != descriptor["sha256"]:
        raise ValueError(f"Input changed: {path.name}")
    return path


def model_binding(row):
    generation = row["generation"]
    identity = {key: generation[key] for key in GENERATION_KEYS}
    if generation["strategy"] != "sequential" or generation["input_id"] != row["id"]:
        raise ValueError("Generated output requires its actual sequential condition and unit ID")
    return identity


def verified_replacement(original, corrective, expected_case):
    if corrective["id"] != original["id"] or corrective["case"] != expected_case:
        raise ValueError("Corrective output does not bind its protected matrix row")
    if model_binding(corrective) != model_binding(original):
        raise ValueError("Corrective output changed the model, tokenizer, prompts or configuration")
    return defects(triplet_text(corrective))


def read_reuse(directory):
    if directory is None:
        return {}
    report = load_json(directory / "review.json")
    path = directory / "replacements.jsonl"
    if file_digest(path) != report["replacements_sha256"]:
        raise ValueError("Pilot corrective output differs from its verification")
    return unique_index(read_rows(path), "id")


def prepare(matrix_path, source_path, directory, previous=None):
    base = unique_index(read_rows(matrix_path), "matrix_item_id")
    observed = unique_index(read_rows(source_path), "id")
    if base.keys() != observed.keys() or not base:
        raise ValueError("First-pass rows must cover the exact input matrix")
    pilot = read_reuse(previous)
    queued, reused, issue_counts = [], [], Counter()
    for item, row in base.items():
        obs = observed[item]
        context = repair_context(row, obs)
        if not context["diagnostics"]:
            continue
        model_binding(obs)
        issue_counts.update(i["code"] for i in context["diagnostics"])
        desired = {**row, "output_defect_repair": context}
        earlier = pilot.get(item)
        if earlier and earlier["case"] == desired:
            if verified_replacement(obs, earlier, desired):
                raise ValueError("Stored repair still contains a confirmed defect")
            reused.append(earlier)
        else:
            queued.append(desired)
    with lock(directory):
        manifest = {
            "policy": "exact_duplicate_standalone_copied_parent_lapa_control_tokens_v1",
            "matrix": proof(matrix_path, directory),
            "observed": proof(source_path, directory),
            "source_count": len(base),
            "flag_count": len(queued) + len(reused),
            "queue_count": len(queued),
            "reused_count": len(reused),
            "issues": dict(issue_counts),
            "queue_sha256": frozen(directory / "queue.jsonl", queued, lines=True),
            "reuse_sha256": frozen(directory / "reuse.jsonl", reused, lines=True),
            "structural_checks_only": True,
        }
        if previous:
            manifest["prior_verification"] = proof(previous / "review.json", directory)
        frozen(directory / "selection.json", manifest)
    return manifest


def check(directory, results_path):
    with lock(directory):
        plan = load_json(directory / "selection.json")
        source = unique_index(read_rows(bound(directory, plan["observed"])), "id")
        matrix = unique_index(read_rows(bound(directory, plan["matrix"])), "matrix_item_id")
        queue = unique_index(
            read_rows(bound(directory, {"path": "queue.jsonl", "sha256": plan["queue_sha256"]})),
            "matrix_item_id",
        )
        reuse = unique_index(
            read_rows(bound(directory, {"path": "reuse.jsonl", "sha256": plan["reuse_sha256"]})),
            "id",
        )
        corrected = unique_index(read_rows(results_path), "id") if results_path else {}
        if queue.keys() != corrected.keys() or queue.keys() & reuse.keys():
            raise ValueError("Missing, overlapping or unexpected corrective output IDs")
        if plan.get("prior_verification"):
            bound(directory, plan["prior_verification"])
        good, residual = [], []
        for item, obs in source.items():
            context = repair_context(matrix[item], obs)
            if not context["diagnostics"]:
                if item in queue or item in reuse:
                    raise ValueError("A first-pass item without a confirmed issue was queued")
                continue
            expected = {**matrix[item], "output_defect_repair": context}
            if item in queue and queue[item] != expected:
                raise ValueError("Repair queue lost source text, original guidance or defects")
            replacement = corrected[item] if item in corrected else reuse[item]
            remaining = verified_replacement(obs, replacement, expected)
            if remaining:
                residual.append(
                    {
                        "id": item,
                        "first_pass_sha256": digest(obs),
                        "source_output_sha256": digest(replacement),
                        "sentences": triplet_text(replacement),
                        "diagnostics": remaining,
                        "diagnostics_sha256": digest(remaining),
                    }
                )
            else:
                good.append(replacement)
        if len(residual) + len(good) != plan["flag_count"]:
            raise ValueError("Repair coverage differs from first-pass flags")
        manifest = {
            "selection_sha256": file_digest(directory / "selection.json"),
            "source_count": len(source),
            "rechecked_count": len(good) + len(residual),
            "successful_replacements": len(good),
            "residual_count": len(residual),
            "structural_checks_only": True,
            "replacements_sha256": frozen(directory / "replacements.jsonl", good, lines=True),
            "residuals_sha256": frozen(directory / "residuals.jsonl", residual, lines=True),
        }
        if results_path:
            manifest["regenerated"] = proof(results_path, directory)
        frozen(directory / "review.json", manifest)
        return manifest


if __name__ == "__main__":
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    select = modes.add_parser("prepare")
    for name in ("matrix", "source", "output", "reuse"):
        select.add_argument("--" + name, type=Path, required=name != "reuse")
    review = modes.add_parser("check")
    review.add_argument("--version", type=Path, required=True)
    review.add_argument("--results", type=Path)
    args = parser.parse_args()
    report = (
        prepare(args.matrix, args.source, args.output, args.reuse)
        if args.mode == "prepare"
        else check(args.version, args.results)
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
