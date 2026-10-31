"""Measure the original training panel using fixed scale-one checkpoints."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue

import training_diagnostic as diagnostic

supplement = diagnostic.supplement
native = supplement.native
read, rows, require, evidence = (
    diagnostic.read,
    diagnostic.rows,
    diagnostic.require,
    diagnostic.evidence,
)
TASK = diagnostic.TASK


def context_for(release, application, submission, alias):
    plan = read(release / "plan.json")
    spec = plan["models"][alias]
    addon = Path(spec["addon"])
    config, collector, _, _ = supplement.validate(addon)
    profile = spec["model_profile"]
    models = supplement.checkpoint_panel(
        collector, application / alias, methods=["task_vector"], model_profile=profile
    )
    receipt_path = application / alias / "task_vector/intervention.json"
    receipt = read(receipt_path)
    require(
        receipt["fixed_scale"] == 1.0 and receipt["scale_search_performed"] is False,
        "Fixed scale one required",
    )
    require(
        receipt["merged_weight_change"]["changed_elements"] > 0, "Actual changed weights required"
    )
    recipe_path = release / spec["recipe"]
    require(receipt["config_sha256"] == native.digest(recipe_path), "Applied recipe differs")
    recipe = read(recipe_path)
    forget = diagnostic.bound(release, recipe["streams"]["forget"], recipe_path.parent)
    forget_ids = {row["source_item_id"] for row in rows(forget)}
    parent = Path(plan["parent_release"])
    manifest = parent / "diagnostics/manifest.json"
    bundle = read(manifest)
    metadata = bundle["training_diagnostic"]
    require(
        metadata.get("kind") == diagnostic.KIND
        and metadata.get("source_repository") == "FairForget/WarBias"
        and metadata.get("source_revision") == diagnostic.PIN,
        "Diagnostic source identity differs",
    )
    original = diagnostic.bound(
        parent, bundle["training_diagnostic"]["source_adapted"], manifest.parent
    )
    require(native.digest(original) == diagnostic.SOURCE_SHA, "Published source changed")
    entry, data, source = supplement.source(manifest, TASK)
    require(
        entry.get("source_role") == "training_signal"
        and entry.get("split") == "train"
        and entry.get("repo") == "FairForget/WarBias"
        and entry.get("revision") == diagnostic.PIN
        and entry.get("measurement_purpose") == diagnostic.KIND,
        "Training role required",
    )
    require(all(row.get("split") == "train" for row in data), "Training rows were relabeled")
    require(
        {row["id"] for row in data} == forget_ids, "Diagnostic and actual acquisition IDs differ"
    )
    originals = {row["id"]: row for row in rows(original)}
    require(all(originals[row["id"]] == row for row in data), "Diagnostic rows changed")
    pairing = entry["pairing"]
    mapping_path = supplement.inside(manifest.parent, pairing["filename"])
    require(native.digest(mapping_path) == pairing["sha256"], "Diagnostic pair map hash differs")
    mapping = rows(mapping_path)
    require(
        pairing.get("uk_task") == TASK
        and pairing.get("en_task") == "warbias_triplets_en"
        and pairing.get("matched_rows") == len(data) == len(mapping)
        and pairing.get("excluded_uk_rows") == 0
        and len({item["uk_id"] for item in mapping}) == len(mapping)
        and {item["uk_id"] for item in mapping} == forget_ids
        and all(row.get("paired_eligible") is True for row in data),
        "Diagnostic source pair map differs from actual acquisition",
    )
    collection = Path(submission["models"][alias]["collection_output"])
    collected = read(collection / "supplemental_execution.json")
    require(
        collected["stage"] == "supplemental_collection_complete"
        and collected["no_update_passed"] is True,
        "Verified native runtime qualification required",
    )
    require(collected["addon_config"] == evidence(addon / "config.json"), "Qualified addon changed")
    require(
        collected["checkpoints"] == {k: str(v) for k, v in models.items()},
        "Qualified checkpoint panel differs",
    )
    for name in models:
        event = collected["qualification"][name + "/" + TASK]
        require(
            event["passed"] is True and len(event["runs"]) == 2,
            "Two unchanged qualification repeats required",
        )
        for run in event["runs"]:
            require(
                evidence(Path(run["predictions"]["path"])) == run["predictions"],
                "Qualification predictions changed",
            )
    return {
        "collector": collector,
        "model_profile": profile,
        "models": models,
        "manifest": manifest,
        "entry": entry,
        "rows": data,
        "baseline_files": {
            item["path"]: item["sha256"]
            for item in receipt["source_model_files"]
            if ".cache" not in Path(item["path"]).parts
        },
        "checkpoint_files": {
            "task_vector": {
                item["path"]: item["sha256"] for item in receipt["merged_checkpoint_files"]
            }
        },
        "collector_code_identity": {
            k.removeprefix("evaluator/lm_eval/"): v
            for k, v in config["evaluator_files_sha256"].items()
        },
        "bindings": {
            "receipt": evidence(receipt_path),
            "recipe": evidence(recipe_path),
            "training_rows": evidence(source),
            "manifest": evidence(manifest),
            "runtime_qualification": evidence(collection / "supplemental_execution.json"),
        },
    }


def run(release, application, submission_path, output):
    submission = read(submission_path)
    require(not output.exists(), "Preserve existing diagnostics")
    contexts = {
        alias: context_for(release, application, submission, alias)
        for alias in submission["models"]
    }
    output.mkdir(parents=True)
    tokens = Queue()
    for gpu in os.environ["CUDA_VISIBLE_DEVICES"].split(","):
        tokens.put(gpu)
    state = {
        "kind": "training_source_diagnostic_not_heldout",
        "fixed_scale": 1.0,
        "stage": "collecting",
        "bindings": {alias: c["bindings"] for alias, c in contexts.items()},
        "tasks": {},
        "pairs": {},
    }
    status = output / "execution.json"
    native.write_json(status, state)

    def collect(alias, name):
        gpu = tokens.get()
        try:
            c = contexts[alias]
            return diagnostic.collect_one(
                c,
                name,
                c["manifest"],
                c["entry"],
                c["rows"],
                output / alias / "native" / name / TASK,
                gpu,
            )
        finally:
            tokens.put(gpu)

    with ThreadPoolExecutor(max_workers=tokens.qsize()) as pool:
        futures = {
            pool.submit(collect, alias, name): alias + "/" + name
            for alias, c in contexts.items()
            for name in c["models"]
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                state["tasks"][key] = future.result()
            except Exception as error:
                state["tasks"][key] = {"status": "failed", "error": str(error)}
            native.write_json(status, state)
    if not all(v["status"] == "completed" for v in state["tasks"].values()):
        state["stage"] = "collection_failed"
        native.write_json(status, state)
        return 1
    for alias, c in contexts.items():
        target = output / alias / "paired/task_vector"
        command = [
            sys.executable,
            "-m",
            "lm_eval.fair_uk",
            "paper",
            "compare-native",
            "--before-run",
            str(output / alias / "native/baseline" / TASK),
            "--after-run",
            str(output / alias / "native/task_vector" / TASK),
            "--dataset-bundle",
            str(c["manifest"]),
            "--bootstrap",
            "1000",
            "--seed",
            "42",
            "--output",
            str(target),
        ]
        with (output / alias / "paired.log").open("x") as log:
            result = subprocess.run(
                command,
                cwd=c["collector"] / "evaluator",
                env={
                    **os.environ,
                    "CUDA_VISIBLE_DEVICES": "",
                    "PYTHONPATH": str(c["collector"] / "evaluator"),
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                },
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        require(result.returncode == 0, "Native training-source pairing failed: " + alias)
        state["pairs"][alias] = evidence(target / "paired.json")
        native.write_json(status, state)
    state["stage"] = "training_source_pairs_complete"
    native.write_json(status, state)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("release", "application", "submission", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    return run(
        args.release.resolve(),
        args.application.resolve(),
        args.submission.resolve(),
        args.output.resolve(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
