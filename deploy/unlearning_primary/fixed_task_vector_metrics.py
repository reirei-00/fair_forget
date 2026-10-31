"""Analyze fixed scale-one saved pairs with the maintained Fair-UK CLI."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import run_warbias_metrics as metrics

native = metrics.native
supplement = metrics.supplemental
require, read, evidence = metrics.require, metrics.read, metrics.evidence


def run(release, application, submission_path, alias, analysis, output):
    require(not output.exists(), "Preserve existing analysis")
    inputs = {}
    hashes = metrics.checked_analysis(analysis, inputs)
    plan = read(metrics.snapshot(release / "plan.json", inputs))
    require(plan["scale"] == 1.0 and plan["scale_search"] is False, "Fixed scale one required")
    spec = plan["models"][alias]
    addon = Path(spec["addon"])
    config, collector, full, small = supplement.validate(addon)
    require(config["methods"] == ["task_vector"], "One selected method required")
    require(config["model_profile"] == spec["model_profile"], "Model profiles differ")
    training = application / alias
    checkpoints = supplement.checkpoint_panel(
        collector, training, methods=["task_vector"], model_profile=spec["model_profile"]
    )
    execution = training / "execution.json"
    receipt_path = training / "task_vector/intervention.json"
    receipt = read(metrics.snapshot(receipt_path, inputs))
    require(
        receipt["fixed_scale"] == 1.0
        and receipt["scale_search_performed"] is False
        and receipt["null_intervention_selected"] is False
        and receipt["merged_weight_change"]["changed_elements"] > 0,
        "A real fixed scale-one intervention is required",
    )
    recipe = release / spec["recipe"]
    require(receipt["config_sha256"] == native.digest(recipe), "Applied recipe differs")
    recipe_data = read(metrics.snapshot(recipe, inputs))
    binding = metrics.inside(release, str(recipe.parent / recipe_data["data_manifest"]))
    require(native.digest(binding) == recipe_data["data_manifest_sha256"], "Data binding differs")
    binding_data = read(metrics.snapshot(binding, inputs))
    review = binding_data["deployment_binding"]
    manifest = metrics.inside(release, str(binding.parent / review["source_review_manifest"]))
    metrics.checked(manifest, review["source_review_manifest_sha256"], inputs)
    targets = metrics.declared_targets(read(manifest))
    submission = read(metrics.snapshot(submission_path, inputs))
    collection = Path(submission["models"][alias]["collection_output"])
    record = read(metrics.snapshot(collection / "supplemental_execution.json", inputs))
    require(
        record["stage"] == "supplemental_collection_complete"
        and record["no_update_passed"] is True
        and record["training"] == evidence(execution)
        and record["addon_config"] == evidence(addon / "config.json")
        and record["checkpoints"] == {name: str(path) for name, path in checkpoints.items()}
        and record["expected_full_collections"] == 24,
        "All selected checkpoint collections must complete",
    )
    require(
        set(record["tasks"])
        == {name + "/" + task for name in checkpoints for task in supplement.PROTOCOLS},
        "Native task inventory differs",
    )
    require(
        set(record["qualification"])
        == {name + "/" + task for name in checkpoints for task in supplement.REPRESENTATIVES},
        "Runtime qualification inventory differs",
    )
    for key, item in record["qualification"].items():
        name, task = key.split("/")
        require(item["passed"] is True and len(item["runs"]) == 2, "Two unchanged repeats required")
        roots = []
        for number, event in enumerate(item["runs"], 1):
            root = collection / "qualification" / name / task / ("repeat-" + str(number)) / task
            require(
                event["returncode"] == 0
                and event["predictions"] == evidence(root / "predictions.jsonl"),
                "Qualification files differ",
            )
            roots.append(root)
        gate = supplement.compare_repeats(
            *roots,
            config["qualification"]["selection"][task]["ordered_ids"],
            supplement.SENTENCE if "triplets" in task else supplement.PROTOCOLS[task],
        )
        require(gate["passed"] is True, "Runtime repeat gate failed")
    for key, event in record["tasks"].items():
        require(
            event["status"] == "collected" and event["returncode"] == 0,
            "Native collection failed: " + key,
        )
        path = Path(event["receipt"]["path"])
        require(evidence(path) == event["receipt"], "Collection receipt changed")
        status = read(path)
        require(
            status["status"] == "completed"
            and status["exit_code"] == 0
            and status["full_supplemental_evaluation_bundle"] is True
            and status["qualification_only"] is False
            and status["manifest_sha256"] == native.digest(full),
            "Full evaluation receipt required",
        )
    output.mkdir(parents=True)
    archive = output / "training_evidence.json"
    files = {
        "task_vector/" + name: native.digest(training / "task_vector" / name)
        for name in ("intervention.json", "losses.jsonl", "exposure.jsonl")
    }
    native.write_json(
        archive,
        {
            "kind": "saved_acquisition_fixed_scale_one_evidence",
            "files": files,
            "new_optimizer_steps": 0,
        },
    )
    config_path = output / "config.json"
    selected = metrics.make_config(
        "task_vector", alias, collection, collector, training, recipe, manifest, binding, archive
    )
    selected["intervention"]["id"] = "warbias_fixed_scale_one_" + alias
    native.write_json(config_path, selected)
    destination = output / "task_vector"
    command = [
        sys.executable,
        "-m",
        "lm_eval.fair_uk",
        "paper",
        "all-metrics",
        "--config",
        str(config_path),
        "--output",
        str(destination),
    ]
    with (output / "analysis.log").open("x") as log:
        completed = subprocess.run(
            command,
            cwd=analysis,
            env={**os.environ, "PYTHONPATH": str(analysis), "CUDA_VISIBLE_DEVICES": ""},
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    require(completed.returncode == 0, "Saved-pair analysis failed; inspect analysis.log")
    report_path = destination / "all_metrics.json"
    report = read(report_path)
    metrics.verify_report(
        report, "task_vector", recipe, receipt_path, destination, hashes, target_values=targets
    )
    native.write_json(
        output / "execution.json",
        {
            "kind": "fixed_scale_one_saved_analysis",
            "stage": "analysis_complete",
            "model": alias,
            "fixed_scale": 1.0,
            "scale_search": False,
            "model_requests": 0,
            "application": evidence(execution),
            "report": evidence(report_path),
            "native_collection": evidence(collection / "supplemental_execution.json"),
            "inputs": inputs,
            "command": command,
        },
    )
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("release", "application", "submission", "analysis", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    return run(
        args.release.resolve(),
        args.application.resolve(),
        args.submission.resolve(),
        args.model,
        args.analysis.resolve(),
        args.output.resolve(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
