"""Collect native training-source likelihood diagnostics from verified checkpoints.

The unchanged Fair-UK collector and paired scorer own every calculation. These
descriptive measurements never select a task-vector scale or replace held-out
evaluation.

Run this code from a separate runner release. The --training-release argument
supplies the frozen matrix and diagnostic data; --training selects actual saved
training outputs. The scheduler wrapper takes RUNNER_RELEASE, TRAINING_RELEASE,
MODEL_ID, TRAINING, ADDON and RESULT_PARENT in that order.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import subprocess
import sys
import threading
from pathlib import Path

import supplemental as supplement

native = supplement.native
METHODS = ("rmu", "task_vector", "wga_kl")
TASK = "warbias_triplets_uk"
PIN = "164f355623c2da0028821e0942cad493558a6c13"
SOURCE_SHA = "96b87403880da7a698e92e733bbf456f29698ff346ae032f4779c434a5659fe5"
KIND = "training_source_diagnostic_not_heldout"
require, read, rows, evidence = (
    supplement.require,
    supplement.read,
    supplement.rows,
    supplement.evidence,
)


def bound(root, reference, base=None):
    """Resolve a checksum-bound ordinary input inside its immutable release."""
    path = ((base or root) / reference["path"]).resolve()
    require(path.is_relative_to(root), "Bound input escapes the training release")
    require(path.is_file() and not path.is_symlink(), "Bound input is missing")
    require(native.digest(path) == reference["sha256"], "Bound input hash differs")
    return path


def inputs(release, model_id, training, addon):
    config, collector, _, small = supplement.validate(addon)
    matrix_path = release / "matrix.json"
    matrix = read(matrix_path)
    require(matrix.get("methods") == list(METHODS), "Three requested method arms required")
    entry = matrix["models"][model_id]
    profile = {key: entry[key] for key in supplement.MODEL_PROFILE_FIELDS}
    require(config.get("model_profile") == profile, "Addon and training model profiles differ")
    models = supplement.checkpoint_panel(
        collector,
        training,
        methods=METHODS,
        model_profile=profile,
        inference_views=config.get("inference_views"),
    )
    execution = read(training / "execution.json")
    require(
        execution.get("matrix_sha256") == native.digest(matrix_path)
        and execution.get("model_id") == model_id
        and Path(execution.get("release", "")).resolve() == release,
        "Actual training execution does not bind this matrix and model",
    )
    forget_ids, binding_path, recipe_records, checkpoint_files = None, None, {}, {}
    baseline_files = None
    for method in METHODS:
        recipe_path = bound(release, entry["recipes"][method])
        require(execution["recipes"][method] == evidence(recipe_path), "Applied recipe differs")
        recipe = read(recipe_path)
        require(recipe.get("method") == method, "Recipe method differs")
        require(
            all(
                recipe.get(key) == profile[key]
                for key in ("model_id", "model_revision", "tokenizer_revision", "model_path")
            ),
            "Recipe model identity differs",
        )
        path = bound(
            release,
            {"path": recipe["data_manifest"], "sha256": recipe["data_manifest_sha256"]},
            recipe_path.parent,
        )
        binding = read(path)
        require(binding.get("reviewed_for_gradients") is True, "Gradient review binding required")
        require(binding_path is None or binding_path == path, "Methods use different data bindings")
        binding_path = path
        stream = bound(release, recipe["streams"]["forget"], recipe_path.parent)
        data = rows(stream)
        ids = [row["source_item_id"] for row in data]
        require(bool(ids) and len(set(ids)) == len(ids), "Unique forget source IDs required")
        require(
            set(ids) <= set(binding["permitted_gradient_ids"]), "Forget IDs lack gradient approval"
        )
        require(forget_ids is None or forget_ids == set(ids), "Method forget cohorts differ")
        forget_ids = set(ids)
        record = execution["methods"][method]["receipt"]
        actual = read(Path(record["path"] if isinstance(record, dict) else record))
        provenance = actual.get("provenance", {}) if method == "rmu" else actual
        require(
            provenance.get("config_sha256") == native.digest(recipe_path),
            "Checkpoint used another recipe",
        )
        files = actual["checkpoint_identity" if method == "rmu" else "merged_checkpoint_files"]
        checkpoint_files[method] = {item["path"]: item["sha256"] for item in files}
        if method != "rmu":
            actual_base = {
                item["path"]: item["sha256"]
                for item in actual["source_model_files"]
                if ".cache" not in Path(item["path"]).parts
            }
            require(
                actual_base.get("config.json") == profile["config_sha256"]
                and any(name.endswith(".safetensors") for name in actual_base),
                "Actual trainer base model hashes are incomplete",
            )
            require(
                baseline_files is None or baseline_files == actual_base,
                "Actual trainer source checkpoint hashes differ",
            )
            baseline_files = actual_base
        recipe_records[method] = evidence(recipe_path)
    manifest = release / "diagnostics/manifest.json"
    bundle = read(manifest)
    diagnostic = bundle["training_diagnostic"]
    require(
        diagnostic.get("kind") == KIND
        and diagnostic.get("source_repository") == "FairForget/WarBias"
        and diagnostic.get("source_revision") == PIN,
        "Diagnostic source identity differs",
    )
    source = bound(release, diagnostic["source_adapted"], manifest.parent)
    require(native.digest(source) == SOURCE_SHA, "Original published adaptation hash differs")
    specification, data, adapted = supplement.source(manifest, TASK)
    require(
        specification.get("source_role") == "training_signal"
        and specification.get("split") == "train"
        and specification.get("repo") == "FairForget/WarBias"
        and specification.get("revision") == PIN
        and specification.get("measurement_purpose") == KIND,
        "Diagnostic must retain its explicit training-source role",
    )
    require(all(row.get("split") == "train" for row in data), "Training rows were relabeled")
    require(
        {row["id"] for row in data} == forget_ids, "Diagnostic IDs differ from final forget stream"
    )
    original = {row["id"]: row for row in rows(source)}
    require(
        all(original.get(row["id"]) == row for row in data), "Diagnostic source rows were changed"
    )
    require(
        any(
            item.get("repo_id") == "FairForget/WarBias"
            and item.get("revision") == PIN
            and item.get("published") is True
            for item in specification.get("source_provenance", [])
        ),
        "Published source provenance required",
    )
    pairing = specification["pairing"]
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
        "Diagnostic source pair map differs from final forget cohort",
    )
    small_entry, small_data, _ = supplement.source(small, TASK)
    require(
        len(small_data) == 2 and all(row.get("split") == "development" for row in small_data),
        "Frozen addon two-row development qualification required",
    )
    return {
        "collector": collector,
        "inference_views": config.get("inference_views"),
        "model_profile": profile,
        "models": models,
        "manifest": manifest,
        "entry": specification,
        "rows": data,
        "small_manifest": small,
        "small_entry": small_entry,
        "small_rows": small_data,
        "checkpoint_files": checkpoint_files,
        "baseline_files": baseline_files,
        "bindings": {
            "matrix": evidence(matrix_path),
            "data_manifest": evidence(binding_path),
            "recipes": recipe_records,
            "source_adapted": evidence(source),
            "diagnostic_manifest": evidence(manifest),
            "diagnostic_rows": evidence(adapted),
            "addon_config": evidence(addon / "config.json"),
            "training_execution": evidence(training / "execution.json"),
            **(
                {"inference_views": config["inference_views"]}
                if config.get("inference_views")
                else {}
            ),
        },
        "collector_code_identity": {
            key.removeprefix("evaluator/lm_eval/"): value
            for key, value in config["evaluator_files_sha256"].items()
        },
    }


def collect_one(context, name, manifest, entry, data, output, gpu):
    checkpoint = context["models"][name]
    require(not output.exists(), "Native diagnostic output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = supplement.command_for(checkpoint, manifest, TASK, output)
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": gpu,
        "PYTHONPATH": str(context["collector"] / "evaluator"),
        **supplement.RUNTIME_ENVIRONMENT,
    }
    record = {
        "command": command,
        "gpu": gpu,
        "runtime_environment": supplement.RUNTIME_ENVIRONMENT,
        "output": str(output),
        "started_at_utc": native.utc_now(),
    }
    log = output.with_suffix(".log")
    try:
        with log.open("x") as handle:
            result = subprocess.run(
                command,
                cwd=context["collector"] / "evaluator",
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        record["returncode"] = result.returncode
        require(result.returncode == 0, "Native diagnostic collector failed")
        record.update(native.verify_complete(TASK, entry, output))
        supplement.verify_collected_options(output)
        run = read(output / "run.json")
        require(
            run.get("code_identity") == context["collector_code_identity"],
            "Collector source identity differs",
        )
        require(
            run.get("dataset") == {**entry, "bundle_manifest_sha256": native.digest(manifest)},
            "Native diagnostic source specification differs",
        )
        require(
            [row["id"] for row in rows(output / "predictions.jsonl")]
            == [row["id"] for row in data],
            "Native diagnostic IDs or order differ",
        )
        identity = run["model_identity"]
        require(
            all(
                type(identity["parameters"].get(key)) is type(native.SETTINGS[key])
                and identity["parameters"][key] == native.SETTINGS[key]
                for key in (
                    "dtype",
                    "tensor_parallel_size",
                    "max_model_len",
                    "seed",
                    "enable_thinking",
                    "enforce_eager",
                )
            ),
            "Native model runtime parameters differ",
        )
        require(
            Path(identity["parameters"]["pretrained"]).resolve() == checkpoint,
            "Native checkpoint path differs",
        )
        if name == "baseline":
            require(
                identity["local_files_sha256"] == context["baseline_files"],
                "Native baseline hashes differ from actual trainer source",
            )
        elif context.get("inference_views"):
            record["inference_view"] = supplement.view_for_native_checkpoint(
                context["collector"],
                context["inference_views"],
                context["model_profile"],
                checkpoint,
                files=identity["local_files_sha256"],
            )
        else:
            require(
                identity["local_files_sha256"] == context["checkpoint_files"][name],
                "Native checkpoint hashes differ from actual trainer",
            )
        record.update(
            status="completed",
            files={
                filename: evidence(output / filename)
                for filename in ("run.json", "report.json", "predictions.jsonl")
            },
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error))
    record["finished_at_utc"] = native.utc_now()
    if log.is_file():
        record["log"] = evidence(log)
    return record


def run(release, model_id, training, addon, output):
    release, training, addon, output = [
        path.resolve() for path in (release, training, addon, output)
    ]
    require(not output.exists(), "Use a new diagnostic output directory")
    context = inputs(release, model_id, training, addon)
    require(
        not any(
            output.is_relative_to(path)
            for path in (
                release,
                training,
                addon,
                context["collector"],
                *context["models"].values(),
            )
        ),
        "Diagnostic output overlaps immutable inputs",
    )
    gpus = [gpu.strip() for gpu in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")]
    require(
        len(gpus) == len(set(gpus)) == 4 and all(gpus), "Exactly four allocated GPU tokens required"
    )
    output.mkdir(parents=True)
    status = {
        "kind": KIND,
        "stage": "likelihood_qualification",
        "model_id": model_id,
        "bindings": context["bindings"],
        "checkpoints": {key: str(value) for key, value in context["models"].items()},
        "qualification": {},
        "tasks": {},
        "pairs": {},
        "no_update_passed": False,
        "qualification_scope": "two development triplets per checkpoint, likelihood only",
        "runtime_environment": supplement.RUNTIME_ENVIRONMENT,
        "task_vector_selection_input": False,
        "heldout_evaluation": False,
        "scientific_calibration_established": False,
        "baseline_lineage": "Full baseline hashes match both WGA and task-vector trainer source_model_files; RMU additionally binds the same matrix model path/pins and its exact saved output, without an independent recorded source-weight hash map.",
        "started_at_utc": native.utc_now(),
        "driver": evidence(Path(__file__)),
    }
    lock = threading.Lock()
    receipt = output / "training_diagnostic_execution.json"

    def save():
        native.write_json(receipt, status)

    def qualify(item):
        (name, _), gpu = item
        record = {"passed": False, "runs": []}
        try:
            paths = []
            for repeat in (1, 2):
                path = output / "qualification" / name / f"repeat-{repeat}" / TASK
                event = collect_one(
                    context,
                    name,
                    context["small_manifest"],
                    context["small_entry"],
                    context["small_rows"],
                    path,
                    gpu,
                )
                record["runs"].append(event)
                require(event["status"] == "completed", "Likelihood qualification collector failed")
                paths.append(path)
            record.update(
                supplement.compare_repeats(
                    *paths, [row["id"] for row in context["small_rows"]], supplement.SENTENCE
                )
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            record["error"] = str(error)
        with lock:
            status["qualification"][name] = record
            save()

    save()
    panel = list(zip(context["models"].items(), gpus))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(qualify, panel))
    status["no_update_passed"] = len(status["qualification"]) == 4 and all(
        item["passed"] for item in status["qualification"].values()
    )
    if not status["no_update_passed"]:
        status.update(
            stage="likelihood_qualification_failed_no_training_source_collection",
            finished_at_utc=native.utc_now(),
        )
        save()
        return 1
    status["stage"] = "training_source_collection"
    save()

    def full(item):
        (name, _), gpu = item
        event = collect_one(
            context,
            name,
            context["manifest"],
            context["entry"],
            context["rows"],
            output / "native" / name / TASK,
            gpu,
        )
        with lock:
            status["tasks"][name] = event
            save()

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(full, panel))
    if not all(item["status"] == "completed" for item in status["tasks"].values()):
        status.update(
            stage="training_source_collection_partial_failure", finished_at_utc=native.utc_now()
        )
        save()
        return 1
    status["stage"] = "saved_native_pairing"
    save()
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONPATH": str(context["collector"] / "evaluator"),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    for method in METHODS:
        target = output / "paired" / method
        command = [
            sys.executable,
            "-m",
            "lm_eval.fair_uk",
            "paper",
            "compare-native",
            "--before-run",
            str(output / "native/baseline" / TASK),
            "--after-run",
            str(output / "native" / method / TASK),
            "--dataset-bundle",
            str(context["manifest"]),
            "--bootstrap",
            "1000",
            "--seed",
            "42",
            "--output",
            str(target),
        ]
        record = {"command": command}
        try:
            with (output / f"paired-{method}.log").open("x") as handle:
                result = subprocess.run(
                    command,
                    cwd=context["collector"] / "evaluator",
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            record["returncode"] = result.returncode
            require(result.returncode == 0, "Saved native pairing failed")
            require(
                all(
                    native.digest(Path(item["path"])) == item["sha256"]
                    for task in status["tasks"].values()
                    for item in task["files"].values()
                ),
                "Native diagnostic inputs changed during pairing",
            )
            record.update(status="analyzed", report=evidence(target / "paired.json"))
        except (OSError, ValueError, KeyError, TypeError) as error:
            record.update(status="failed", error_type=type(error).__name__, error=str(error))
        status["pairs"][method] = record
        save()
    complete = all(item["status"] == "analyzed" for item in status["pairs"].values())
    status.update(
        stage="training_diagnostic_complete"
        if complete
        else "training_diagnostic_pairing_partial_failure",
        finished_at_utc=native.utc_now(),
    )
    save()
    return 0 if complete else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("training-release", "training", "addon", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    args = parser.parse_args(argv)
    try:
        return run(args.training_release, args.model_id, args.training, args.addon, args.output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
