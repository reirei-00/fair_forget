"""Analyze three real WarBias-trained arms with the frozen saved-only CLI.

Twelve native WarBias tasks can be available. Five other native tasks remain
pending and English StereoSet remains source-policy blocked. Eighteen metric
entries do not mean eighteen qualified or available metrics.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

import evaluate_checkpoint as native
import supplemental

METHODS = ("rmu", "task_vector", "wga_kl")
WARBIAS_REVISION = "164f355623c2da0028821e0942cad493558a6c13"
ANALYSIS_COMMIT = "8f5915e89bc35713de32155ef5ebeb16202556e0"
METRIC_IDS = frozenset(
    "forget_efficacy NNR retain_utility LMS ICAT M1 M1b M2 M3a M3b M3c M3d RI M4a M4b BBQ training_imbalance severity".split()
)
PENDING_TASKS = frozenset(
    "bbq_uk stereoset_uk winobias_uk_controlled winobias_uk_natural winopron_uk".split()
)
TASK_IDS = frozenset(supplemental.PROTOCOLS) | PENDING_TASKS | {"stereoset_en"}
ANALYSIS_FILES = (
    "all_metrics.py",
    "checkpoint_views.py",
    "collector_export.py",
    "intervention_metrics.py",
    "transfer_metrics.py",
    "analysis_inputs.py",
    "baseline.py",
    "study_metric_sources.json",
)
RULES = {
    "tau_by_task": {},
    "min_independent_support": None,
    "min_target_improvement": None,
    "bootstrap": 1000,
    "seed": 42,
}
require, read, evidence = supplemental.require, supplemental.read, supplemental.evidence


def checked(path, expected, inputs):
    path = Path(path).resolve()
    actual = native.digest(path)
    require(actual == expected, "Input SHA-256 differs: " + str(path))
    inputs[str(path)] = actual
    return path


def inside(root, value):
    path = (root / value).resolve()
    require(path.is_relative_to(root.resolve()), "Input escapes its selected release")
    return path


def snapshot(path, inputs):
    return checked(path, native.digest(path), inputs)


def declared_targets(manifest, forget_rows=None):
    """Use only predeclared training-source actor coverage, never outcomes."""
    values = manifest.get("training_target", {}).get("actor_values")
    if values is None:
        return {}
    require(
        isinstance(values, list)
        and bool(values)
        and all(
            isinstance(value, str) and bool(value.strip()) and value == value.strip()
            for value in values
        )
        and len(set(values)) == len(values),
        "Declared training actors must be unique nonempty strings",
    )
    if forget_rows is not None:
        observed = {row.get("actor") for row in forget_rows}
        require(
            observed == set(values), "Forget-source actors differ from the declared training target"
        )
    return {"actor": list(values)}


def checked_analysis(source, inputs):
    """Bind the selected frozen analysis release, without replacing its kernels."""
    root = source.parent
    package = read(snapshot(root / "package.json", inputs))
    require(package.get("analysis_commit") == ANALYSIS_COMMIT, "Analysis source commit differs")
    checksums = read(snapshot(root / "checksums.json", inputs))
    paths = [
        path for path in source.rglob("*") if path.is_file() and "__pycache__" not in path.parts
    ]
    require(
        paths and (source / "lm_eval/fair_uk/all_metrics.py").is_file(), "Analysis source missing"
    )
    for path in paths:
        checked(path, checksums[str(path.relative_to(root))], inputs)
    return {name: native.digest(source / "lm_eval/fair_uk" / name) for name in ANALYSIS_FILES}


def validate_loss_history(method, events, optimizer_steps):
    """Preserve the deployed trainers' method-specific raw index conventions."""
    require(
        method in METHODS and type(optimizer_steps) is int and optimizer_steps > 0,
        "Invalid method or optimizer-step budget",
    )
    first = 1 if method == "rmu" else 0
    expected = list(range(first, first + optimizer_steps))
    require(
        len(events) == optimizer_steps
        and all(type(row.get("step")) is int for row in events)
        and [row["step"] for row in events] == expected,
        "Complete ordered actual loss history is required: "
        + method
        + " steps "
        + str(first)
        + ".."
        + str(expected[-1]),
    )


def checked_training(release, model_id, training, inputs):
    matrix_path = snapshot(release / "matrix.json", inputs)
    matrix = read(matrix_path)
    require(
        matrix.get("methods") == list(METHODS),
        "Matrix must select exactly the requested three methods",
    )
    entry = matrix["models"][model_id]
    execution_path = snapshot(training / "execution.json", inputs)
    execution = read(execution_path)
    require(
        execution.get("stage") == "training_complete"
        and execution.get("model_id") == model_id
        and execution.get("repository") == entry["model_id"]
        and execution.get("model_revision") == entry["model_revision"]
        and execution.get("tokenizer_revision") == entry["tokenizer_revision"]
        and execution.get("matrix_sha256") == native.digest(matrix_path)
        and set(execution.get("methods", {})) == set(METHODS),
        "Complete selected-model training with its frozen matrix is required",
    )
    require(
        set(execution.get("qualification", {})) == set(METHODS), "Missing method qualifications"
    )
    require(
        all(
            arm.get("status") == "trained_checkpoint_saved"
            and arm.get("returncode") == 0
            and arm.get("steps") == 1
            for arm in execution["qualification"].values()
        ),
        "All three one-step training qualifications must pass",
    )
    profile = {key: entry[key] for key in supplemental.MODEL_PROFILE_FIELDS}
    checkpoints = supplemental.checkpoint_panel(
        None, training, methods=METHODS, model_profile=profile
    )
    recipes, receipts, files, common_binding = {}, {}, {}, None
    for method in METHODS:
        reference = entry["recipes"][method]
        recipe_path = checked(inside(release, reference["path"]), reference["sha256"], inputs)
        recipe = read(recipe_path)
        require(recipe.get("method") == method, "Recipe method differs")
        require(
            all(
                recipe.get(key) == entry[key]
                for key in ("model_id", "model_revision", "tokenizer_revision", "model_path")
            ),
            "Recipe model or tokenizer pin differs",
        )
        arm = execution["methods"][method]
        require(
            arm.get("recipe") == evidence(recipe_path)
            and execution["recipes"][method] == evidence(recipe_path),
            "Executed recipe identity differs",
        )
        receipt_path = inside(
            training, method + ("/run.json" if method == "rmu" else "/intervention.json")
        )
        require(arm["receipt"] == evidence(receipt_path), "Actual receipt identity differs")
        snapshot(receipt_path, inputs)
        receipt = read(receipt_path)
        config = receipt.get("config", {}) if method == "rmu" else receipt
        provenance = receipt.get("provenance", {}) if method == "rmu" else receipt
        require(
            provenance.get("config_sha256") == reference["sha256"],
            "Receipt is not bound to the frozen method recipe",
        )
        require(
            all(
                config.get(key) == recipe[key]
                for key in ("model_id", "model_revision", "tokenizer_revision")
            ),
            "Receipt model or tokenizer pin differs",
        )
        require(
            type(receipt.get("steps")) is int and receipt["steps"] == arm["steps"] > 0,
            "Actual optimizer steps differ",
        )
        binding = checked(
            inside(release, str((recipe_path.parent / recipe["data_manifest"]).resolve())),
            recipe["data_manifest_sha256"],
            inputs,
        )
        require(common_binding in (None, binding), "Methods use different reviewed bindings")
        common_binding = binding
        for role, stream in recipe["streams"].items():
            checked(
                inside(release, str((recipe_path.parent / stream["path"]).resolve())),
                stream["sha256"],
                inputs,
            )
        history = training / method / ("history.jsonl" if method == "rmu" else "losses.jsonl")
        events = supplemental.rows(snapshot(history, inputs))
        validate_loss_history(method, events, receipt["steps"])
        exposure = snapshot(training / method / "exposure.jsonl", inputs)
        require(bool(supplemental.rows(exposure)), "Actual exposure events are missing")
        if method != "rmu":
            require(
                receipt.get("loss_log_sha256") == native.digest(history)
                and receipt.get("exposure_log_sha256") == native.digest(exposure),
                "Receipt loss or exposure hash differs",
            )
        else:
            require(
                provenance.get("data_manifest", {}).get("sha256") == recipe["data_manifest_sha256"],
                "RMU reviewed binding differs",
            )
        if method == "task_vector":
            scale = (receipt.get("task_vector_selection") or {}).get("selected")
            require(
                type(scale) in (int, float)
                and scale in recipe["task_vector"]["scales"]
                and receipt.get("null_intervention_selected") is (scale == 0),
                "Task-vector selection differs from the frozen recipe",
            )
        for path in (receipt_path, history, exposure):
            files[str(path.relative_to(training))] = native.digest(path)
        recipes[method], receipts[method] = recipe_path, receipt
    files["execution.json"] = native.digest(execution_path)
    binding = read(common_binding)
    review = binding["deployment_binding"]
    review_path = checked(
        inside(release, str((common_binding.parent / review["source_review_manifest"]).resolve())),
        review["source_review_manifest_sha256"],
        inputs,
    )
    manifest = read(review_path)
    target, published = (
        manifest.get("training_target", {}),
        manifest.get("sources", {}).get("warbias_train", {}),
    )
    require(
        target.get("dataset") == published.get("repo") == "FairForget/WarBias"
        and target.get("source_revision") == published.get("revision") == WARBIAS_REVISION
        and manifest.get("reviewed_for_gradients") is True
        and binding.get("reviewed_for_gradients") is True
        and binding.get("language") == "uk",
        "A separately reviewed Ukrainian WarBias gradient target is required",
    )
    streams = {}
    for role, spec in binding["streams"].items():
        path = checked(
            inside(release, str((common_binding.parent / spec["filename"]).resolve())),
            spec["sha256"],
            inputs,
        )
        streams[role] = supplemental.rows(path)
        require(len(streams[role]) == spec["rows"], "Reviewed stream row count differs")
    require(
        bool(streams["forget_train"])
        and all(
            row.get("source_hash") == published.get("sha256") for row in streams["forget_train"]
        ),
        "Forget gradients do not come from the reviewed WarBias train source",
    )
    declared_targets(manifest, streams["forget_train"])
    permitted = set(manifest["permitted_gradient_ids"])
    require(
        all(
            row["source_item_id"] in permitted
            for key in ("forget_train", "correct_train", "retain_train")
            for row in streams[key]
        ),
        "Gradient source ID is outside the reviewed training set",
    )
    for method, recipe_path in recipes.items():
        recipe = read(recipe_path)
        for role, name in (
            ("forget", "forget_train"),
            ("correction", "correct_train"),
            ("retain", "retain_train"),
            ("development", "dev"),
            ("retain_development", "retain_dev"),
        ):
            require(
                recipe["streams"][role]["sha256"] == binding["streams"][name]["sha256"],
                "Recipe stream differs from the reviewed binding",
            )
    return entry, checkpoints, recipes, receipts, review_path, common_binding, files


def checked_collection(collection, collector, training, entry, checkpoints, receipts, inputs):
    record_path = snapshot(collection / "supplemental_execution.json", inputs)
    record = read(record_path)
    addon_ref = record["addon_config"]
    config_path = checked(addon_ref["path"], addon_ref["sha256"], inputs)
    config, actual_collector, full, small = supplemental.validate(config_path.parent)
    require(
        actual_collector == collector and config.get("methods") == list(METHODS),
        "Selected collector or method subset differs",
    )
    require(
        config.get("model_profile")
        == supplemental.checked_model_profile(
            {key: entry[key] for key in supplemental.MODEL_PROFILE_FIELDS}
        ),
        "Collection model pins differ from the matrix",
    )
    if config.get("inference_views"):
        snapshot(
            checked(config["inference_views"]["path"], config["inference_views"]["sha256"], inputs),
            inputs,
        )
        selected = supplemental.checkpoint_panel(
            collector,
            training,
            methods=METHODS,
            model_profile={key: entry[key] for key in supplemental.MODEL_PROFILE_FIELDS},
            inference_views=config["inference_views"],
        )
        checkpoints.clear()
        checkpoints.update(selected)
    require(
        record.get("stage") == "supplemental_collection_complete"
        and record.get("no_update_passed") is True
        and record.get("collection_mode") != "baseline_only"
        and record.get("training") == evidence(training / "execution.json")
        and record.get("checkpoints") == {key: str(path) for key, path in checkpoints.items()}
        and record.get("expected_full_collections") == 48,
        "All 48 qualified selected-checkpoint collections are required",
    )
    expected = {name + "/" + task for name in checkpoints for task in supplemental.PROTOCOLS}
    require(set(record.get("tasks", {})) == expected, "Full collection task inventory differs")
    expected_qualification = {
        name + "/" + task for name in checkpoints for task in supplemental.REPRESENTATIVES
    }
    require(
        set(record.get("qualification", {})) == expected_qualification,
        "Qualification inventory differs",
    )
    snapshot(full, inputs)
    snapshot(small, inputs)
    for key, value in record["qualification"].items():
        name, task = key.split("/")
        require(
            value.get("passed") is True and len(value.get("runs", [])) == 2,
            "Fresh repeated no-update qualification missing",
        )
        roots = []
        for repeat, event in enumerate(value["runs"], 1):
            directory = collection / "qualification" / name / task / f"repeat-{repeat}" / task
            require(
                event.get("returncode") == 0
                and event.get("predictions") == evidence(directory / "predictions.jsonl"),
                "Qualification prediction identity differs",
            )
            for filename in ("run.json", "report.json", "predictions.jsonl"):
                snapshot(directory / filename, inputs)
            roots.append(directory)
        gate = supplemental.compare_repeats(
            *roots,
            config["qualification"]["selection"][task]["ordered_ids"],
            supplemental.SENTENCE if "triplets" in task else supplemental.PROTOCOLS[task],
        )
        require(gate["passed"] is True, "Repeated no-update qualification no longer passes")
    base_hashes = None
    for task in supplemental.PROTOCOLS:
        dataset, source_rows, source_path = supplemental.source(full, task)
        snapshot(source_path, inputs)
        require(
            all(row.get("split") == "evaluation" for row in source_rows),
            "Only locked evaluation rows may fill WarBias pairs",
        )
        ids = [row["id"] for row in source_rows]
        for name, checkpoint in checkpoints.items():
            key = name + "/" + task
            queued = record["tasks"][key]
            status_path = collection / name / "status" / f"{task}.json"
            require(
                queued.get("status") == "collected"
                and queued.get("returncode") == 0
                and queued.get("receipt") == evidence(status_path),
                "Full native collection receipt differs: " + key,
            )
            status = read(snapshot(status_path, inputs))
            require(
                status.get("status") == "completed"
                and status.get("exit_code") == 0
                and status.get("full_supplemental_evaluation_bundle") is True
                and status.get("qualification_only") is False
                and status.get("manifest_sha256") == native.digest(full)
                and status.get("checkpoint") == str(checkpoint),
                "Full native receipt identity differs: " + key,
            )
            directory = collection / name / task
            verified = native.verify_complete(task, dataset, directory)
            require(
                verified["native_files_sha256"] == status["native_files_sha256"],
                "Native saved file changed: " + key,
            )
            supplemental.verify_collected_options(directory)
            require(
                [row["id"] for row in supplemental.rows(directory / "predictions.jsonl")] == ids,
                "Native full ordered IDs differ: " + key,
            )
            for filename in ("run.json", "report.json", "predictions.jsonl"):
                checked(directory / filename, status["native_files_sha256"][filename], inputs)
            run = read(directory / "run.json")
            identity = run["model_identity"]
            require(
                identity["parameters"].get("pretrained") == str(checkpoint),
                "Native checkpoint path differs: " + key,
            )
            hashes = identity["local_files_sha256"]
            if name == "baseline":
                require(
                    hashes.get("config.json") == entry["config_sha256"],
                    "Baseline model config pin differs",
                )
                require(
                    base_hashes in (None, hashes), "Baseline file identities differ across tasks"
                )
                base_hashes = hashes
            else:
                expected_files = receipts[name][
                    "checkpoint_identity" if name == "rmu" else "merged_checkpoint_files"
                ]
                if config.get("inference_views"):
                    supplemental.view_for_native_checkpoint(
                        collector,
                        config["inference_views"],
                        config["model_profile"],
                        checkpoint,
                        files=hashes,
                    )
                else:
                    require(
                        hashes == {item["path"]: item["sha256"] for item in expected_files},
                        "Native post checkpoint differs from the actual training receipt: " + key,
                    )
    for method in ("wga_kl", "task_vector"):
        source = {item["path"]: item["sha256"] for item in receipts[method]["source_model_files"]}
        require(
            all(source.get(name) == checksum for name, checksum in base_hashes.items()),
            "Native baseline differs from the actual trainer source",
        )
    return config, full


def make_config(
    method,
    model_id,
    collection,
    collector,
    training,
    recipe,
    manifest,
    binding,
    archive,
    inference_views=None,
):
    return {
        "schema_version": 1,
        "kind": "fair_uk_all_metrics",
        "collector": {"python": sys.executable, "evaluator_root": str(collector / "evaluator")},
        "intervention": {
            "id": "warbias_20261031_" + model_id + "_" + method,
            "method": method,
            "training_language": "uk",
            "target_values": declared_targets(read(manifest)),
        },
        "required_tasks": sorted(TASK_IDS),
        "task_blocks": {
            "stereoset_en": "Unpublished matched English StereoSet remains preparation-only under the approved source policy."
        },
        "pairs": [
            {
                "task": task,
                "phase": "evaluation",
                "before_run": str(collection / "baseline" / task),
                "after_run": str(collection / method / task),
                "dataset_bundle": str(collector / "evaluation/manifest.json"),
                "annotations": None,
                "before_judgments": None,
                "after_judgments": None,
            }
            for task in supplemental.PROTOCOLS
        ],
        "rules": RULES,
        "exposure": {
            "execution": str(training / "execution.json"),
            "data_manifest": str(manifest),
            "training_binding": str(binding),
            "method_directory": str(training / method),
            "recipe": str(recipe),
            "evidence_manifest": str(archive),
            **({"inference_views": inference_views} if inference_views else {}),
        },
        "scope": "Actual reviewed Ukrainian WarBias training and twelve WarBias endpoint views. Five other native task pairs remain pending; English StereoSet is source-policy blocked. Actor target values, if explicitly frozen in the reviewed training manifest, describe data-defined training coverage; they are not inferred from outcomes or an additional hypothesis. No other target axis, judge, calibration threshold or human labels are inferred.",
    }


def verify_report(report, method, recipe, receipt, directory, analysis_hashes, target_values=None):
    require(
        report.get("schema_version") == 1
        and report.get("kind") == "all_18_saved_study_metrics"
        and report.get("model_requests") == 0,
        "Unexpected saved-only report",
    )
    require(
        len(report.get("metrics", [])) == 18
        and {row["id"] for row in report["metrics"]} == METRIC_IDS,
        "All eighteen distinct metric entries are required",
    )
    require(
        all(row.get("status") in {"available", "partial", "pending"} for row in report["metrics"]),
        "Unknown metric dependency status",
    )
    require(
        report.get("analysis_source_sha256") == analysis_hashes,
        "Frozen analysis source identity differs",
    )
    require(
        report.get("intervention", {}).get("method") == method
        and report["intervention"].get("training_language") == "uk"
        and report["intervention"].get("target_values") == (target_values or {}),
        "Report intervention differs",
    )
    require(report.get("rules") == RULES, "Scientific thresholds or analysis rules changed")
    require(set(report.get("task_status", {})) == TASK_IDS, "Task inventory differs")
    for task, value in report["task_status"].items():
        expected = (
            "available"
            if task in supplemental.PROTOCOLS
            else "blocked"
            if task == "stereoset_en"
            else "pending"
        )
        require(value.get("status") == expected, task + ": native task coverage differs")
        if expected != "available":
            continue
        require(
            value.get("phase") == "evaluation", "Development output cannot fill an evaluation pair"
        )
        comparison = directory / task / "analysis.json"
        exported = directory / task / "validated_pair.json"
        require(
            native.digest(comparison) == value["analysis_sha256"]
            and native.digest(exported) == value["validated_export"]["sha256"],
            "Native paired output hashes differ",
        )
        pair = read(comparison)
        linkage = pair.get("training_binding", {})
        require(
            linkage.get("status") == "bound_to_actual_training_arm"
            and linkage.get("method") == method
            and linkage.get("receipt_sha256") == native.digest(receipt)
            and linkage.get("recipe_sha256") == native.digest(recipe),
            "Paired analysis is not bound to this actual training arm",
        )
    require(
        len(report.get("pairs", [])) == 12
        and report.get("training_exposure", {}).get("status") == "available",
        "Native pairs or actual exposure report missing",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "training-release",
        "training",
        "collection",
        "analysis-source",
        "collector-release",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    args = parser.parse_args(argv)
    inputs = {}
    try:
        paths = {
            name: getattr(args, name).resolve()
            for name in (
                "training_release",
                "training",
                "collection",
                "analysis_source",
                "collector_release",
                "output",
            )
        }
        output = paths.pop("output")
        require(
            not output.exists() and not any(output.is_relative_to(path) for path in paths.values()),
            "Use a new output directory outside saved inputs",
        )
        analysis_hashes = checked_analysis(paths["analysis_source"], inputs)
        entry, checkpoints, recipes, receipts, manifest, binding, files = checked_training(
            paths["training_release"], args.model_id, paths["training"], inputs
        )
        addon, full = checked_collection(
            paths["collection"],
            paths["collector_release"],
            paths["training"],
            entry,
            checkpoints,
            receipts,
            inputs,
        )
        if addon.get("inference_views"):
            require(
                analysis_hashes["checkpoint_views.py"]
                == native.digest(
                    paths["collector_release"] / "evaluator/lm_eval/fair_uk/checkpoint_views.py"
                ),
                "Collector and analysis inference-view validators differ",
            )
        require(
            full == paths["collector_release"] / "evaluation/manifest.json",
            "Full collector manifest differs",
        )
        output.mkdir(parents=True)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        print(json.dumps({"stage": "preflight_failed", "error": str(error)}), file=sys.stderr)
        return 2
    archive = output / "training_evidence.json"
    native.write_json(
        archive,
        {
            "kind": "actual_three_method_training_evidence",
            "status": "three_method_training_completed",
            "files": files,
            "weights_included": False,
        },
    )
    configs = {}
    for method in METHODS:
        config_path = output / (method + "-config.json")
        native.write_json(
            config_path,
            make_config(
                method,
                args.model_id,
                paths["collection"],
                paths["collector_release"],
                paths["training"],
                recipes[method],
                manifest,
                binding,
                archive,
                inference_views=addon.get("inference_views"),
            ),
        )
        snapshot(config_path, inputs)
        configs[method] = config_path
    snapshot(archive, inputs)
    status = {
        "stage": "warbias_saved_only_analysis",
        "model_requests": 0,
        "model_id": args.model_id,
        "workers": 3,
        "methods": {},
        "inputs_sha256": inputs,
        "driver_sha256": native.digest(Path(__file__)),
        "analysis_commit": ANALYSIS_COMMIT,
        "started_at_utc": native.utc_now(),
        "available_native_tasks_expected": 12,
        "pending_native_tasks_expected": sorted(PENDING_TASKS),
        "blocked_native_tasks_expected": ["stereoset_en"],
        "all_metrics_qualified": False,
    }
    status_path = output / "warbias_metrics_execution.json"
    lock, processes = threading.RLock(), {}
    native.write_json(status_path, status)
    environment = {
        **os.environ,
        "PYTHONPATH": str(paths["analysis_source"]),
        "CUDA_VISIBLE_DEVICES": "",
    }
    for name in (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "PYTHONDONTWRITEBYTECODE",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[name] = "1"

    def stop(signum, frame):
        with lock:
            for process in processes.values():
                native.stop_collector(process)
            status.update(stage="warbias_all_metrics_interrupted", finished_at_utc=native.utc_now())
            native.write_json(status_path, status)
        raise SystemExit(128 + signum)

    previous = {signum: signal.signal(signum, stop) for signum in (signal.SIGTERM, signal.SIGINT)}

    def worker(method):
        command = [
            sys.executable,
            "-m",
            "lm_eval.fair_uk",
            "paper",
            "all-metrics",
            "--config",
            str(configs[method]),
            "--output",
            str(output / method),
        ]
        record = {"status": "running", "command": command, "started_at_utc": native.utc_now()}
        with lock:
            status["methods"][method] = record
            native.write_json(status_path, status)
        try:
            with (output / (method + ".log")).open("x") as log:
                with lock:
                    process = subprocess.Popen(
                        command,
                        env=environment,
                        cwd=paths["analysis_source"],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    processes[method] = process
                code = process.wait()
                with lock:
                    processes.pop(method, None)
            with lock:
                record["returncode"] = code
            require(code == 0, "Maintained Fair-UK analysis failed")
            report_path = output / method / "all_metrics.json"
            receipt_path = (
                paths["training"]
                / method
                / ("run.json" if method == "rmu" else "intervention.json")
            )
            report = read(report_path)
            require(
                report.get("config") == evidence(configs[method]),
                "Report is not bound to the generated analysis config",
            )
            verify_report(
                report,
                method,
                recipes[method],
                receipt_path,
                report_path.parent,
                analysis_hashes,
                declared_targets(read(manifest)),
            )
            with lock:
                record.update(
                    status="analyzed_with_explicit_pending_metrics", report=evidence(report_path)
                )
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            with lock:
                record.update(status="failed", error=str(error))
        finally:
            with lock:
                record["finished_at_utc"] = native.utc_now()
                native.write_json(status_path, status)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(worker, METHODS))
        try:
            unchanged = all(
                native.digest(Path(path)) == checksum for path, checksum in inputs.items()
            )
        except OSError as error:
            unchanged = False
            status["input_validation_error"] = str(error)
        complete = (
            set(status["methods"]) == set(METHODS)
            and all(
                record["status"] == "analyzed_with_explicit_pending_metrics"
                for record in status["methods"].values()
            )
            and unchanged
        )
        status.update(
            stage="warbias_all_metrics_reports_complete"
            if complete
            else "warbias_all_metrics_partial_failure",
            input_hashes_unchanged=unchanged,
            finished_at_utc=native.utc_now(),
        )
        native.write_json(status_path, status)
        return 0 if complete else 1
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
