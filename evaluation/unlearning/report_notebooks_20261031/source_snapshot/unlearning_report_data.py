"""Read hash-bound primary-study reports without training or model inference."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

MODEL_IDS = (
    "qwen3_8b",
    "llama31_8b_instruct",
    "gemma3_12b_it",
    "lapa12b_instruct",
    "mamay12b_v2",
    "mamay27b_v2",
)
METHODS = ("rmu", "task_vector", "wga_kl")
METRIC_IDS = (
    "forget_efficacy",
    "NNR",
    "retain_utility",
    "LMS",
    "ICAT",
    "M1",
    "M1b",
    "M2",
    "M3a",
    "M3b",
    "M3c",
    "M3d",
    "RI",
    "M4a",
    "M4b",
    "BBQ",
    "training_imbalance",
    "severity",
)
WARBIAS_TASKS = tuple(
    f"warbias_{view}_{language}" if view else f"warbias_{language}"
    for view in ("triplets", "cross_actor", "expanded_qa", "benign", "", "intersectional")
    for language in ("uk", "en")
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bound(root, reference):
    require(isinstance(reference, dict), "A source requires a path and SHA-256")
    path = (root / reference["path"]).resolve()
    payload = path.read_bytes()
    require(
        hashlib.sha256(payload).hexdigest() == reference["sha256"],
        "Source hash differs: " + str(path),
    )
    return path, payload


def bound_json(root, reference):
    path, payload = bound(root, reference)
    return path, json.loads(payload)


def hash_map(records):
    result = {item["path"]: item["sha256"] for item in records}
    require(
        bool(result) and len(result) == len(records), "Checkpoint identity is empty or duplicated"
    )
    return result


def _training(root, entry, model, method, evidence, scope):
    if not entry.get("training"):
        return None
    references = entry["training"]
    loaded = {}
    for name in (
        "recipe",
        "receipt",
        "execution",
        "evidence_manifest",
        "loss_log",
        "exposure_summary",
    ):
        path, payload = bound(root, references[name])
        evidence.append({"role": name, "path": str(path), "sha256": references[name]["sha256"]})
        loaded[name] = (
            [json.loads(line) for line in payload.splitlines() if line.strip()]
            if name == "loss_log"
            else json.loads(payload)
        )
    recipe, receipt, execution = (loaded[key] for key in ("recipe", "receipt", "execution"))
    require(recipe["method"] == receipt["method"] == method, "Training method differs")
    require(
        receipt.get("status") == ("trained_not_evaluated" if method == "rmu" else "trained_merged"),
        "Training receipt does not record a completed method run",
    )
    require(
        recipe["model_id"] == model["repository"] and recipe["model_revision"] == model["revision"],
        "Training model differs from the six-model registry",
    )
    require(
        recipe["streams"]["forget"]["sha256"] == scope["forget_stream_sha256"],
        "Training forget stream differs from the declared reviewed target",
    )
    if scope.get("training_binding_sha256"):
        require(
            recipe.get("data_manifest_sha256") == scope["training_binding_sha256"],
            "Training binding differs from the declared reviewed target",
        )
    receipt_config = receipt.get("config", {}) if method == "rmu" else receipt
    require(
        recipe.get("tokenizer_revision")
        == receipt_config.get("tokenizer_revision")
        == model["tokenizer_revision"],
        "Training tokenizer pin differs from the independent registry",
    )
    require(
        receipt_config.get("model_id") == model["repository"]
        and receipt_config.get("model_revision") == model["revision"],
        "Training receipt model pin differs",
    )
    config_sha = receipt.get("config_sha256", receipt.get("provenance", {}).get("config_sha256"))
    require(
        config_sha == references["recipe"]["sha256"],
        "Training recipe does not match the actual receipt",
    )
    require(execution.get("stage") == "training_complete", "Training execution is incomplete")
    arm = execution["methods"][method]
    require(
        arm.get("status") == "trained_checkpoint_saved" and arm.get("returncode") == 0,
        "The requested training arm did not complete",
    )
    require(
        type(receipt["steps"]) is int and receipt["steps"] == arm["steps"] > 0,
        "Actual optimizer-step evidence differs",
    )
    for name in ("receipt", "loss_log"):
        filename = Path(references[name]["path"]).name
        require(
            loaded["evidence_manifest"]["files"][method + "/" + filename]
            == references[name]["sha256"],
            "Training file differs from the archived evidence",
        )
    require(len(loaded["loss_log"]) == receipt["steps"], "Training history is incomplete")
    if method != "rmu":
        require(
            receipt["loss_log_sha256"] == references["loss_log"]["sha256"],
            "Loss-log receipt hash differs",
        )
    post_files = hash_map(
        receipt["checkpoint_identity" if method == "rmu" else "merged_checkpoint_files"]
    )
    exposure = loaded["exposure_summary"]
    require(
        exposure.get("kind") == "observed_training_exposure_not_outcome_correlation",
        "Expected actual observed exposure summary",
    )
    exposure = exposure["methods"][method]
    require(exposure["optimizer_steps"] == receipt["steps"], "Exposure optimizer steps differ")
    require(
        exposure["sha256"] == loaded["evidence_manifest"]["files"][method + "/exposure.jsonl"],
        "Exposure summary is not bound to the archived log",
    )
    totals = {role: values["valid_loss_tokens"] for role, values in exposure["streams"].items()}
    if method != "rmu":
        require(
            totals == receipt["exposure_valid_tokens"],
            "Actual exposure totals differ from training",
        )
    else:
        require(
            totals
            == {
                "forget": sum(row["forget_exposure"]["tokens"] for row in loaded["loss_log"]),
                "preservation": sum(row["retain_exposure"]["tokens"] for row in loaded["loss_log"]),
            },
            "RMU exposure totals differ from history",
        )
    scale = (receipt.get("task_vector_selection") or {}).get("selected")
    null = receipt.get("null_intervention_selected")
    if method == "task_vector":
        require(
            type(scale) in (int, float) and math.isfinite(scale), "Task-vector selection is missing"
        )
        require(
            scale in recipe["task_vector"]["scales"],
            "Selected task-vector scale is outside the frozen recipe",
        )
        require(
            type(null) is bool and null == (scale == 0),
            "Task-vector null-selection evidence differs",
        )
    return {
        "recipe": recipe,
        "receipt": receipt,
        "arm": arm,
        "post_files": post_files,
        "steps": receipt["steps"],
        "seed": recipe["training"]["seed"],
        "losses": loaded["loss_log"],
        "selected_scale": scale,
        "null_intervention_selected": null,
        "recipe_sha256": references["recipe"]["sha256"],
        "receipt_sha256": references["receipt"]["sha256"],
        "exposure": exposure,
    }


def _pair(analysis_path, report, task, expected, training, model, method, evidence):
    state = report["task_status"][task]
    require(
        state.get("status") == "available" and state.get("phase") == "evaluation",
        "A development/preflight task is not a locked WarBias measurement",
    )
    directory = analysis_path.parent / task
    exported_path = directory / "validated_pair.json"
    comparison_path = directory / "analysis.json"
    require(
        digest(exported_path) == state["validated_export"]["sha256"],
        "Validated pair export hash differs",
    )
    require(digest(comparison_path) == state["analysis_sha256"], "Paired analysis hash differs")
    exported, paired = (json.loads(path.read_text()) for path in (exported_path, comparison_path))
    require(
        exported.get("kind") == "collector_validated_saved_pair"
        and exported.get("task") == paired.get("task") == task,
        "Saved pair task identity differs",
    )
    require(paired.get("phase") == "evaluation", "Paired analysis is not locked evaluation")
    binding = paired["training_binding"]
    require(
        binding.get("status") == "bound_to_actual_training_arm" and binding.get("method") == method,
        "Paired report is not bound to the requested training arm",
    )
    require(
        binding["base_model_id"] == model["repository"]
        and binding["base_revision"] == model["revision"],
        "Paired report model pin differs",
    )
    require(
        binding["receipt_sha256"] == training["receipt_sha256"]
        and binding["recipe_sha256"] == training["recipe_sha256"],
        "Paired report training evidence differs",
    )
    source_rows = exported["rows"]
    ids = [row["id"] for row in source_rows]
    require(
        len(source_rows) == expected["rows"] and len(set(ids)) == len(ids),
        "WarBias pair has incomplete full source coverage",
    )
    require(
        all(row.get("split") == "evaluation" and row.get("task") == task for row in source_rows),
        "WarBias rows are not the locked evaluation panel",
    )
    runs = {phase: exported["provenance"][phase]["run"] for phase in ("before", "after")}
    source_ids = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    for phase, run in runs.items():
        require(
            run["task"] == task and run["protocol"] == expected["protocol"],
            "Native task/protocol differs",
        )
        require(
            run["dataset"]["sha256"] == expected["sha256"]
            and run["dataset"]["measurement_eligibility"] == "active",
            "Native source is unapproved or differs",
        )
        require(
            run.get("limit_clusters_per_stratum") is None
            and run["selected_rows"] == expected["rows"]
            and run["selected_ids_sha256"] == source_ids,
            "Smoke/development output cannot satisfy full paired coverage",
        )
        provenance = exported["provenance"][phase]
        require(
            provenance["predictions_sha256"] == provenance["files_sha256"]["predictions.jsonl"],
            "Prediction provenance differs",
        )
    before, after = runs["before"], runs["after"]
    require(
        exported["collector_code_identity"]
        == state["collector_code_identity"]
        == before["code_identity"],
        "Frozen collector identity differs across artifacts",
    )
    for key in (
        "dataset",
        "protocol",
        "model_backend",
        "seed",
        "batch_size",
        "runtime_versions",
        "harness_version",
        "generation_config",
        "scoring_policy",
        "evaluation_asset_identity",
        "code_identity",
    ):
        require(
            before.get(key) == after.get(key),
            "Before/after runtime, source or effective-input identity differs: " + key,
        )
    for run in runs.values():
        assets = run.get("evaluation_asset_identity", {})
        require(
            all(
                isinstance(assets.get(key), str) and len(assets[key]) == 64
                for key in ("tokenizer_sha256", "input_sha256")
            ),
            "Effective-input evidence is missing",
        )
    before_model, after_model = (runs[p]["model_identity"] for p in ("before", "after"))
    require(
        before_model["parameters"]["pretrained"] == training["recipe"]["model_path"],
        "Wrong baseline checkpoint",
    )
    require(
        after_model["parameters"]["pretrained"] == training["arm"]["checkpoint"],
        "Wrong post checkpoint",
    )
    require(
        after_model.get("local_files_sha256") == training["post_files"],
        "Post checkpoint hashes differ from training",
    )
    parameters = [
        {k: v for k, v in item["parameters"].items() if k not in ("pretrained", "revision")}
        for item in (before_model, after_model)
    ]
    require(parameters[0] == parameters[1], "Before/after model execution arguments differ")
    if method != "rmu":
        base = hash_map(training["receipt"]["source_model_files"])
        require(
            before_model.get("local_files_sha256")
            and all(
                base.get(name) == checksum
                for name, checksum in before_model["local_files_sha256"].items()
            ),
            "Base checkpoint hashes differ from training",
        )
    require(paired["comparison"]["task"] == task, "Comparison task differs")
    evidence.extend(
        {"role": task + "/" + role, "path": str(path), "sha256": digest(path)}
        for role, path in (("validated_pair", exported_path), ("analysis", comparison_path))
    )
    return paired["comparison"]


def _schedule(root, manifest, models, evidence):
    """Read submitted jobs as a separate plan, never as completed experiments."""
    if not manifest.get("schedule_source"):
        return []
    source, schedule = bound_json(root, manifest["schedule_source"])
    require(
        schedule.get("kind") == "five_remaining_models_three_methods_before_training_after"
        and schedule.get("status") == "scheduler_submitted_not_completed",
        "Expected an actual submitted, incomplete execution plan",
    )
    require(
        schedule.get("requested_models") == 6
        and set(schedule.get("requested_methods", [])) == set(METHODS)
        and set(schedule["models"]) == set(MODEL_IDS) - {"qwen3_8b"}
        and schedule.get("new_training_cells") == 15
        and schedule.get("current_completed_new_model_method_pairs") == 0
        and schedule.get("no_new_WarBias_gradient_target_inferred") is True,
        "Submitted plan scope or incomplete status differs",
    )
    require(
        schedule.get("source_binding_sha256")
        == manifest["training_scope"].get("training_binding_sha256")
        and bool(schedule.get("source_binding_sha256")),
        "Submitted plan training binding differs",
    )
    states = {}
    for line in schedule["scheduler_snapshot"].splitlines():
        parts = line.split("|", 3)
        require(len(parts) == 4 and parts[0] not in states, "Invalid scheduler snapshot")
        states[parts[0]] = parts[1] + " " + parts[2]
    result, job_ids = [], []
    for model in models:
        identifier = model["id"]
        if identifier == "qwen3_8b":
            result.append(
                {
                    "model": model["label"],
                    "baseline_job": None,
                    "training_job": schedule["reused_qwen_training_job"],
                    "paired_job": schedule["existing_qwen_serial_measurement_job"],
                    "recorded_baseline_state": "Included in existing serial collection",
                    "recorded_training_state": "Saved receipt required for actual completion",
                    "recorded_paired_state": "Existing submitted job; state not asserted here",
                }
            )
            continue
        jobs = schedule["models"][identifier]
        require(
            set(jobs) == {"baseline", "training", "paired_evaluation"},
            "Submitted model stages differ",
        )
        row = {"model": model["label"]}
        for stage, label in (
            ("baseline", "baseline"),
            ("training", "training"),
            ("paired_evaluation", "paired"),
        ):
            job = jobs[stage]
            job_id = job["job_id"]
            require(
                isinstance(job_id, str)
                and job_id.isdigit()
                and job.get("returncode") == 0
                and job.get("stdout", "").strip().split(";")[0] == job_id
                and job_id in states,
                "Job submission or recorded scheduler identity differs",
            )
            job_ids.append(job_id)
            row[label + "_job"] = job_id
            row["recorded_" + label + "_state"] = states[job_id]
        require(
            "--dependency=afterok:" + jobs["training"]["job_id"]
            in jobs["paired_evaluation"]["command"]
            and "--kill-on-invalid-dep=yes" in jobs["paired_evaluation"]["command"],
            "Paired job lacks the actual successful-training dependency",
        )
        result.append(row)
    require(
        len(job_ids) == len(set(job_ids)) == len(states) == 15,
        "Submitted jobs must be fifteen distinct recorded jobs",
    )
    evidence.append(
        {
            "role": "submitted_execution_plan_not_completed",
            "path": str(source),
            "sha256": manifest["schedule_source"]["sha256"],
        }
    )
    return result


def load_manifest(path, method):
    """Return six-model coverage plus only verified saved measurement values."""
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    require(
        manifest.get("schema_version") in (1, 2)
        and manifest.get("kind") == "fairforget_unlearning_notebook_inputs",
        "Unsupported notebook input manifest",
    )
    require(method in METHODS, "Unsupported report method")
    models = manifest["models"]
    require(
        len(models) == 6 and {model["id"] for model in models} == set(MODEL_IDS),
        "Reports must retain exactly the six promised models",
    )
    require(set(manifest["warbias_tasks"]) == set(WARBIAS_TASKS), "WarBias view inventory differs")
    require(set(manifest["runs"]) == set(MODEL_IDS), "Missing model coverage entry")
    evidence = [
        {"role": "report_manifest", "path": str(path), "sha256": digest(path)},
        {
            "role": "replay_loader",
            "path": str(Path(__file__).resolve()),
            "sha256": digest(__file__),
        },
    ]
    if manifest.get("schema_version") == 2:
        parent_path, parent = bound_json(path.parent, manifest["parent_manifest"])
        require(parent.get("kind") == manifest["kind"], "Parent report manifest differs")
        evidence.append(
            {
                "role": "preceding_dated_report_manifest",
                "path": str(parent_path),
                "sha256": manifest["parent_manifest"]["sha256"],
            }
        )
    inventory_path, inventory = bound_json(path.parent, manifest["inventory_source"])
    require(
        inventory.get("kind") == "read_only_six_model_three_method_current_evidence_inventory",
        "Expected six-model evidence audit",
    )
    pins = {item["model_id"]: item for item in inventory["models"]}
    require(
        all(
            pins.get(model["id"], {}).get("repository") == model["repository"]
            and pins[model["id"]]["revision"] == model["revision"]
            and model.get("tokenizer_revision", pins[model["id"]]["tokenizer_revision"])
            == pins[model["id"]]["tokenizer_revision"]
            for model in models
        ),
        "Model registry differs from the source audit",
    )
    models = [
        {**model, "tokenizer_revision": pins[model["id"]]["tokenizer_revision"]} for model in models
    ]
    schedule = _schedule(path.parent, manifest, models, evidence)
    evidence.append(
        {
            "role": "six_model_audit",
            "path": str(inventory_path),
            "sha256": manifest["inventory_source"]["sha256"],
        }
    )
    coverage, metric_values, metric_status, histories, parameters, exposures, settings = (
        [],
        [],
        [],
        {},
        [],
        [],
        {},
    )
    task_coverage = []
    for model in models:
        require(len(model["revision"]) == 40, "Model pin must be immutable")
        require(set(manifest["runs"][model["id"]]) == set(METHODS), "Missing method coverage entry")
        entry = manifest["runs"][model["id"]][method]
        training = _training(
            path.parent, entry, model, method, evidence, manifest["training_scope"]
        )
        available = []
        report = None
        if entry.get("analysis"):
            require(training is not None, "Analysis has no actual matching training evidence")
            report_path, report = bound_json(path.parent, entry["analysis"])
            require(
                report.get("kind") == "all_18_saved_study_metrics"
                and report.get("model_requests") == 0,
                "Expected maintained saved-only analysis",
            )
            require(
                report["intervention"]["method"] == method
                and report["intervention"]["training_language"] == "uk",
                "Analysis method/language differs",
            )
            require(
                len(report["metrics"]) == 18
                and {row["id"] for row in report["metrics"]} == set(METRIC_IDS),
                "Incomplete metric inventory",
            )
            evidence.append(
                {
                    "role": model["id"] + "/all_metrics",
                    "path": str(report_path),
                    "sha256": entry["analysis"]["sha256"],
                }
            )
            for task, expected in manifest["warbias_tasks"].items():
                if report.get("task_status", {}).get(task, {}).get("status") != "available":
                    continue
                comparison = _pair(
                    report_path, report, task, expected, training, model, method, evidence
                )
                available.append(task)
                for scoped in comparison["native_changes"]:
                    scope = {
                        k: v
                        for k, v in scoped.items()
                        if k not in ("metrics", "weighting", "units")
                    }
                    for name, value in scoped["metrics"].items():
                        metric_values.append(
                            {
                                "model": model["label"],
                                "task": task,
                                "scope": json.dumps(scope, sort_keys=True),
                                "metric": name,
                                "before": value["pre"],
                                "after": value["post"],
                                "change": value["delta"],
                                "status": value["status"],
                                "ci95": value.get("uncertainty", {}).get("ci95"),
                                "units": scoped.get("units", "native benchmark scale"),
                            }
                        )
        if training:
            histories[model["label"]] = training["losses"]
            recipe = training["recipe"]
            parameters.append(
                {
                    "model": model["label"],
                    "actual_steps": training["steps"],
                    "seed": training["seed"],
                    "learning_rate": recipe["training"]["learning_rate"],
                    "batch_size": recipe["training"]["batch_size"],
                    "maximum_length": recipe["training"]["max_length"],
                    "selected_scale": training["selected_scale"],
                    "null_intervention": training["null_intervention_selected"],
                    "checkpoint": training["arm"]["checkpoint"],
                    "checkpoint_files": len(training["post_files"]),
                    "base_training_hash_binding": "path_and_recipe_pin_only"
                    if method == "rmu"
                    else "source_model_files",
                }
            )
            settings[model["label"]] = {
                key: recipe[key]
                for key in (
                    "method",
                    "model_id",
                    "model_revision",
                    "training",
                    "lora",
                    "rmu",
                    "task_vector",
                )
                if key in recipe
            }
            for role, values in training["exposure"]["streams"].items():
                exposures.append(
                    {
                        "model": model["label"],
                        "stream": role,
                        **{
                            key: value
                            for key, value in values.items()
                            if key != "family_valid_tokens"
                        },
                    }
                )
        for task in WARBIAS_TASKS:
            supplied = report.get("task_status", {}).get(task, {}) if report else {}
            task_coverage.append(
                {
                    "model": model["label"],
                    "task": task,
                    "status": "available" if task in available else "pending",
                    "reason": None
                    if task in available
                    else supplied.get("reason", entry.get("pending_reason", "Missing actual pair")),
                }
            )
        paired_status = (
            "complete"
            if len(available) == len(WARBIAS_TASKS)
            else "partial"
            if available
            else "pending"
        )
        coverage.append(
            {
                "model_id": model["id"],
                "model": model["label"],
                "repository": model["repository"],
                "revision": model["revision"],
                "tokenizer_revision": model["tokenizer_revision"],
                "training": "actual_completed" if training else "pending",
                "actual_steps": training["steps"] if training else None,
                "warbias_pairs": len(available),
                "expected_warbias_pairs": len(WARBIAS_TASKS),
                "measurement_status": paired_status,
                "reason": None
                if paired_status == "complete"
                else entry.get(
                    "pending_reason", "Full locked WarBias measurements are not available"
                ),
                "null_intervention": training["null_intervention_selected"] if training else None,
            }
        )
        for metric in METRIC_IDS:
            source = (
                next((row for row in report["metrics"] if row["id"] == metric), None)
                if report
                else None
            )
            metric_status.append(
                {
                    "model": model["label"],
                    "metric": metric,
                    "status": source["status"]
                    if source
                    else "partial"
                    if metric == "training_imbalance" and training
                    else "pending",
                    "reason": source.get("reason")
                    if source
                    else "Actual exposure observed; outcome association unavailable"
                    if metric == "training_imbalance" and training
                    else "No complete saved metric report supplied",
                }
            )
    return {
        "method": method,
        "mode": "saved_evidence_replay_no_training_or_inference",
        "manifest_sha256": digest(path),
        "as_of_utc": manifest["as_of_utc"],
        "training_scope": manifest["training_scope"],
        "scheduled_jobs": schedule,
        "coverage": coverage,
        "training_parameters": parameters,
        "training_history": histories,
        "training_exposure": exposures,
        "recipe_settings": settings,
        "warbias_metric_values": metric_values,
        "warbias_task_coverage": task_coverage,
        "metric_status": metric_status,
        "provenance": evidence,
        "six_model_measurement_complete": all(
            row["measurement_status"] == "complete" for row in coverage
        ),
    }
