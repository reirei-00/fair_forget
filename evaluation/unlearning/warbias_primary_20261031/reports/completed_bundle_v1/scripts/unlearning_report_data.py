"""Read hash-bound primary-study reports without training or model inference."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from collections import Counter
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
WARBIAS_REVISION = "164f355623c2da0028821e0942cad493558a6c13"


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


def _view_validator(root, references, evidence):
    """Load a byte-bound copy of the maintained stdlib view validator."""
    require(
        set(references) == {"inference_views", "inference_views_validator"},
        "Portable inference views require their certificate and maintained validator",
    )
    sidecar, _ = bound(root, references["inference_views"])
    source, _ = bound(root, references["inference_views_validator"])
    spec = importlib.util.spec_from_file_location("fairforget_checkpoint_views", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for role, path in (("inference_views", sidecar), ("inference_views_validator", source)):
        evidence.append({"role": role, "path": str(path), "sha256": references[role]["sha256"]})
    return module, sidecar


def _post_identity(training, observed, report=None, before_files=None):
    view = training.get("inference_view_validation")
    if view is None:
        require(
            observed["parameters"]["pretrained"] == training["arm"]["checkpoint"],
            "Wrong post checkpoint",
        )
        require(
            observed.get("local_files_sha256") == training["post_files"],
            "Post checkpoint hashes differ from training",
        )
        return None
    kwargs = {
        **view["kwargs"],
        "evaluation_checkpoint": observed["parameters"]["pretrained"],
        "evaluation_files_sha256": observed["local_files_sha256"],
        "base_files_sha256": before_files,
    }
    validated = view["module"].validate_inference_view(view["sidecar"], **kwargs)
    if report is not None:
        require(
            report.get("analysis_source_sha256", {}).get("checkpoint_views.py")
            == view["validator_sha256"],
            "Portable inference-view validator differs from maintained analysis source",
        )
    return validated


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
    result = {
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
    view_refs = {
        key: references[key]
        for key in ("inference_views", "inference_views_validator")
        if key in references
    }
    if view_refs:
        module, sidecar = _view_validator(root, view_refs, evidence)
        certificate = module.load_inference_views(sidecar, view_refs["inference_views"]["sha256"])
        descriptor = certificate["models"][model["repository"]]["methods"][method]
        source_files = {
            item["path"]: item["sha256"] for item in receipt.get("source_model_files", [])
        }
        kwargs = {
            "sidecar_sha256": view_refs["inference_views"]["sha256"],
            "model_id": model["repository"],
            "method": method,
            "receipt_path": root / references["receipt"]["path"],
            "receipt_sha256": references["receipt"]["sha256"],
            "original_checkpoint": result["arm"]["checkpoint"],
            "original_files_sha256": post_files,
            "evaluation_checkpoint": descriptor["evaluation_checkpoint"],
            "evaluation_files_sha256": descriptor["evaluation_files_sha256"],
            "model_revision": model["revision"],
            "tokenizer_revision": model["tokenizer_revision"],
            "base_path": recipe["model_path"],
            "base_config_sha256": execution.get("model_config_sha256")
            or source_files.get("config.json"),
        }
        module.validate_inference_view(sidecar, **kwargs)
        result["inference_view_validation"] = {
            "module": module,
            "sidecar": sidecar,
            "kwargs": kwargs,
            "validator_sha256": view_refs["inference_views_validator"]["sha256"],
        }
    return result


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
    view = _post_identity(training, after_model, report, before_model.get("local_files_sha256"))
    if view is not None:
        linked = paired["training_binding"].get("inference_view", {})
        require(
            linked.get("sidecar", {}).get("sha256") == view["sidecar"]["sha256"]
            and all(
                linked.get(key) == view[key]
                for key in (
                    "kind",
                    "original_receipt",
                    "original_checkpoint",
                    "original_files_sha256",
                    "evaluation_checkpoint",
                    "evaluation_files_sha256",
                    "added_assets",
                )
            ),
            "Native paired inference-view proof differs",
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
    if manifest.get("study") == "warbias":
        return _warbias_schedule(root, manifest, models, evidence)
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


def _warbias_schedule(root, manifest, models, evidence):
    source, schedule = bound_json(root, manifest["schedule_source"])
    require(
        schedule.get("kind") == "six_models_three_methods_fresh_WarBias_training"
        and schedule.get("status") == "scheduler_submitted_not_completed"
        and set(schedule["models"]) == set(MODEL_IDS)
        and set(schedule["methods"]) == set(METHODS)
        and schedule.get("current_completed_training_cells") == 0
        and schedule.get("new_training_cells_expected") == 18
        and schedule.get("qwen_stereoset_training_reused") is False,
        "Expected fresh six-model WarBias submissions, not completed training",
    )
    require(
        schedule["plan"]["binding_sha256"] == manifest["training_scope"]["training_binding_sha256"]
        and schedule["target_dataset"] == "FairForget/WarBias"
        and schedule["source_revision"] == WARBIAS_REVISION,
        "Submitted WarBias source binding differs",
    )
    states = {}
    for line in schedule["scheduler_snapshot"].splitlines():
        fields = line.split("|", 2)
        require(len(fields) == 3 and fields[0] not in states, "Invalid recorded job snapshot")
        states[fields[0]] = " ".join(fields[1:])
    rows = []
    for model in models:
        job = schedule["models"][model["id"]]
        identifier = job["job_id"]
        require(
            isinstance(identifier, str)
            and identifier.isdigit()
            and job["returncode"] == 0
            and job["stdout"].strip().split(";")[0] == identifier
            and identifier in states,
            "WarBias submission identity differs",
        )
        rows.append(
            {
                "model": model["label"],
                "training_job": identifier,
                "recorded_training_state": states[identifier],
                "recorded_at_utc": schedule["submitted_at_utc"],
                "completion_evidence": "Separate actual trainer receipts required",
            }
        )
    require(len({r["training_job"] for r in rows}) == len(states) == 6, "Duplicate training job")
    evidence.append(
        {
            "role": "queued_warbias_training_not_results",
            "path": str(source),
            "sha256": digest(source),
        }
    )
    return rows


def _completed_training_dependencies(root, manifest, models, source, training, evidence):
    """Verify completed-training scheduling evidence without filling result cells."""
    reference = manifest.get("training_completion_source")
    require(reference, "Completed training dependency policy requires a bound completion proof")
    path, completion = bound_json(root, reference)
    plan = source["plan"]
    require(
        reference["sha256"] == plan.get("training_completion_sha256"),
        "Completion proof hash differs from submission plan",
    )
    matrix_path, matrix = bound_json(root, manifest["preparation"]["training_matrix"])
    require(
        completion.get("schema_version") == 1
        and completion.get("kind") == "actual_six_model_eighteen_WarBias_training_completion"
        and completion.get("status") == "all_18_actual_training_arms_completed"
        and completion.get("source_dataset") == "FairForget/WarBias"
        and completion.get("target_rows") == matrix["training_budget"]["max_steps"] == 137
        and completion.get("primary_epochs") == matrix["training_budget"]["epochs"] == 1
        and completion.get("source_binding_sha256")
        == matrix["source_training_binding_sha256"]
        == manifest["training_scope"]["training_binding_sha256"]
        and completion.get("source_matrix_sha256") == digest(matrix_path)
        and set(completion.get("models", {})) == set(MODEL_IDS),
        "Completed training proof has wrong source, matrix, scope or completion status",
    )
    jobs = {training["models"][model["id"]]["job_id"] for model in models}
    accounting = plan.get("satisfied_training_dependencies", {})
    command = accounting.get("command", [])
    require(
        accounting.get("returncode") == 0
        and len(command) == 7
        and command[:6] == ["sacct", "-X", "-n", "-P", "--format=JobIDRaw,State,ExitCode", "-j"]
        and len(command[6].split(",")) == 6
        and set(command[6].split(",")) == jobs,
        "Completed training requires exact successful six-job sacct evidence",
    )
    states = {}
    for line in accounting.get("stdout", "").splitlines():
        fields = line.strip().split("|")
        require(
            len(fields) == 3 and fields[0] in jobs and fields[0] not in states,
            "Invalid raw training sacct evidence",
        )
        states[fields[0]] = {"state": fields[1], "exit_code": fields[2]}
    require(
        set(states) == jobs
        and len(states) == 6
        and all(value == {"state": "COMPLETED", "exit_code": "0:0"} for value in states.values())
        and accounting.get("verified_states") == states,
        "Training sacct raw and verified states must all be COMPLETED0:0",
    )

    def valid_hash(value):
        return (
            isinstance(value, str)
            and len(value) == 64
            and not (set(value) - set("0123456789abcdef"))
        )

    for model in models:
        alias = model["id"]
        actual, submitted, original = (
            completion["models"][alias],
            plan["models"][alias],
            training["models"][alias],
        )
        profile = matrix["models"][alias]
        require(
            actual.get("training_job_id") == submitted["training_job_id"] == original["job_id"]
            and submitted["training"] == original["output"]
            and actual.get("repository") == profile["model_id"] == model["repository"]
            and actual.get("revision") == profile["model_revision"] == model["revision"]
            and profile["tokenizer_revision"] == model["tokenizer_revision"]
            and actual.get("stage") == "training_complete",
            "Completed training model, tokenizer pin, job or source path differs",
        )
        require(
            actual["execution"]["path"] == str(Path(submitted["training"]) / "execution.json")
            and valid_hash(actual["execution"]["sha256"])
            and actual["execution"]["sha256"] == submitted.get("training_execution_sha256")
            and set(actual.get("qualification", {}))
            == set(actual.get("methods", {}))
            == set(submitted.get("original_receipts", {}))
            == set(METHODS),
            "Completed training execution hash or method registry differs",
        )
        for method in METHODS:
            qualified, arm = actual["qualification"][method], actual["methods"][method]
            expected_receipt = (
                Path(submitted["training"])
                / method
                / ("run.json" if method == "rmu" else "intervention.json")
            )
            expected_checkpoint = expected_receipt.parent / (
                "checkpoint" if method == "rmu" else "merged_checkpoint"
            )
            require(
                qualified.get("status") == arm.get("status") == "trained_checkpoint_saved"
                and qualified.get("returncode") == arm.get("returncode") == 0
                and type(qualified.get("actual_steps")) is int
                and qualified["actual_steps"] == 1
                and type(arm.get("actual_steps")) is int
                and arm["actual_steps"] == 137
                and arm.get("receipt") == submitted["original_receipts"][method]
                and arm["receipt"]["path"] == str(expected_receipt)
                and valid_hash(arm["receipt"]["sha256"])
                and arm.get("original_checkpoint") == str(expected_checkpoint),
                "Completed method qualification, 137-step status or original receipt differs",
            )
    evidence.append(
        {
            "role": "completed_training_for_scheduler_dependencies_only",
            "path": str(path),
            "sha256": digest(path),
        }
    )


def _completed_heldout_dependencies(root, manifest, models, source, evidence):
    """Accept only hash-bound completed native jobs as fulfilled CPU inputs."""
    declared = source["plan"].get("completed_heldout_dependencies", {})
    require(isinstance(declared, dict), "Invalid fulfilled heldout dependency registry")
    if not declared:
        return set()
    aliases = {model["id"] for model in models}
    require(set(declared).issubset(aliases), "Unknown fulfilled heldout model")
    reference = manifest.get("heldout_completion_source")
    require(reference, "Fulfilled heldout dependency requires a bound completion proof")
    path, audit = bound_json(root, reference)
    _, matrix = bound_json(root, manifest["preparation"]["training_matrix"])
    _, training = bound_json(root, manifest["training_completion_source"])
    plan = source["plan"]
    require(
        audit.get("schema_version") == 1
        and audit.get("kind") == "actual_six_model_heldout_collection_completion"
        and audit.get("status") in {"completed_subset_heldout_verified", "all_six_heldout_verified"}
        and set(audit.get("models", {})) == aliases
        and set(declared).issubset(audit.get("completed_models", []))
        and audit.get("collector_release") == plan["collector_release"]
        and audit.get("collector_checksums_sha256") == plan["collector_checksums_sha256"]
        and audit.get("training_release") == plan["training_release"]
        and audit.get("training_checksums_sha256") == plan["training_checksums_sha256"]
        and audit.get("model_calls") == 0
        and audit.get("new_metrics_computed") is False
        and audit.get("weights_hashed_or_downloaded") is False
        and audit.get("remote_mutations") is False,
        "Heldout completion proof has wrong scope, source or completion status",
    )

    def valid_hash(value):
        return (
            isinstance(value, str)
            and len(value) == 64
            and not (set(value) - set("0123456789abcdef"))
        )

    def fingerprint(value):
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()

    def file_reference(value, expected):
        return (
            isinstance(value, dict)
            and set(value) == {"path", "sha256"}
            and value["path"] == expected
            and valid_hash(value["sha256"])
        )

    jobs = {source["models"][alias]["heldout"]["job_id"] for alias in aliases}
    accounting = audit.get("accounting", {})
    command = accounting.get("command", [])
    require(
        accounting.get("returncode") == 0
        and len(command) == 7
        and command[:6] == ["sacct", "-X", "-n", "-P", "--format=JobIDRaw,State,ExitCode", "-j"]
        and len(command[6].split(",")) == 6
        and set(command[6].split(",")) == jobs,
        "Heldout completion requires the exact original six-job sacct query",
    )
    states = {}
    for line in accounting.get("stdout", "").splitlines():
        fields = line.split("|")
        require(
            len(fields) == 3 and fields[0] in jobs and fields[0] not in states,
            "Invalid heldout raw sacct evidence",
        )
        states[fields[0]] = {"state": fields[1], "exit_code": fields[2]}
    require(
        set(states) == jobs and states == audit.get("verified_accounting_states"),
        "Heldout raw and verified sacct states differ",
    )
    identity = audit.get("collector_code_identity", {})
    require(
        identity
        and all(valid_hash(value) for value in identity.values())
        and fingerprint(identity) == audit.get("collector_code_identity_sha256"),
        "Heldout collector code identity differs",
    )
    tasks = manifest["warbias_tasks"]
    checkpoint_names = {"baseline", *METHODS}
    expected_native = {name + "/" + task for name in checkpoint_names for task in WARBIAS_TASKS}
    expected_pairs = {method + "/" + task for method in METHODS for task in WARBIAS_TASKS}
    for alias, selected in declared.items():
        actual, event, profile = (
            audit["models"][alias],
            source["models"][alias]["heldout"],
            matrix["models"][alias],
        )
        submitted = plan["models"][alias]
        require(
            selected.get("audit_sha256") == reference["sha256"]
            and selected.get("job_id") == actual.get("job_id") == event["job_id"]
            and actual.get("heldout_submission") == event
            and actual.get("status") == "completed_zero_exit_and_all_saved_evidence_verified"
            and actual.get("accounting")
            == states[event["job_id"]]
            == {"state": "COMPLETED", "exit_code": "0:0"},
            "Fulfilled heldout job, proof hash or COMPLETED0:0 evidence differs",
        )
        require(
            actual.get("model_profile")
            == {
                key: profile[key]
                for key in (
                    "model_path",
                    "model_id",
                    "model_revision",
                    "tokenizer_revision",
                    "config_sha256",
                )
            }
            and actual.get("training_execution") == training["models"][alias]["execution"]
            and actual.get("original_training_receipts") == submitted["original_receipts"],
            "Heldout completed model pins or training source differs",
        )
        collection, analysis = event["collection_output"], event["native_analysis_output"]
        for field, expected in (
            ("collection_execution", collection + "/supplemental_execution.json"),
            ("native_analysis_execution", analysis + "/supplemental_analysis.json"),
        ):
            require(
                file_reference(actual.get(field), expected)
                and selected.get(field) == actual[field],
                "Heldout completion execution reference differs",
            )
        require(
            actual.get("source_manifest")
            == {
                "path": plan["collector_release"] + "/evaluation/manifest.json",
                "sha256": manifest["preparation"]["heldout_manifest"]["sha256"],
            }
            and actual.get("source_tasks") == tasks
            and actual.get("no_update_passed") is True
            and actual.get("qualification_gates") == 24
            and actual.get("native_collection_count") == 48
            and actual.get("native_pair_count") == 36
            and set(actual.get("native_collections", {})) == expected_native
            and set(actual.get("native_pairs", {})) == expected_pairs,
            "Heldout full source, 48 collections or 36 pairs are incomplete",
        )
        checkpoints, identities = (
            actual.get("checkpoints", {}),
            actual.get("native_checkpoint_identities", {}),
        )
        require(
            set(checkpoints) == set(identities) == checkpoint_names
            and checkpoints["baseline"] == profile["model_path"],
            "Heldout checkpoint registry differs",
        )
        for name, checkpoint in checkpoints.items():
            observed = identities[name]
            files = observed.get("local_files_sha256", {})
            require(
                observed.get("parameters", {}).get("pretrained") == checkpoint
                and files
                and all(valid_hash(value) for value in files.values()),
                "Heldout native checkpoint identity differs",
            )
        require(
            identities["baseline"]["local_files_sha256"].get("config.json")
            == profile["config_sha256"],
            "Heldout baseline config pin differs",
        )
        require(
            actual.get("inference_views") == plan["inference_views"]
            and set(actual.get("inference_view_proofs", {})) == set(METHODS),
            "Heldout inference-view source differs",
        )
        for method in METHODS:
            view = actual["inference_view_proofs"][method]
            trained = training["models"][alias]["methods"][method]
            original, observed, assets = (
                view.get("original_files_sha256", {}),
                identities[method]["local_files_sha256"],
                view.get("added_assets", {}),
            )
            require(
                view.get("status") == "validated_inference_view_of_actual_training_checkpoint"
                and view.get("model_id") == profile["model_id"]
                and view.get("method") == method
                and view.get("sidecar") == plan["inference_views"]
                and view.get("original_receipt") == trained["receipt"]
                and view.get("original_checkpoint") == trained["original_checkpoint"]
                and view.get("evaluation_checkpoint") == checkpoints[method]
                and view.get("evaluation_files_sha256") == observed
                and original
                and all(valid_hash(value) for value in original.values())
                and view.get("original_weight_config_tokenizer_files_unchanged") is True,
                "Heldout post checkpoint is not the verified original training view",
            )
            if view.get("kind") == "original":
                require(
                    not assets
                    and original == observed
                    and checkpoints[method] == trained["original_checkpoint"],
                    "Original heldout checkpoint view changed",
                )
            else:
                require(
                    view.get("kind") == "add_pinned_processor_assets"
                    and alias in MODEL_IDS[2:]
                    and set(assets) == {"preprocessor_config.json", "processor_config.json"}
                    and not (set(original) & set(assets)),
                    "Unsupported heldout derived view",
                )
                expected = dict(original)
                for filename, asset in assets.items():
                    require(
                        asset.get("source_path") == profile["model_path"] + "/" + filename
                        and asset.get("sha256")
                        == identities["baseline"]["local_files_sha256"].get(filename)
                        and valid_hash(asset.get("sha256")),
                        "Heldout processor source differs",
                    )
                    expected[filename] = asset["sha256"]
                require(observed == expected, "Heldout derived view changes original files")
        for key, native in actual["native_collections"].items():
            name, task = key.split("/")
            require(
                native.get("status") == "verified_full_native_collection"
                and native.get("checkpoint") == checkpoints[name]
                and native.get("source_task") == task
                and native.get("source_sha256") == tasks[task]["sha256"]
                and native.get("source_rows") == native.get("selected_rows") == tasks[task]["rows"]
                and native.get("protocol") == tasks[task]["protocol"]
                and native.get("source_bundle_manifest_sha256")
                == actual["source_manifest"]["sha256"]
                and valid_hash(native.get("selected_ids_sha256"))
                and native.get("model_identity_sha256") == fingerprint(identities[name])
                and file_reference(
                    native.get("status_receipt"),
                    collection + "/" + name + "/status/" + task + ".json",
                )
                and set(native.get("files", {}))
                == {"run.json", "report.json", "predictions.jsonl"},
                "Heldout full native source or checkpoint binding differs",
            )
            for filename, record in native["files"].items():
                require(
                    file_reference(record, collection + "/" + name + "/" + task + "/" + filename)
                    and actual.get("validated_input_sha256", {}).get(record["path"])
                    == record["sha256"],
                    "Heldout native file reference differs",
                )
            baseline = actual["native_collections"]["baseline/" + task]
            require(
                native.get("selected_ids_sha256") == baseline.get("selected_ids_sha256")
                and native.get("evaluation_asset_identity")
                == baseline.get("evaluation_asset_identity")
                and all(
                    valid_hash(native["evaluation_asset_identity"].get(field))
                    for field in ("tokenizer_sha256", "input_sha256")
                ),
                "Heldout paired tokenizer, inputs or selected IDs differ",
            )
        for key, pair in actual["native_pairs"].items():
            method, task = key.split("/")
            expected_inputs = {
                name + "/" + filename: actual["native_collections"][name + "/" + task]["files"][
                    filename
                ]
                for name in ("baseline", method)
                for filename in ("run.json", "report.json", "predictions.jsonl")
            }
            require(
                pair.get("status") == "verified_native_pair"
                and file_reference(
                    pair.get("report"), analysis + "/" + method + "/" + task + "/paired.json"
                )
                and pair.get("inputs") == expected_inputs
                and pair.get("source_rows") == tasks[task]["rows"]
                and valid_hash(pair.get("source_rows_sha256"))
                and pair.get("direction") == "after_minus_before"
                and pair.get("bootstrap")
                == {
                    "draws": 1000,
                    "paired": True,
                    "seed": 42,
                    "unit": "semantic_component_within_stratum",
                },
                "Heldout native pair hash or source binding differs",
            )
    evidence.append(
        {
            "role": "completed_heldout_for_scheduler_dependencies_only",
            "path": str(path),
            "sha256": reference["sha256"],
        }
    )
    return set(declared)


def _warbias_measurement_schedule(root, manifest, models, evidence):
    if not manifest.get("measurement_source"):
        return []
    path, source = bound_json(root, manifest["measurement_source"])
    _, training = bound_json(root, manifest["schedule_source"])
    require(
        source.get("kind") == "six_WarBias_models_heldout_training_source_and_18_metric_entries"
        and source.get("status") == "scheduler_submitted_not_completed"
        and set(source["models"]) == set(MODEL_IDS)
        and source["plan"]["training_checksums_sha256"] == training["plan"]["checksums_sha256"]
        and source["heldout_native_pairs_expected"] == 216
        and source["training_source_pairs_expected"] == 18
        and source["all_metrics_reports_expected"] == 18,
        "WarBias measurement submission scope/source differs",
    )
    policy = source["plan"].get("training_dependency_policy")
    completed_dependencies = policy == "verified_completed_sacct_and_exact_audit"
    require(
        policy in (None, "verified_completed_sacct_and_exact_audit"),
        "Unknown training dependency policy",
    )
    if completed_dependencies:
        _completed_training_dependencies(root, manifest, models, source, training, evidence)
    completed_heldout = _completed_heldout_dependencies(root, manifest, models, source, evidence)
    require(
        not completed_heldout or completed_dependencies,
        "Fulfilled heldout inputs require the verified completed-training policy",
    )
    states = {}
    for line in source["scheduler_snapshot"].splitlines():
        fields = line.split("|", 2)
        require(len(fields) == 3 and fields[0] not in states, "Invalid measurement job snapshot")
        states[fields[0]] = " ".join(fields[1:])
    rows, identifiers = [], []
    for model in models:
        jobs = source["models"][model["id"]]
        trained = training["models"][model["id"]]["job_id"]
        require(
            source["plan"]["models"][model["id"]]["training_job_id"] == trained,
            "Measurement job targets another training job",
        )
        row = {"model": model["label"], "recorded_at_utc": source["submitted_at_utc"]}
        for name in ("heldout", "training_diagnostic", "all_metrics"):
            job = jobs[name]
            identifier = job["job_id"]
            require(
                isinstance(identifier, str)
                and identifier.isdigit()
                and job["returncode"] == 0
                and job["stdout"].strip().split(";")[0] == identifier
                and identifier in states,
                "Invalid actual measurement job submission",
            )
            dependency = jobs["heldout"]["job_id"] if name == "all_metrics" else trained
            if completed_dependencies:
                edges = [
                    argument
                    for argument in job["command"]
                    if argument.startswith("--dependency") or argument.startswith("-d")
                ]
                require(
                    edges
                    == (
                        ["--dependency=afterok:" + dependency]
                        if name == "all_metrics" and model["id"] not in completed_heldout
                        else []
                    ),
                    "Completed training policy has an unexpected or missing fresh success dependency",
                )
                require(
                    "--kill-on-invalid-dep=yes" in job["command"],
                    "Measurement job lacks invalid-dependency protection",
                )
            else:
                require(
                    "--dependency=afterok:" + dependency in job["command"]
                    and "--kill-on-invalid-dep=yes" in job["command"],
                    "Measurement job lacks its success dependency",
                )
            row[name + "_job"] = identifier
            row[name + "_recorded_state"] = states[identifier]
            identifiers.append(identifier)
        rows.append(row)
    require(
        len(identifiers) == len(set(identifiers)) == 18 and set(identifiers).issubset(states),
        "Measurement jobs are not eighteen distinct submissions",
    )
    evidence.append(
        {"role": "measurement_submissions_not_results", "path": str(path), "sha256": digest(path)}
    )
    return rows


def _warbias_preparation(root, manifest, models, evidence):
    """Validate preparation separately from actual training and performance."""
    loaded = {}
    for name, reference in manifest["preparation"].items():
        path, value = bound_json(root, reference)
        loaded[name] = (path, value)
        evidence.append({"role": "prepared/" + name, "path": str(path), "sha256": digest(path)})
    review_path, review = loaded["review_manifest"]
    binding_path, binding = loaded["training_binding"]
    _, audit = loaded["family_audit"]
    _, tokenizers = loaded["tokenizer_audit"]
    _, architecture = loaded["architecture_audit"]
    matrix_path, matrix = loaded["training_matrix"]
    diagnostic_path, diagnostic = loaded["diagnostic_manifest"]
    _, heldout = loaded["heldout_manifest"]
    require(
        review["reviewed_for_gradients"] is True
        and review["language"] == "uk"
        and review["training_target"]["dataset"] == "FairForget/WarBias"
        and review["sources"]["warbias_train"]["revision"] == WARBIAS_REVISION
        and review["sources"]["warbias_train"]["repo"] == "FairForget/WarBias",
        "Readiness requires the reviewed WarBias gradient source",
    )
    require(
        binding["deployment_binding"]["source_review_manifest_sha256"] == digest(review_path)
        and manifest["training_scope"]["training_binding_sha256"] == digest(binding_path)
        and manifest["training_scope"]["forget_stream_sha256"]
        == review["files"]["forget_train"]["sha256"],
        "Readiness source/stream binding differs",
    )
    banks = {}
    for name in (
        "forget_train",
        "correct_train",
        "retain_train",
        "dev",
        "retain_dev",
        "case_review",
    ):
        item = review["files"][name]
        path = (review_path.parent / item["filename"]).resolve()
        require(path.parent == review_path.parent, "Reviewed bank escapes its input directory")
        require(digest(path) == item["sha256"], "Reviewed bank hash differs")
        banks[name] = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        require(len(banks[name]) == item["rows"], "Reviewed bank count differs")
        evidence.append({"role": "prepared/" + name, "path": str(path), "sha256": digest(path)})
    train = banks["forget_train"]
    identifiers = {r["source_item_id"] for r in train}
    require(
        identifiers == set(binding["permitted_gradient_ids"])
        and len(identifiers) == len(train)
        and all(
            {r["source_item_id"] for r in banks[name]} == identifiers
            for name in ("correct_train", "retain_train")
        ),
        "Reviewed gradient source IDs differ",
    )
    actors = dict(Counter(r["actor"] for r in train))
    require(
        actors == review["training_target"]["actor_counts"]
        and sorted(actors) == review["training_target"]["actor_values"],
        "Prepared target actors differ from actual forget provenance",
    )
    for name in ("forget_train", "correct_train", "retain_train", "retain_dev"):
        for row in banks[name]:
            require(
                row["prompt"] + row["target"] == row["source_full_sentence"]
                and row["source_char_span"]
                == [len(row["prompt"]), len(row["source_full_sentence"])],
                "A prepared mask rewrites its source sentence",
            )
            phase = "warbias_development" if name == "retain_dev" else "warbias_train"
            require(
                row["source_hash"] == review["sources"][phase]["sha256"]
                and row["source_repository"] == "FairForget/WarBias"
                and row["source_revision"] == WARBIAS_REVISION,
                "Prepared source is not the pinned WarBias row bank",
            )
    gradient_families = {
        r["family_id"]
        for name in ("forget_train", "retain_train", "correct_train")
        for r in banks[name]
    }
    development_families = {r["family_id"] for name in ("dev", "retain_dev") for r in banks[name]}
    protected = set(audit["protected_locked_components"])
    require(
        not (
            gradient_families & development_families
            or gradient_families & protected
            or development_families & protected
        ),
        "Prepared gradient/development/locked semantic families overlap",
    )
    require(
        tokenizers["passed"] is True
        and tokenizers["binding_sha256"] == digest(binding_path)
        and {m["model_id"] for m in tokenizers["models"]} == set(MODEL_IDS)
        and all(m["forget_lexical_prompt_overlap_count"] == 0 for m in tokenizers["models"]),
        "Six-tokenizer source-mask audit is not qualified",
    )
    for name in ("forget_train", "correct_train", "retain_train", "dev", "retain_dev"):
        require(
            tokenizers["data_files_sha256"][name + ".jsonl"] == review["files"][name]["sha256"],
            "Tokenizer audit describes another reviewed bank",
        )
    require(
        matrix["source_training_binding_sha256"] == digest(binding_path)
        and matrix["source_architecture_audit_sha256"] == digest(loaded["architecture_audit"][0])
        and matrix["training_budget"]["epochs"] == 1
        and matrix["training_budget"]["max_steps"] == len(train)
        and matrix["methods"] == list(METHODS),
        "One-full-pass training budget or source binding differs",
    )
    plans = {}
    architecture_models = {item["id"]: item for item in architecture["models"]}
    for model in models:
        profile = matrix["models"][model["id"]]
        require(
            profile["model_id"] == model["repository"]
            and profile["model_revision"] == model["revision"]
            and profile["tokenizer_revision"] == model["tokenizer_revision"],
            "Prepared model/tokenizer pin differs",
        )
        require(
            profile["config_sha256"] == architecture_models[model["id"]]["config_sha256"],
            "Model geometry/config audit differs from the planned checkpoint",
        )
        plans[model["label"]] = {}
        for method in METHODS:
            ref = profile["recipes"][method]
            path, recipe = bound_json(matrix_path.parent, ref)
            require(
                recipe["method"] == method
                and recipe["data_manifest_sha256"] == digest(binding_path)
                and recipe["streams"]["forget"]["sha256"]
                == review["files"]["forget_train"]["sha256"]
                and recipe["training"]["epochs"] == 1
                and recipe["training"]["max_steps"] == len(train)
                and recipe["training"]["batch_size"] == 1,
                "Prepared method recipe differs from the reviewed full pass",
            )
            plans[model["label"]][method] = recipe
            evidence.append(
                {
                    "role": "planned/" + model["id"] + "/" + method,
                    "path": str(path),
                    "sha256": digest(path),
                }
            )
    task = "warbias_triplets_uk"
    diagnostic_entry = diagnostic["datasets"][task]
    rows_path = diagnostic_path.parent / diagnostic_entry["filename"]
    require(digest(rows_path) == diagnostic_entry["sha256"], "Diagnostic subset hash differs")
    diagnostic_rows = [
        json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()
    ]
    require(
        diagnostic["training_diagnostic"]["kind"] == "training_source_diagnostic_not_heldout"
        and diagnostic_entry["source_role"] == "training_signal"
        and diagnostic_entry["split"] == "train"
        and {r["id"] for r in diagnostic_rows} == identifiers
        and all(r["split"] == "train" for r in diagnostic_rows),
        "Training-source diagnostics are not the exact reviewed training subset",
    )
    original_ref = diagnostic["training_diagnostic"]["source_adapted"]
    original_path, original_payload = bound(diagnostic_path.parent, original_ref)
    original_rows = {
        row["id"]: row
        for row in (json.loads(line) for line in original_payload.splitlines() if line.strip())
    }
    require(
        original_ref["sha256"] == review["sources"]["warbias_train"]["sha256"]
        and all(row == original_rows.get(row["id"]) for row in diagnostic_rows),
        "Training-source diagnostic rows differ from the unchanged original source",
    )
    evidence.append(
        {
            "role": "prepared/original_training_source",
            "path": str(original_path),
            "sha256": digest(original_path),
        }
    )
    for task in WARBIAS_TASKS:
        require(
            manifest["warbias_tasks"][task]
            == {key: heldout["datasets"][task][key] for key in ("rows", "sha256", "protocol")},
            "Held-out source specification differs",
        )
    return {
        "counts": review["counts"],
        "human_validated": review["human_validated"],
        "actor_counts": actors,
        "mask_policy": review["mask_policy"],
        "case_review": banks["case_review"],
        "mask_examples": [
            {
                key: r[key]
                for key in (
                    "source_item_id",
                    "source_case_id",
                    "prompt",
                    "target",
                    "source_char_span",
                )
            }
            for r in train
            if r["source_case_id"] in ("WB-C009", "WB-C063", "WB-C119")
            and r["gender"] == "not_stated"
            and r["age"] == "not_stated"
        ],
        "plans": plans,
        "budget": matrix["training_budget"],
        "tokenizer_models": tokenizers["models"],
        "model_architecture": architecture_models,
        "protected_components": len(protected),
        "diagnostic_expected": {
            "task": "warbias_triplets_uk",
            "protocol": diagnostic_entry["protocol"],
            "rows": len(diagnostic_rows),
            "sha256": diagnostic_entry["sha256"],
            "selected_ids_sha256": hashlib.sha256(
                "\n".join(r["id"] for r in diagnostic_rows).encode()
            ).hexdigest(),
            "source_rows_sha256": hashlib.sha256(
                json.dumps(
                    sorted(diagnostic_rows, key=lambda r: r["id"]),
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                ).encode()
            ).hexdigest(),
            "manifest_sha256": digest(diagnostic_path),
            "binding_sha256": digest(binding_path),
        },
    }


def _training_diagnostic_native_run(execution_path, execution, arm, task_id, provenance, evidence):
    """Bind native-schema provenance to the copied original run bytes."""
    task = execution["tasks"][arm]
    files = task.get("files", {})
    names = {"run.json", "report.json", "predictions.jsonl"}
    require(
        task.get("status") == "completed"
        and task.get("returncode") == 0
        and set(files) == names
        and task.get("native_files_sha256") == {name: files[name]["sha256"] for name in names}
        and all(
            isinstance(files[name]["sha256"], str)
            and len(files[name]["sha256"]) == 64
            and not (set(files[name]["sha256"]) - set("0123456789abcdef"))
            and files[name]["path"] == str(Path(task["output"]) / name)
            for name in names
        )
        and provenance.get("predictions_sha256") == files["predictions.jsonl"]["sha256"],
        "Native diagnostic files or prediction provenance differ from completed collection",
    )
    # The assembler already copies this fixed portable layout. Original remote
    # paths in execution/provenance remain unchanged; the bound SHA locates bytes.
    relative = Path("native") / arm / task_id / "run.json"
    native_path, native_run = bound_json(
        execution_path.parent,
        {"path": str(relative), "sha256": files["run.json"]["sha256"]},
    )
    require(
        native_run == provenance["run"],
        "Native diagnostic copied run differs from paired run provenance",
    )
    evidence.append(
        {
            "role": "training_source_diagnostic/native_run/" + arm,
            "path": str(native_path),
            "sha256": digest(native_path),
        }
    )
    return native_run


def _training_diagnostic(root, entry, model, method, training, preparation, evidence):
    """Read an actual native TRAIN comparison without promoting it to held-out evidence."""
    if not entry.get("training_diagnostic"):
        return None
    require(training is not None, "A training-source comparison needs actual WarBias training")
    references = entry["training_diagnostic"]
    execution_path, execution = bound_json(root, references["execution"])
    comparison_path, comparison = bound_json(root, references["comparison"])
    require(
        execution.get("kind") == "training_source_diagnostic_not_heldout"
        and execution.get("stage") == "training_diagnostic_complete"
        and execution.get("model_id") == model["id"]
        and execution.get("heldout_evaluation") is False
        and execution.get("no_update_passed") is True,
        "Native training-source diagnostic is incomplete or belongs to another model",
    )
    pair = execution["pairs"][method]
    require(
        pair.get("status") == "analyzed"
        and pair.get("returncode") == 0
        and pair["report"]["sha256"] == digest(comparison_path),
        "Training-source paired report differs from its execution receipt",
    )
    expected = preparation["diagnostic_expected"]
    require(
        comparison.get("kind") == "paired_native_checkpoint_evaluation"
        and comparison.get("task") == expected["task"]
        and comparison.get("source_rows_sha256") == expected["source_rows_sha256"],
        "Diagnostic comparison does not cover the exact training-source panel",
    )
    runs = {phase: comparison["provenance"][phase]["run"] for phase in ("before", "after")}
    for phase, run in runs.items():
        require(
            run["task"] == expected["task"]
            and run["protocol"] == expected["protocol"]
            and run["dataset"]["sha256"] == expected["sha256"]
            and run["dataset"]["source_role"] == "training_signal"
            and run["dataset"]["split"] == "train"
            and run["dataset"]["measurement_purpose"] == "training_source_diagnostic_not_heldout"
            and run["dataset"]["bundle_manifest_sha256"] == expected["manifest_sha256"]
            and run["selected_rows"] == expected["rows"]
            and run["selected_ids_sha256"] == expected["selected_ids_sha256"]
            and run.get("limit_clusters_per_stratum") is None,
            "Diagnostic source, phase or full-row coverage differs",
        )
        assets = run.get("evaluation_asset_identity", {})
        require(
            all(
                isinstance(assets.get(key), str) and len(assets[key]) == 64
                for key in ("tokenizer_sha256", "input_sha256")
            ),
            "Diagnostic effective-input evidence is missing",
        )
        _training_diagnostic_native_run(
            execution_path,
            execution,
            "baseline" if phase == "before" else method,
            expected["task"],
            comparison["provenance"][phase],
            evidence,
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
            runs["before"].get(key) == runs["after"].get(key),
            "Diagnostic native execution inputs differ",
        )
    before, after = (runs[p]["model_identity"] for p in ("before", "after"))
    require(
        before["parameters"]["pretrained"] == training["recipe"]["model_path"],
        "Diagnostic checkpoint lineage differs from actual training",
    )
    view = _post_identity(training, after, before_files=before.get("local_files_sha256"))
    if view is not None:
        require(
            execution.get("bindings", {}).get("inference_views", {}).get("sha256")
            == view["sidecar"]["sha256"],
            "Diagnostic inference-view certificate differs",
        )
    parameters = [
        {k: v for k, v in item["parameters"].items() if k not in ("pretrained", "revision")}
        for item in (before, after)
    ]
    require(parameters[0] == parameters[1], "Diagnostic model execution arguments differ")
    if method != "rmu":
        base = hash_map(training["receipt"]["source_model_files"])
        require(
            before.get("local_files_sha256")
            and all(
                base.get(name) == checksum
                for name, checksum in before["local_files_sha256"].items()
            ),
            "Diagnostic base hashes differ from training",
        )
    require(
        execution["bindings"]["recipes"][method]["sha256"] == training["recipe_sha256"],
        "Diagnostic recipe differs from the actual intervention",
    )
    require(
        execution["bindings"]["data_manifest"]["sha256"] == expected["binding_sha256"]
        and execution["bindings"]["diagnostic_manifest"]["sha256"] == expected["manifest_sha256"],
        "Diagnostic training source or diagnostic manifest binding differs",
    )
    evidence.extend(
        {"role": "training_source_diagnostic/" + label, "path": str(path), "sha256": digest(path)}
        for label, path in (("execution", execution_path), ("comparison", comparison_path))
    )
    return comparison["effects"]


def _saved_native_metric_row(model, task, scope, name, value, units):
    """Copy maintained native values and distinguish availability from uncertainty."""
    require(
        isinstance(value, dict)
        and all(key in value for key in ("pre", "post", "delta", "uncertainty"))
        and isinstance(value["uncertainty"], dict)
        and "ci95" in value["uncertainty"]
        and isinstance(value["uncertainty"].get("status"), str),
        "Saved native metric values or uncertainty schema are incomplete",
    )
    uncertainty = value["uncertainty"]
    units = units.get(name) if isinstance(units, dict) else units
    require(units is None or isinstance(units, str), "Native metric unit is not a label or null")
    return {
        "model": model,
        "task": task,
        "scope": json.dumps(scope, sort_keys=True),
        "metric": name,
        "before": value["pre"],
        "after": value["post"],
        "change": value["delta"],
        # The maintained native_changes schema supplies uncertainty.status,
        # not an availability status. Preserve absence instead of inventing one.
        "status": value.get("status"),
        "ci95": uncertainty["ci95"],
        "uncertainty_status": uncertainty["status"],
        "valid_replicates": uncertainty.get("valid_replicates"),
        "units": units,
    }


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
        inventory.get("kind") == "read_only_six_model_three_method_current_evidence_inventory"
        or (manifest.get("study") == "warbias" and inventory.get("schema_version") == 1),
        "Expected six-model evidence audit",
    )
    pins = {item.get("model_id", item.get("id")): item for item in inventory["models"]}
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
    preparation = (
        _warbias_preparation(path.parent, manifest, models, evidence)
        if manifest.get("study") == "warbias"
        else None
    )
    schedule = _schedule(path.parent, manifest, models, evidence)
    measurement_schedule = (
        _warbias_measurement_schedule(path.parent, manifest, models, evidence)
        if manifest.get("study") == "warbias"
        else []
    )
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
    metric_artifacts = []
    diagnostic_coverage, diagnostic_values = [], []
    for model in models:
        require(len(model["revision"]) == 40, "Model pin must be immutable")
        require(set(manifest["runs"][model["id"]]) == set(METHODS), "Missing method coverage entry")
        entry = manifest["runs"][model["id"]][method]
        training = _training(
            path.parent, entry, model, method, evidence, manifest["training_scope"]
        )
        if preparation:
            native_diagnostic = _training_diagnostic(
                path.parent, entry, model, method, training, preparation, evidence
            )
            diagnostic_coverage.append(
                {
                    "model": model["label"],
                    "source_rows": preparation["diagnostic_expected"]["rows"],
                    "status": "available" if native_diagnostic is not None else "pending",
                    "heldout": False,
                    "reason": "Native saved training-source comparison not supplied"
                    if native_diagnostic is None
                    else None,
                }
            )
            for value in native_diagnostic or []:
                diagnostic_values.append({"model": model["label"], **value})
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
            metric_artifacts.append(
                {
                    "model": model["label"],
                    "path": str(report_path),
                    "sha256": entry["analysis"]["sha256"],
                    "saved_values": {
                        key: report.get(key)
                        for key in ("metrics", "transfer", "rules", "limitations")
                    },
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
                            _saved_native_metric_row(
                                model["label"],
                                task,
                                scope,
                                name,
                                value,
                                scoped.get("units"),
                            )
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
        "study": manifest.get("study", "historical_stereoset"),
        "preparation": preparation,
        "training_source_coverage": diagnostic_coverage,
        "training_source_metric_values": diagnostic_values,
        "scheduled_jobs": schedule,
        "measurement_jobs": measurement_schedule,
        "coverage": coverage,
        "training_parameters": parameters,
        "training_history": histories,
        "training_exposure": exposures,
        "recipe_settings": settings,
        "warbias_metric_values": metric_values,
        "warbias_task_coverage": task_coverage,
        "metric_status": metric_status,
        "saved_metric_artifacts": metric_artifacts,
        "provenance": evidence,
        "six_model_measurement_complete": all(
            row["measurement_status"] == "complete" for row in coverage
        ),
    }
