"""Assemble portable, complete WarBias evidence and replay three result notebooks.

This program reads saved files only. It never loads weights, starts model
inference, evaluates new predictions, requests judgments or submits jobs.
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import unlearning_report_data as reports

METHODS, MODELS, TASKS = reports.METHODS, reports.MODEL_IDS, reports.WARBIAS_TASKS
RECEIPTS = {"rmu": "run.json", "task_vector": "intervention.json", "wga_kl": "intervention.json"}
HISTORIES = {"rmu": "history.jsonl", "task_vector": "losses.jsonl", "wga_kl": "losses.jsonl"}
BASE_PATH = Path("evaluation/unlearning/warbias_primary_20261031")
FORBIDDEN = {".safetensors", ".bin", ".pt", ".pth", ".gguf"}
require, digest = reports.require, reports.digest


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def reference(path, manifest):
    return {"path": os.path.relpath(path, manifest.parent), "sha256": digest(path)}


def copied(source, target, expected=None):
    source, target = Path(source), Path(target)
    require(
        source.is_file() and not source.is_symlink(),
        "Expected an ordinary saved evidence file: " + str(source),
    )
    require(source.suffix not in FORBIDDEN, "Model weights cannot enter the report bundle")
    actual = digest(source)
    require(expected is None or actual == expected, "Saved evidence hash changed: " + str(source))
    require(not target.exists(), "Preserve existing output: " + str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    require(digest(target) == actual, "Copied evidence hash differs")
    return target


def job_output(source, model, key=None):
    job = source["models"][model] if key is None else source["models"][model][key]
    require(
        isinstance(job.get("job_id"), str)
        and job["job_id"].isdigit()
        and job.get("returncode") == 0
        and job.get("stdout", "").strip().split(";")[0] == job["job_id"],
        "Unverified scheduler submission",
    )
    return Path(job["output"]).resolve()


def export_tables(output, results):
    """Export the already validated replay tables; calculate no new metrics."""
    require(
        set(results) == set(METHODS), "All three shared-loader replays must pass before CSV export"
    )
    tables = {
        "heldout": (
            "warbias_metric_values",
            (
                "model",
                "task",
                "scope",
                "metric",
                "before",
                "after",
                "change",
                "status",
                "ci95",
                "units",
            ),
        ),
        "training_source": (
            "training_source_metric_values",
            ("model", "scope", "metric", "before", "after", "change", "status"),
        ),
        "metrics": ("metric_status", ("model", "metric", "status", "reason")),
        "training_exposure": (
            "training_exposure",
            ("model", "stream", "events", "valid_loss_tokens", "weighted_loss_tokens"),
        ),
        "coverage": (
            "coverage",
            (
                "model_id",
                "model",
                "training",
                "actual_steps",
                "warbias_pairs",
                "expected_warbias_pairs",
                "measurement_status",
            ),
        ),
        "heldout_coverage": ("warbias_task_coverage", ("model", "task", "status", "reason")),
        "training_source_coverage": (
            "training_source_coverage",
            ("model", "source_rows", "status", "heldout", "reason"),
        ),
        "training_parameters": (
            "training_parameters",
            ("model", "actual_steps", "selected_scale", "null_intervention", "checkpoint"),
        ),
    }
    # Validate every method before creating even the first CSV directory/file.
    for method, result in results.items():
        coverage = result["coverage"]
        labels = {row["model"] for row in coverage}
        require(
            result.get("method") == method
            and len(coverage) == len(labels) == 6
            and {row["model_id"] for row in coverage} == set(MODELS),
            "CSV coverage must retain exactly the six verified models",
        )
        require(
            all(
                row["training"] == "actual_completed"
                and row["measurement_status"] == "complete"
                and row["actual_steps"] > 0
                for row in coverage
            ),
            "Partial training or measurements cannot be exported as completed tables",
        )
        metrics = result["metric_status"]
        require(
            len(metrics) == 108
            and {(row["model"], row["metric"]) for row in metrics}
            == {(model, metric) for model in labels for metric in reports.METRIC_IDS},
            "CSV metric inventory must retain all eighteen entries for six models",
        )
        diagnostic = result["training_source_coverage"]
        heldout = result["warbias_task_coverage"]
        require(
            len(diagnostic) == 6
            and {row["model"] for row in diagnostic} == labels
            and all(row["status"] == "available" for row in diagnostic),
            "CSV training-source diagnostic coverage is incomplete",
        )
        require(
            len(heldout) == 72
            and {(row["model"], row["task"]) for row in heldout}
            == {(model, task) for model in labels for task in TASKS}
            and all(row["status"] == "available" for row in heldout),
            "CSV held-out native coverage is incomplete",
        )
        for key in (
            "warbias_metric_values",
            "training_source_metric_values",
            "training_exposure",
            "training_parameters",
        ):
            require(
                {row["model"] for row in result[key]} == labels,
                "CSV table lacks verified model rows: " + key,
            )
    directory = output / "tables"
    require(not directory.exists(), "Preserve existing CSV tables")
    directory.mkdir()
    exported = {}
    for method in METHODS:
        for label, (key, empty_columns) in tables.items():
            data = results[method][key]
            columns = list(dict.fromkeys(name for row in data for name in row)) or list(
                empty_columns
            )
            path = directory / (method + "_" + label + ".csv")
            with path.open("x", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                for row in data:
                    writer.writerow(
                        {
                            name: json.dumps(
                                value, ensure_ascii=False, sort_keys=True, allow_nan=False
                            )
                            if isinstance(value, (dict, list, tuple))
                            else value
                            for name, value in row.items()
                        }
                    )
            exported[str(path.relative_to(output))] = {
                "sha256": digest(path),
                "rows": len(data),
                "source_table": key,
            }
    return exported


def preflight(base_manifest, training_submission, measurement_submission, release):
    ready = reports.load_manifest(base_manifest, "rmu")
    require(ready["study"] == "warbias", "The base manifest must select fresh WarBias training")
    train, measurement, matrix = (
        read(training_submission),
        read(measurement_submission),
        read(release / "matrix.json"),
    )
    require(
        set(train["models"]) == set(measurement["models"]) == set(MODELS)
        and set(train["methods"]) == set(METHODS),
        "All six models and three methods are required",
    )
    require(
        train["target_dataset"] == "FairForget/WarBias"
        and train["source_revision"] == reports.WARBIAS_REVISION
        and train["qwen_stereoset_training_reused"] is False,
        "Historical/non-WarBias training cannot fill this report",
    )
    require(
        digest(release / "checksums.json")
        == train["plan"]["checksums_sha256"]
        == measurement["plan"]["training_checksums_sha256"],
        "Training release identity differs",
    )
    require(
        matrix["source_training_binding_sha256"]
        == train["plan"]["binding_sha256"]
        == ready["training_scope"]["training_binding_sha256"],
        "Reviewed WarBias source binding differs",
    )
    count = ready["preparation"]["counts"]["forget_train"]
    binding_path = release / "data/training_binding.json"
    require(
        digest(binding_path) == matrix["source_training_binding_sha256"],
        "Released trainer binding differs",
    )
    expected_ids = set(read(binding_path)["permitted_gradient_ids"])
    contexts = {}
    for identifier in MODELS:
        training = job_output(train, identifier)
        metrics = job_output(measurement, identifier, "all_metrics")
        diagnostic = job_output(measurement, identifier, "training_diagnostic")
        execution = read(training / "execution.json")
        metric_execution = read(metrics / "warbias_metrics_execution.json")
        diagnostic_execution = read(diagnostic / "training_diagnostic_execution.json")
        profile = matrix["models"][identifier]
        require(
            execution.get("stage") == "training_complete"
            and execution.get("model_id") == identifier
            and execution.get("slurm_job_id") == train["models"][identifier]["job_id"]
            and execution.get("matrix_sha256") == digest(release / "matrix.json"),
            "Actual selected-model training is incomplete or mismatched",
        )
        require(
            execution.get("repository") == profile["model_id"]
            and execution.get("model_revision") == profile["model_revision"]
            and execution.get("tokenizer_revision") == profile["tokenizer_revision"],
            "Executed model/tokenizer pin differs",
        )
        require(
            set(execution.get("methods", {})) == set(METHODS)
            and set(execution.get("qualification", {})) == set(METHODS)
            and all(
                v.get("status") == "trained_checkpoint_saved"
                and v.get("returncode") == 0
                and v.get("steps") == 1
                for v in execution["qualification"].values()
            ),
            "All three independent training qualifications are required",
        )
        require(
            metric_execution.get("stage") == "warbias_all_metrics_reports_complete"
            and metric_execution.get("model_id") == identifier
            and metric_execution.get("input_hashes_unchanged") is True
            and metric_execution.get("model_requests") == 0
            and set(metric_execution.get("methods", {})) == set(METHODS),
            "Saved all-metrics execution is incomplete",
        )
        require(
            diagnostic_execution.get("stage") == "training_diagnostic_complete"
            and diagnostic_execution.get("kind") == "training_source_diagnostic_not_heldout"
            and diagnostic_execution.get("model_id") == identifier
            and diagnostic_execution.get("heldout_evaluation") is False
            and diagnostic_execution.get("no_update_passed") is True
            and set(diagnostic_execution.get("pairs", {})) == set(METHODS),
            "Native TRAIN diagnostics are incomplete",
        )
        archive_path = metrics / "training_evidence.json"
        archive = read(archive_path)
        require(
            archive["files"]["execution.json"] == digest(training / "execution.json"),
            "Archived training execution differs",
        )
        arms = {}
        for method in METHODS:
            record = execution["methods"][method]
            require(
                record.get("status") == "trained_checkpoint_saved"
                and record.get("returncode") == 0
                and type(record.get("steps")) is int
                and record["steps"] == count,
                "Every arm must record the complete reviewed pass",
            )
            recipe_path = release / profile["recipes"][method]["path"]
            require(
                digest(recipe_path) == profile["recipes"][method]["sha256"],
                "Frozen recipe hash differs",
            )
            recipe = read(recipe_path)
            require(
                recipe["method"] == method
                and recipe["model_id"] == profile["model_id"]
                and recipe["model_revision"] == profile["model_revision"]
                and recipe["tokenizer_revision"] == profile["tokenizer_revision"]
                and recipe["data_manifest_sha256"] == matrix["source_training_binding_sha256"],
                "Executed recipe method/model/source differs",
            )
            receipt_path, history_path = (
                training / method / RECEIPTS[method],
                training / method / HISTORIES[method],
            )
            receipt, history = read(receipt_path), rows(history_path)
            require(
                receipt.get("status")
                == ("trained_not_evaluated" if method == "rmu" else "trained_merged")
                and receipt.get("method") == method
                and receipt.get("steps") == count
                and receipt.get("config_sha256", receipt.get("provenance", {}).get("config_sha256"))
                == digest(recipe_path),
                "Actual method receipt is incomplete or recipe-mismatched",
            )
            require(
                digest(receipt_path) == archive["files"][method + "/" + RECEIPTS[method]]
                and digest(history_path) == archive["files"][method + "/" + HISTORIES[method]],
                "Archived receipt/history differs",
            )
            require(
                # The actual RMU logger increments before recording; the PEFT
                # trainer records its current zero-based step before incrementing.
                len(history) == count
                and all(type(row.get("step")) is int for row in history)
                and [row["step"] for row in history]
                == (list(range(1, count + 1)) if method == "rmu" else list(range(count))),
                "Applied training history does not cover the full pass",
            )
            report_path = metrics / method / "all_metrics.json"
            metric_record = metric_execution["methods"][method]
            require(
                metric_record.get("status") == "analyzed_with_explicit_pending_metrics"
                and metric_record["report"]["sha256"] == digest(report_path),
                "Method all-metrics report did not complete",
            )
            report = read(report_path)
            view_sources = {}
            view = report.get("training_exposure", {}).get("inference_view")
            if view is not None:
                sidecar = view["sidecar"]
                certificate = Path(sidecar["path"]).resolve()
                require(
                    digest(certificate) == sidecar["sha256"], "Inference-view certificate changed"
                )
                source_sha = report["analysis_source_sha256"]["checkpoint_views.py"]
                sources = [
                    Path(path).resolve()
                    for path, sha in metric_execution["inputs_sha256"].items()
                    if path.endswith("/lm_eval/fair_uk/checkpoint_views.py") and sha == source_sha
                ]
                require(
                    len(sources) == 1 and digest(sources[0]) == source_sha,
                    "Maintained inference-view validator source is missing or changed",
                )
                view_sources = {
                    "inference_views": certificate,
                    "inference_views_validator": sources[0],
                }
            require(
                report.get("kind") == "all_18_saved_study_metrics"
                and report.get("model_requests") == 0
                and len(report["metrics"]) == 18
                and {r["id"] for r in report["metrics"]} == set(reports.METRIC_IDS),
                "Full eighteen-entry maintained metric inventory is required",
            )
            require(
                report["intervention"]["method"] == method
                and report["intervention"]["training_language"] == "uk"
                and report["intervention"]["target_values"]
                == {"actor": list(read(binding_path)["training_target"]["actor_values"])},
                "Metric intervention/source actors differ",
            )
            require(
                all(
                    report.get("task_status", {}).get(task, {}).get("status") == "available"
                    and report["task_status"][task].get("phase") == "evaluation"
                    for task in TASKS
                ),
                "All twelve held-out native pairs per arm are required",
            )
            exposure = report["training_exposure"]
            require(
                exposure.get("status") == "available"
                and exposure["optimizer_steps"] == count
                and exposure["log"]["sha256"]
                == archive["files"][method + "/exposure.jsonl"]
                == digest(training / method / "exposure.jsonl"),
                "Actual observed exposure binding differs",
            )
            role = "positive_acquisition" if method == "task_vector" else "forget"
            expected_roles = {role} if method == "task_vector" else {role, "preservation"}
            require(
                set(exposure["summary"]) == expected_roles,
                "Unexpected corrective or missing gradient role",
            )
            for observed in exposure["summary"].values():
                require(
                    observed["events"] == count
                    and set(observed["source_ids"]) == expected_ids
                    and isinstance(observed["steps"], list)
                    and all(type(step) is int for step in observed["steps"])
                    and observed["steps"]
                    == (list(range(1, count + 1)) if method == "rmu" else list(range(count)))
                    and observed["valid_loss_tokens"] > 0,
                    "Every reviewed source ID must have actual exposure in the full pass",
                )
            pair = diagnostic_execution["pairs"][method]
            pair_path = diagnostic / "paired" / method / "paired.json"
            require(
                pair.get("status") == "analyzed"
                and pair.get("returncode") == 0
                and pair["report"]["sha256"] == digest(pair_path),
                "Native training-source paired report differs",
            )
            arms[method] = {
                **view_sources,
                "recipe": recipe_path,
                "receipt": receipt_path,
                "history": history_path,
                "report": report_path,
                "report_data": report,
                "diagnostic_pair": pair_path,
            }
        verified = {
            str(path.resolve()): digest(path)
            for path in [
                training / "execution.json",
                archive_path,
                metrics / "warbias_metrics_execution.json",
                diagnostic / "training_diagnostic_execution.json",
                *(
                    arm[key]
                    for arm in arms.values()
                    for key in ("recipe", "receipt", "history", "report", "diagnostic_pair")
                ),
            ]
        }
        for arm in arms.values():
            for key in ("inference_views", "inference_views_validator"):
                if key in arm:
                    verified[str(arm[key].resolve())] = digest(arm[key])
            for task in TASKS:
                state = arm["report_data"]["task_status"][task]
                for filename, expected in (
                    ("analysis.json", state["analysis_sha256"]),
                    ("validated_pair.json", state["validated_export"]["sha256"]),
                ):
                    path = arm["report"].parent / task / filename
                    require(digest(path) == expected, "Held-out native export hash differs")
                    verified[str(path.resolve())] = expected
        require(
            set(diagnostic_execution["tasks"]) == {"baseline", *METHODS},
            "Four diagnostic checkpoints are required",
        )
        for task in diagnostic_execution["tasks"].values():
            require(task.get("status") == "completed", "A native diagnostic collection failed")
            require(
                set(task["files"]) == {"run.json", "report.json", "predictions.jsonl"},
                "Diagnostic native files are incomplete",
            )
            for name, item in task["files"].items():
                source = Path(item["path"]).resolve()
                require(
                    name in {"run.json", "report.json", "predictions.jsonl"}
                    and source.is_relative_to(diagnostic)
                    and digest(source) == item["sha256"],
                    "Native diagnostic file identity differs",
                )
                verified[str(source)] = item["sha256"]
        contexts[identifier] = {
            "training": training,
            "metrics": metrics,
            "diagnostic": diagnostic,
            "execution": execution,
            "metric_execution": metric_execution,
            "diagnostic_execution": diagnostic_execution,
            "archive": archive_path,
            "arms": arms,
            "verified_files": verified,
        }
    return ready, contexts


def materialize(base_manifest, train_source, measure_source, release, output, ready, contexts):
    public = base_manifest.parent.parent
    base = output / BASE_PATH
    for source in sorted(public.rglob("*")):
        if source.is_file():
            copied(source, base / source.relative_to(public))
        elif source.is_symlink():
            raise ValueError("Source bundle cannot contain symlinks")
    final = base / "reports/manifest_completed.json"
    require(
        not final.exists(),
        "Choose an initial/readiness base manifest, not an existing completed snapshot",
    )
    manifest = copy.deepcopy(read(base_manifest))
    manifest.update(
        schema_version=2,
        as_of_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        parent_manifest=reference(base / "reports" / base_manifest.name, final),
        assembly_kind="complete_actual_warbias_saved_evidence_no_new_model_calls",
    )
    # These are new references; original saved report/config/provenance bytes stay unchanged.
    copied(train_source, base / "reports/assembly_training_submission.json")
    copied(measure_source, base / "reports/assembly_measurement_submission.json")
    manifest["schedule_source"] = reference(
        base / "reports/assembly_training_submission.json", final
    )
    manifest["measurement_source"] = reference(
        base / "reports/assembly_measurement_submission.json", final
    )
    for identifier, context in contexts.items():
        target = base / "evidence" / identifier
        training = target / "training"

        def evidence_copy(source, destination, expected=None):
            verified = context["verified_files"][str(Path(source).resolve())]
            require(expected is None or expected == verified, "Preflight evidence hash differs")
            return copied(source, destination, verified)

        evidence_copy(context["training"] / "execution.json", training / "execution.json")
        evidence_copy(context["archive"], training / "training_evidence.json")
        evidence_copy(
            context["metrics"] / "warbias_metrics_execution.json",
            target / "metrics/warbias_metrics_execution.json",
        )
        archive = read(context["archive"])
        summary = {"kind": "observed_training_exposure_not_outcome_correlation", "methods": {}}
        for method, arm in context["arms"].items():
            receipt = evidence_copy(
                arm["receipt"],
                training / method / RECEIPTS[method],
                archive["files"][method + "/" + RECEIPTS[method]],
            )
            history = evidence_copy(
                arm["history"],
                training / method / HISTORIES[method],
                archive["files"][method + "/" + HISTORIES[method]],
            )
            recipe = base / "recipes" / identifier / (method + ".json")
            require(
                digest(recipe) == digest(arm["recipe"]),
                "Portable planned recipe is not the actual executed recipe",
            )
            report_target = target / "metrics" / method / "all_metrics.json"
            evidence_copy(arm["report"], report_target)
            for task in TASKS:
                state = arm["report_data"]["task_status"][task]
                for name, expected in (
                    ("analysis.json", state["analysis_sha256"]),
                    ("validated_pair.json", state["validated_export"]["sha256"]),
                ):
                    evidence_copy(
                        arm["report"].parent / task / name,
                        report_target.parent / task / name,
                        expected,
                    )
            observed = arm["report_data"]["training_exposure"]
            summary["methods"][method] = {
                "optimizer_steps": observed["optimizer_steps"],
                "sha256": observed["log"]["sha256"],
                "streams": observed["summary"],
                "derivation": "Copied validated maintained all_metrics training_exposure.summary; no performance metric was recalculated.",
            }
            entry = manifest["runs"][identifier][method]
            entry["training"] = {
                "recipe": reference(recipe, final),
                "receipt": reference(receipt, final),
                "loss_log": reference(history, final),
                "execution": reference(training / "execution.json", final),
                "evidence_manifest": reference(training / "training_evidence.json", final),
                "exposure_summary": None,
            }
            for key in ("inference_views", "inference_views_validator"):
                if key in arm:
                    destination = (
                        target
                        / "inference_views"
                        / method
                        / (
                            "certificate.json"
                            if key == "inference_views"
                            else "checkpoint_views.py"
                        )
                    )
                    evidence_copy(arm[key], destination)
                    entry["training"][key] = reference(destination, final)
            entry["analysis"] = reference(report_target, final)
            diagnostic_pair = evidence_copy(
                arm["diagnostic_pair"],
                target / "training_diagnostic/paired" / method / "paired.json",
            )
            entry["training_diagnostic"] = {
                "execution": reference(
                    target / "training_diagnostic/training_diagnostic_execution.json", final
                )
                if (target / "training_diagnostic/training_diagnostic_execution.json").exists()
                else None,
                "comparison": reference(diagnostic_pair, final),
            }
            entry["pending_reason"] = (
                "Native pairs/training completed; individual metric dependencies remain explicit in the maintained report."
            )
        save(training / "observed_exposure_summary.json", summary)
        diagnostic_execution = evidence_copy(
            context["diagnostic"] / "training_diagnostic_execution.json",
            target / "training_diagnostic/training_diagnostic_execution.json",
        )
        for method in METHODS:
            manifest["runs"][identifier][method]["training_diagnostic"]["execution"] = reference(
                diagnostic_execution, final
            )
        for name, record in context["diagnostic_execution"]["tasks"].items():
            for filename, item in record["files"].items():
                evidence_copy(
                    item["path"],
                    target / "training_diagnostic/native" / name / "warbias_triplets_uk" / filename,
                    item["sha256"],
                )
    # Exposure summaries must exist before their new manifest references are hashed.
    for identifier in MODELS:
        for method in METHODS:
            manifest["runs"][identifier][method]["training"]["exposure_summary"] = reference(
                base / "evidence" / identifier / "training/observed_exposure_summary.json", final
            )
    save(final, manifest)
    return final


def assemble(args):
    paths = [
        Path(getattr(args, name)).resolve()
        for name in (
            "base_manifest",
            "training_submission",
            "measurement_submission",
            "training_release",
            "output",
        )
    ]
    base_manifest, train, measurement, release, output = paths
    require(not output.exists(), "Output must be new; preserve previous report bundles")
    require(
        not any(output.is_relative_to(path) for path in (base_manifest.parent.parent, release)),
        "Output overlaps immutable input evidence",
    )
    ready, contexts = preflight(base_manifest, train, measurement, release)
    require(
        not any(
            output.is_relative_to(context[key])
            for context in contexts.values()
            for key in ("training", "metrics", "diagnostic")
        ),
        "Output overlaps original execution evidence",
    )
    output.mkdir(parents=True)
    execution_path = output / "assembly_execution.json"
    status = {
        "kind": "saved_only_warbias_report_assembly",
        "stage": "copying_verified_evidence",
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model_calls": 0,
        "weights_copied": False,
        "exposure_token_logs_copied": False,
        "inputs": {
            str(p): digest(p)
            for p in (base_manifest, train, measurement, release / "checksums.json")
        },
        "source_sha256": {
            name: digest(Path(__file__).with_name(name))
            for name in (
                "assemble_warbias_report.py",
                "unlearning_report_data.py",
                "build_unlearning_report_notebooks.py",
            )
        },
    }
    save(execution_path, status)
    try:
        for name in status["source_sha256"]:
            copied(Path(__file__).with_name(name), output / "scripts" / name)
        final = materialize(base_manifest, train, measurement, release, output, ready, contexts)
        validated_reports = {}
        for method in METHODS:
            result = reports.load_manifest(final, method)
            require(
                len(result["coverage"]) == 6
                and all(
                    row["training"] == "actual_completed"
                    and row["actual_steps"] == ready["preparation"]["counts"]["forget_train"]
                    and row["measurement_status"] == "complete"
                    for row in result["coverage"]
                ),
                "Complete six-model actual evidence did not replay",
            )
            require(
                len(result["metric_status"]) == 108
                and len(result["training_source_coverage"]) == 6
                and all(row["status"] == "available" for row in result["training_source_coverage"]),
                "Metric inventory or training-source diagnostics did not replay",
            )
            validated_reports[method] = result
        status["tables"] = export_tables(output, validated_reports)
        status.update(
            stage="executing_saved_report_replay",
            manifest={"path": str(final.relative_to(output)), "sha256": digest(final)},
        )
        save(execution_path, status)
        command = [
            sys.executable,
            str(output / "scripts/build_unlearning_report_notebooks.py"),
            "--repo",
            str(output),
            "--study",
            "warbias",
            "--manifest",
            str(final),
            "--kernel-name",
            args.kernel_name,
        ]
        if args.execute:
            command.append("--execute")
        with (output / "report_execution.log").open("x") as log:
            run = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        require(run.returncode == 0, "Notebook replay failed; preserve report_execution.log")
        notebooks = list(
            (output / "notebooks" / ("executed" if args.execute else "not_executed")).glob(
                "warbias_*_results.ipynb"
            )
        )
        require(len(notebooks) == 3, "Exactly three result notebooks are required")
        for path in notebooks:
            document = read(path)
            if args.execute:
                metadata = document["metadata"]["fairforget"]
                require(
                    metadata["replay_manifest_sha256"] == digest(final)
                    and all(
                        not any(o.get("output_type") == "error" for o in c.get("outputs", []))
                        for c in document["cells"]
                    ),
                    "Executed notebook provenance/error gate failed",
                )
        status.update(
            stage="complete_actual_warbias_reports",
            finished_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            training_arms=18,
            training_source_pairs=18,
            heldout_native_pairs=216,
            metric_inventory_entries=324,
            notebooks_executed=args.execute,
            kernel_name=args.kernel_name,
            scientific_calibration_inferred=False,
            notebooks={str(p.relative_to(output)): digest(p) for p in notebooks},
        )
        if args.execute:
            status["report_runtime_versions"] = {
                name: importlib.metadata.version(name)
                for name in ("nbformat", "nbclient", "pandas", "matplotlib")
            }
        save(execution_path, status)
        (output / "README.md").write_text(
            "# Complete saved WarBias reports\n\nThis bundle contains 18 actual training arms, 18 native training-source comparisons and 216 held-out native pairs across six models and three methods. All eighteen metric commitments per arm remain visible; missing judging, calibration and support are not promoted to qualified outcomes.\n\nThe three notebooks are under notebooks/"
            + ("executed" if args.execute else "not_executed")
            + ". Replays use local saved evidence only. The tables/ directory contains UTF-8 CSV exports of the validated held-out results, training-source comparisons, metric dependency inventory, exposure, parameters and coverage for every method. Missing numeric values stay blank with their original statuses; nested native scopes and intervals are preserved as JSON cells. These exports calculate no new metrics and are included in checksums.json. Original native JSON/provenance paths are preserved without rewriting; portable manifest references locate copied artifacts. Model weights, evaluator source and exposure token logs are not copied. Their recorded hashes and maintained observed-exposure summaries remain available.\n"
        )
        checksums = {
            str(p.relative_to(output)): digest(p) for p in sorted(output.rglob("*")) if p.is_file()
        }
        save(output / "checksums.json", checksums)
        return status
    except Exception as error:
        status.update(
            stage="assembly_or_replay_failed", error_type=type(error).__name__, error=str(error)
        )
        save(execution_path, status)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in (
        "base-manifest",
        "training-submission",
        "measurement-submission",
        "training-release",
        "output",
    ):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--kernel-name", default="python3")
    args = parser.parse_args(argv)
    try:
        result = assemble(args)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        print(json.dumps({"stage": "report_assembly_failed", "error": str(error)}), file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "stage": result["stage"],
                "output": str(args.output.resolve()),
                "notebooks_executed": args.execute,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
