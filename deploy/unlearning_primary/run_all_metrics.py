"""Run a frozen saved-only Fair-UK package after both collections complete."""

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

METHODS = ("wga_kl", "npo_kl", "rmu", "biasunlearn", "task_vector")
METRIC_IDS = frozenset(
    "forget_efficacy NNR retain_utility LMS ICAT M1 M1b M2 M3a M3b M3c M3d RI M4a M4b BBQ training_imbalance severity".split()
)
TASK_IDS = frozenset(
    "bbq_uk stereoset_en stereoset_uk warbias_benign_en warbias_benign_uk warbias_cross_actor_en warbias_cross_actor_uk warbias_en warbias_expanded_qa_en warbias_expanded_qa_uk warbias_intersectional_en warbias_intersectional_uk warbias_triplets_en warbias_triplets_uk warbias_uk winobias_uk_controlled winobias_uk_natural winopron_uk".split()
)
STAGES = {
    "native": "native_collection_complete",
    "supplemental": "supplemental_collection_complete",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def checked_inputs(package):
    root = package.parent.resolve()
    plan = read(package)
    require(plan.get("schema_version") == 1, "Unsupported analysis package")
    require(set(plan["configs"]) == set(METHODS), "All five method configs are required")
    configs, evidence = {}, {}
    for method, entry in plan["configs"].items():
        path = (root / entry["filename"]).resolve()
        require(root in path.parents, "Config escapes the package")
        require(digest(path) == entry["sha256"], "Frozen config changed")
        config = read(path)
        require(config["intervention"]["method"] == method, "Config method differs")
        require(set(config["required_tasks"]) == TASK_IDS, "Config task inventory differs")
        configs[method] = path
    for label, entry in plan["collections"].items():
        require(
            label in STAGES and entry["required_stage"] == STAGES[label],
            "Invalid required collection stage",
        )
        path = Path(entry["path"])
        record = read(path)
        require(record.get("stage") == entry["required_stage"], label + ": collection incomplete")
        if label == "native":
            require(record.get("no_update", {}).get("passed") is True, "Native gate failed")
        elif label == "supplemental":
            require(record.get("no_update_passed") is True, "Supplemental gate failed")
        else:
            raise ValueError("Unknown collection role")
        evidence[label] = {"path": str(path), "sha256": digest(path)}
    require(set(evidence) == {"native", "supplemental"}, "Both collections are required")
    return configs, evidence


def verify_report(report, expected_tasks):
    require(
        report.get("schema_version") == 1 and report.get("kind") == "all_18_saved_study_metrics",
        "Unexpected report schema",
    )
    require(report.get("model_requests") == 0, "Analysis made model requests")
    require(len(report.get("metrics", [])) == 18, "Incomplete metric inventory")
    require(len({row["id"] for row in report["metrics"]}) == 18, "Duplicate metric entry")
    require({row["id"] for row in report["metrics"]} == METRIC_IDS, "Metric inventory differs")
    require(set(expected_tasks) == TASK_IDS, "Expected task inventory differs")
    require(
        set(report["task_status"]) == set(expected_tasks), "Task report differs from frozen config"
    )
    for task, row in report["task_status"].items():
        expected = "blocked" if task == "stereoset_en" else "available"
        require(row["status"] == expected, task + ": saved analysis incomplete")
    require(len(report["task_status"]) == 18, "Full task inventory is required")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("package", "evaluator", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        require(not args.output.exists(), "Use a new analysis output directory")
        root, output = args.package.parent.resolve(), args.output.resolve()
        require(root != output and root not in output.parents, "Output overlaps frozen inputs")
        configs, inputs = checked_inputs(args.package.resolve())
        require(
            all(Path(entry["path"]).parent not in output.parents for entry in inputs.values()),
            "Output overlaps a saved collection",
        )
        require(
            (args.evaluator / "lm_eval/fair_uk/all_metrics.py").is_file(), "Analysis source missing"
        )
        output.mkdir(parents=True)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"stage": "preflight_failed", "error": str(error)}), file=sys.stderr)
        return 2
    status = {
        "stage": "saved_only_analysis",
        "model_requests": 0,
        "methods": {},
        "workers": 5,
        "collections": inputs,
        "package": {"path": str(args.package.resolve()), "sha256": digest(args.package)},
        "driver_sha256": digest(__file__),
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    receipt = output / "all_metrics_execution.json"
    save(receipt, status)
    env = {**os.environ, "PYTHONPATH": str(args.evaluator.resolve()), "CUDA_VISIBLE_DEVICES": ""}
    for name in (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "PYTHONDONTWRITEBYTECODE",
    ):
        env[name] = "1"
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[name] = "1"
    lock = threading.Lock()

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
        record = {"status": "running", "command": command, "config_sha256": digest(configs[method])}
        with lock:
            status["methods"][method] = record
            save(receipt, status)
        try:
            with (output / (method + ".log")).open("x") as log:
                result = subprocess.run(
                    command,
                    env=env,
                    cwd=args.evaluator,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            with lock:
                record["returncode"] = result.returncode
                save(receipt, status)
            require(result.returncode == 0, "Fair-UK analysis failed")
            path = output / method / "all_metrics.json"
            verify_report(read(path), read(configs[method])["required_tasks"])
            require(
                digest(configs[method]) == record["config_sha256"], "Config changed during analysis"
            )
            require(
                all(digest(entry["path"]) == entry["sha256"] for entry in inputs.values()),
                "Collection receipt changed during analysis",
            )
            outcome = dict(
                status="analyzed",
                returncode=result.returncode,
                report={"path": str(path), "sha256": digest(path)},
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            outcome = dict(status="failed", error=str(error))
        with lock:
            record.update(outcome)
            save(receipt, status)

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        list(pool.map(worker, METHODS))
    complete = all(record["status"] == "analyzed" for record in status["methods"].values())
    status.update(
        stage="all_metrics_reports_complete" if complete else "all_metrics_partial_failure",
        finished_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
    )
    save(receipt, status)
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
