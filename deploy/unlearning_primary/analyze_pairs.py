"""Analyze completed saved native runs through the maintained Fair-UK CLI."""

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

TASKS = tuple(
    "stereoset_uk bbq_uk winobias_uk_natural winobias_uk_controlled "
    "winopron_uk warbias_triplets_uk".split()
)
METHODS = ("wga_kl", "npo_kl", "biasunlearn", "task_vector", "rmu")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def event(path):
    return {"path": str(path), "sha256": digest(path)}


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def command_for(release, evaluation, output, method, task):
    command = [sys.executable, "-m", "lm_eval.fair_uk", "paper", "compare-native"]
    for flag, value in (
        ("before-run", evaluation / "baseline" / task),
        ("after-run", evaluation / method / task),
        ("dataset-bundle", release / "evaluation/manifest.json"),
        ("output", output / method / task),
        ("bootstrap", 1000),
        ("seed", 42),
    ):
        command.extend(["--" + flag, str(value)])
    return command


def verify_native(evaluation, queue, task, manifest_path, entry, model, checkpoint):
    key = model + "/" + task
    queued = queue["tasks"].get(key, {})
    require(
        queued.get("status") == "collected" and queued.get("returncode") == 0,
        key + ": native queue pair unavailable or failed",
    )
    require(
        Path(queue["checkpoints"].get(model, "")).resolve() == checkpoint,
        key + ": evaluated checkpoint differs from training",
    )
    status_path = evaluation / model / "status" / (task + ".json")
    status = read(status_path)
    expected = dict(status="completed", exit_code=0, collector_exit_code=0, task=task)
    expected.update(full_primary_evaluation_bundle=True, manifest_sha256=digest(manifest_path))
    size = entry["rows"]
    expected.update(source_sha256=entry["sha256"], expected_rows=size, prediction_rows=size)
    require(all(status.get(k) == v for k, v in expected.items()), key + ": native receipt differs")
    require(
        Path(status.get("manifest_path", "")).resolve() == manifest_path
        and Path(status.get("checkpoint", "")).resolve() == checkpoint,
        key + ": native receipt source or checkpoint path differs",
    )
    native = status.get("native_files_sha256", {})
    evidence = {"receipt": event(status_path)}
    for filename in ("run.json", "report.json", "predictions.jsonl"):
        item = event(evaluation / model / task / filename)
        require(item["sha256"] == native.get(filename), key + ": native file hash differs")
        evidence[filename] = item
    return evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("release", "training", "evaluation", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    release, training, evaluation, output = (
        getattr(args, name).resolve() for name in ("release", "training", "evaluation", "output")
    )
    try:
        require(not output.exists(), "Use a new analysis output directory")
        require(
            all(p != output and p not in output.parents for p in (release, training, evaluation)),
            "Output must stay outside saved inputs",
        )
        train_path = training / "execution.json"
        queue_path = evaluation / "evaluation_execution.json"
        train, queue = read(train_path), read(queue_path)
        training_done = train.get("stage") in ("training_complete", "training_partial_failure")
        require(training_done, "Training queue has not reached a final state")
        final_states = ("native_collection_complete", "native_collection_partial_failure")
        require(
            queue.get("stage") in final_states, "Evaluation queue has not reached a final state"
        )
        proof_path = evaluation / "no_update_check.json"
        require(
            queue.get("no_update", {}).get("passed") is True
            and read(proof_path) == queue["no_update"],
            "No-update qualification differs or failed",
        )
        manifest_path = release / "evaluation/manifest.json"
        manifest = read(manifest_path)
        require(manifest.get("schema_version") == 1, "Unsupported evaluation manifest")
        baseline = Path(read(release / "recipes/wga_kl.json")["model_path"]).resolve()
        for filename in ("paper.py", "paired.py"):
            path = release / "evaluator/lm_eval/fair_uk" / filename
            require(path.is_file(), "Frozen maintained evaluator is missing " + filename)
        sources = {}
        for task in TASKS:
            entry = manifest["datasets"][task]
            path = (manifest_path.parent / entry["filename"]).resolve()
            require(
                manifest_path.parent in path.parents and digest(path) == entry["sha256"],
                task + ": frozen source checksum or path differs",
            )
            sources[task] = event(path)
        inputs = {
            name: event(path)
            for name, path in (
                ("training", train_path),
                ("evaluation", queue_path),
                ("no_update", proof_path),
                ("bundle", manifest_path),
                ("base_recipe", release / "recipes/wga_kl.json"),
            )
        }
        output.mkdir(parents=True, exist_ok=False)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"stage": "preflight_failed", "error": str(error)}), file=sys.stderr)
        return 2

    env = {**os.environ, "PYTHONPATH": str(release / "evaluator"), "CUDA_VISIBLE_DEVICES": ""}
    for name in "OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS NUMEXPR_NUM_THREADS".split():
        env[name] = "1"
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        env[name] = "1"
    env.update(PYTHONDONTWRITEBYTECODE="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1")
    status = dict(stage="analysis", inputs=inputs, sources=sources, pairs={})
    status.update(driver=event(Path(__file__).resolve()), workers=4, saved_only=True)
    status.update(
        bootstrap=1000, seed=42, started_at_utc=dt.datetime.now(dt.timezone.utc).isoformat()
    )
    lock = threading.Lock()
    status_path = output / "analysis_execution.json"
    save(status_path, status)

    def worker(pair):
        method, task = pair
        key, record = method + "/" + task, {"status": "skipped"}
        try:
            arm = train["methods"].get(method, {})
            require(
                arm.get("status") == "trained_checkpoint_saved" and arm.get("returncode") == 0,
                "Training arm did not save a successful checkpoint",
            )
            checkpoint = Path(arm["checkpoint"]).resolve()
            receipt = Path(arm["receipt"]).resolve()
            require(
                training in receipt.parents, "Training receipt is outside the selected execution"
            )
            record["training_receipt"] = event(receipt)
            native_args = (evaluation, queue, task, manifest_path, manifest["datasets"][task])
            record["inputs"] = {
                "before": verify_native(*native_args, "baseline", baseline),
                "after": verify_native(*native_args, method, checkpoint),
            }
        except (OSError, ValueError, KeyError, TypeError) as error:
            record["reason"] = str(error)
        else:
            command = command_for(release, evaluation, output, method, task)
            record.update(status="running", command=command)
            with lock:
                status["pairs"][key] = dict(record)
                save(status_path, status)
            try:
                log = output / (method + "-" + task + ".log")
                record["log"] = str(log)
                with log.open("x") as stream:
                    result = subprocess.run(
                        command,
                        cwd=release / "evaluator",
                        env=env,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                record["returncode"] = result.returncode
                require(result.returncode == 0, "Native compare-native process failed")
                report_path = output / method / task / "paired.json"
                report = read(report_path)
                require(
                    report.get("kind") == "paired_native_checkpoint_evaluation"
                    and report.get("task") == task,
                    "Native paired report identity differs",
                )
                evidence = [*inputs.values(), sources[task], record["training_receipt"]]
                for phase in ("before", "after"):
                    evidence.extend(record["inputs"][phase].values())
                    require(
                        report["provenance"][phase]["predictions_sha256"]
                        == record["inputs"][phase]["predictions.jsonl"]["sha256"],
                        "Paired report prediction bytes differ from the native receipt",
                    )
                require(
                    all(digest(item["path"]) == item["sha256"] for item in evidence),
                    "Saved input evidence changed during analysis",
                )
                record.update(status="analyzed", report=event(report_path))
            except (OSError, ValueError, KeyError, TypeError) as error:
                record.update(status="failed", reason=str(error))
        with lock:
            status["pairs"][key] = record
            save(status_path, status)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(worker, ((method, task) for method in METHODS for task in TASKS)))
    complete = all(record["status"] == "analyzed" for record in status["pairs"].values())
    status.update(
        stage="analysis_complete" if complete else "analysis_partial_failure",
        finished_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
    )
    save(status_path, status)
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
