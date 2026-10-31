"""Qualify each saved checkpoint twice before collecting locked native panels."""

import argparse
import concurrent.futures
import datetime
import hashlib
import json
import math
import os
import queue
import subprocess
import threading
from pathlib import Path

TOLERANCE = 1e-6


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def evaluate(release, output, name, checkpoint, root, task, gpu, qualification=False):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
    env.pop("FAIRFORGET_EVAL_BUNDLE", None)
    if qualification:
        env["FAIRFORGET_EVAL_BUNDLE"] = str(release / "qualification/manifest.json")
    wrapper = release / "deploy/unlearning_primary/evaluate_checkpoint.sh"
    command = ["bash", str(wrapper), str(release), str(checkpoint), str(root), task]
    with (output / (name + "-" + task + ".log")).open("w") as log:
        return subprocess.run(
            command, env=env, stdout=log, stderr=subprocess.STDOUT, check=False
        ).returncode


def qualification_inputs(release):
    manifest_path = release / "qualification/manifest.json"
    payload = manifest_path.read_bytes()
    manifest = json.loads(payload)
    entry = manifest["datasets"]["stereoset_uk"]
    source = (manifest_path.parent / entry["filename"]).resolve()
    if manifest.get("schema_version") != 1 or manifest_path.parent not in source.parents:
        raise ValueError("Qualification bundle schema or source path differs")
    data = source.read_bytes()
    if hashlib.sha256(data).hexdigest() != entry["sha256"] or entry["rows"] != 2:
        raise ValueError("Qualification requires the exact two frozen development rows")
    ids = [json.loads(line)["id"] for line in data.splitlines() if line.strip()]
    if len(ids) != 2 or any(not isinstance(item, str) or not item for item in ids):
        raise ValueError("Qualification requires exactly two nonempty string IDs")
    if len(set(ids)) != 2:
        raise ValueError("Qualification development IDs must be unique")
    return ids, hashlib.sha256(payload).hexdigest()


def qualification_predictions(path, expected_ids):
    payload = path.read_bytes()
    rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
    ids = [row["id"] for row in rows]
    if len(ids) != 2 or len(set(ids)) != 2 or ids != expected_ids:
        raise ValueError(
            "Qualification predictions require the exact two ordered unique source IDs"
        )
    for row in rows:
        scores = row.get("scores")
        if not isinstance(scores, list) or len(scores) != 3:
            raise ValueError("Qualification requires exactly three candidate scores per item")
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in scores):
            raise ValueError("Qualification candidate scores must be finite numbers")
    return rows, hashlib.sha256(payload).hexdigest()


def qualify_checkpoint(release, output, name, checkpoint, gpu, expected_ids):
    proof = {"passed": False, "checkpoint": str(checkpoint), "gpu": gpu, "runs": []}
    records = []
    for index in (1, 2):
        root = output / "qualification" / name / ("repeat-" + str(index))
        run = {"repeat": index, "output": str(root)}
        try:
            run["returncode"] = evaluate(
                release,
                output,
                "no-update-" + name + "-" + str(index),
                checkpoint,
                root,
                "stereoset_uk",
                gpu,
                True,
            )
            if run["returncode"] != 0:
                raise RuntimeError("Development-only collector failed")
            rows, checksum = qualification_predictions(
                root / "stereoset_uk/predictions.jsonl", expected_ids
            )
            records.append(rows)
            run["predictions_sha256"] = checksum
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            run["error"] = str(error)
        proof["runs"].append(run)
    if any("error" in run for run in proof["runs"]):
        proof["error"] = "One or both independent development collections failed validation"
        proof["max_absolute_score_difference"] = None
        return proof
    maximum = max(
        abs(a - b)
        for first, second in zip(*records)
        for a, b in zip(first["scores"], second["scores"])
    )
    proof.update(passed=maximum <= TOLERANCE, max_absolute_score_difference=maximum)
    if not proof["passed"]:
        proof["error"] = "Development-only no-update numerical tolerance exceeded"
    return proof


def qualify_all(release, output, models, gpus, expected_ids, manifest_sha256):
    pending = queue.Queue()
    for name, checkpoint in models.items():
        pending.put((name, checkpoint))
    proof = {
        "passed": False,
        "max_absolute_score_difference": None,
        "absolute_tolerance": TOLERANCE,
        "scope": "two reviewed development items per checkpoint, not locked final evaluation",
        "qualification_manifest_sha256": manifest_sha256,
        "ordered_ids": expected_ids,
        "checkpoints": {},
    }
    lock = threading.Lock()
    path = output / "no_update_check.json"
    write_json(path, proof)

    def worker(gpu):
        while True:
            try:
                name, checkpoint = pending.get_nowait()
            except queue.Empty:
                return
            record = qualify_checkpoint(release, output, name, checkpoint, gpu, expected_ids)
            with lock:
                proof["checkpoints"][name] = record
                write_json(path, proof)
            pending.task_done()

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(worker, gpus))
    proof["passed"] = all(record["passed"] for record in proof["checkpoints"].values())
    proof["failed_checkpoints"] = [
        name for name, r in proof["checkpoints"].items() if not r["passed"]
    ]
    differences = [r["max_absolute_score_difference"] for r in proof["checkpoints"].values()]
    proof["max_absolute_score_difference"] = max(
        (d for d in differences if d is not None), default=None
    )
    write_json(path, proof)
    return proof


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("release", "training", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    release, output = args.release.resolve(), args.output.resolve()
    try:
        if output.exists():
            raise FileExistsError(output)
        training = json.loads((args.training / "execution.json").read_text())
        models = {
            "baseline": Path(
                json.loads((release / "recipes/wga_kl.json").read_text())["model_path"]
            )
        }
        for name, record in training["methods"].items():
            if record["status"] == "trained_checkpoint_saved":
                models[name] = Path(record["checkpoint"])
        if len(models) < 2:
            raise ValueError("No trained checkpoint is available; no locked collection started")
        gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if len(gpus) != 4 or any(not gpu for gpu in gpus):
            raise ValueError("Expected four allocated GPUs")
        expected_ids, manifest_sha256 = qualification_inputs(release)
        output.mkdir(parents=True)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"stage": "preflight_failed", "error": str(error)}))
        return 2
    proof = qualify_all(release, output, models, gpus, expected_ids, manifest_sha256)
    if not proof["passed"]:
        print("Checkpoint qualification failed; no locked collection started", flush=True)
        return 1

    tasks = queue.Queue()
    # Larger panels start first; short panels fill otherwise free workers.
    native = [
        "bbq_uk",
        "winopron_uk",
        "stereoset_uk",
        "winobias_uk_natural",
        "winobias_uk_controlled",
        "warbias_triplets_uk",
    ]
    for task in native:
        for name, checkpoint in models.items():
            tasks.put((name, checkpoint, task))
    status = {
        "stage": "native_collection",
        "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "checkpoints": {key: str(value) for key, value in models.items()},
        "no_update": proof,
        "tasks": {},
    }
    lock = threading.Lock()
    path = output / "evaluation_execution.json"
    write_json(path, status)

    def worker(gpu):
        while True:
            try:
                name, checkpoint, task = tasks.get_nowait()
            except queue.Empty:
                return
            key = name + "/" + task
            with lock:
                status["tasks"][key] = {"status": "running", "gpu": gpu}
                write_json(path, status)
            code = evaluate(release, output, name, checkpoint, output / name, task, gpu)
            with lock:
                status["tasks"][key] = {
                    "status": "collected" if code == 0 else "failed",
                    "returncode": code,
                }
                write_json(path, status)
            tasks.task_done()

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(worker, gpus))
    status["stage"] = (
        "native_collection_complete"
        if all(record["status"] == "collected" for record in status["tasks"].values())
        else "native_collection_partial_failure"
    )
    status["finished_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    write_json(path, status)
    return 0 if status["stage"] == "native_collection_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
